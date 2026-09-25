"""
Handwritten text recognition with TrOCR, run through ONNX Runtime.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

TrOCR is a vision-encoder-decoder: a ViT reads a line image into 577 patch
embeddings, and a RoBERTa-style decoder generates the transcription one token
at a time. This module runs both halves as ONNX graphs and implements the
greedy decode and the byte-level BPE detokenizer directly.

WHY ONNX AND NOT PyTorch
------------------------
PyTorch installs on this machine but cannot LOAD: Windows Application Control
blocks torch/lib/shm.dll (WinError 4551). Installing it also broke spaCy,
because thinc imports torch eagerly when it sees it. onnxruntime is a
separate binary and loads without complaint, so the model runs here as an
exported graph even though the framework it was trained in will not start.

The weights are the ones published for transformers.js
(Xenova/trocr-base-handwritten), int8-quantised: 88 MB encoder, 249 MB
decoder, against 1.3 GB for the float32 pair.

THE LIMITATION THAT MATTERS MOST
--------------------------------
TrOCR's handwritten checkpoints are trained on IAM, which is ENGLISH
handwriting in the Latin alphabet. Shown a Patwari's Hindi entry it will not
fail loudly - it will emit confident English words. That is precisely the
failure mode this project exists to prevent, so:

  * every transcription is returned with `script_supported` saying whether
    the crop was Latin at all, and
  * a transcription is NEVER auto-accepted. It is a suggestion for a human,
    on the same footing as the cloud-LLM suggestions in llm_extractor.py.

For Devanagari handwriting no usable PRETRAINED model exists here. What is
missing is training, not capacity:

The tokenizer is byte-level BPE, so it carries all 256 single-byte tokens and
Devanagari IS representable - 'नरहरपुर' round-trips exactly through this
vocabulary, as 21 byte-tokens for 7 characters. It is inefficient (roughly
three tokens per character, with no learned merges to compress them) but it
is not impossible. An earlier version of this comment claimed Devanagari was
"unrepresentable", which was simply wrong and would have discouraged the one
fix that can work.

So fine-tuning this architecture on an Indic handwriting corpus is a viable
route, and the resulting model exports to ONNX and drops into this module
without changing the loader, the decode loop, the detokenizer or the script
guard. See SUPPORTED_SCRIPTS - that set is the only thing that needs to grow.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

try:
    import numpy as _np
except Exception:                                    # pragma: no cover
    _np = None
try:
    import onnxruntime as _ort
except Exception:                                    # pragma: no cover
    _ort = None
try:
    import cv2 as _cv2
except Exception:                                    # pragma: no cover
    _cv2 = None

ONNX_AVAILABLE = _ort is not None and _np is not None

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "..", "storage", "models", "trocr")
# The ENCODER must be full precision; the DECODER may be int8.
#
# Measured on the canonical IAM fixture (a cursive "industrie"):
#   int8 encoder + int8 decoder -> "insalums true"   (unusable)
#   fp32 encoder + int8 decoder -> "indus the"
#   fp32 encoder + fp32 decoder -> "indus the"       (identical)
#
# So int8 quantisation of the VISION tower destroys the reading, while the
# same quantisation of the language decoder costs nothing measurable. Keeping
# the decoder at int8 saves 740 MB for no loss. Had this not been checked,
# the obvious choice - quantise both, they are the same model - would have
# shipped a recogniser that emits fluent nonsense.
ENCODER_FILE = "encoder_model.onnx"
DECODER_FILE = "decoder_model_quantized.onnx"
TOKENIZER_FILE = "tokenizer.json"

# From the model's config.json. Start and end are the same id, which is
# normal for this family: generation begins with </s> and ends when </s> is
# produced again.
DECODER_START_TOKEN = 2
EOS_TOKEN = 2
PAD_TOKEN = 1
IMAGE_SIZE = 384
MAX_NEW_TOKENS = 48

# ViT image normalisation: scale to [0,1] then (x - 0.5) / 0.5, i.e. [-1,1].
_IMAGE_MEAN = 0.5
_IMAGE_STD = 0.5


def _bytes_to_unicode() -> Dict[int, str]:
    """
    The GPT-2 byte-to-unicode table that byte-level BPE is written in.

    Byte-level BPE does not store text, it stores BYTES re-encoded as
    printable codepoints so that every possible byte has a visible glyph.
    Decoding therefore has to run this mapping backwards before the UTF-8
    decode, or every non-ASCII character comes out as mojibake.
    """
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    mapped = printable[:]
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {b: chr(c) for b, c in zip(printable, mapped)}


class Tokenizer:
    """Just enough of the tokenizer to DECODE ids back into text."""

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        vocab = (payload.get("model") or {}).get("vocab") or {}
        self.id_to_token: Dict[int, str] = {int(v): k for k, v in vocab.items()}
        for added in payload.get("added_tokens", []):
            self.id_to_token[int(added["id"])] = added["content"]
        self.specials = {a["content"] for a in payload.get("added_tokens", [])}
        self._byte_decoder = {c: b for b, c in _bytes_to_unicode().items()}

    def decode(self, ids: List[int]) -> str:
        pieces = []
        for i in ids:
            token = self.id_to_token.get(int(i))
            if token is None or token in self.specials:
                continue
            pieces.append(token)
        text = "".join(pieces)
        raw = bytes(self._byte_decoder.get(ch, ord("?") if ord(ch) < 256 else 63)
                    for ch in text)
        return raw.decode("utf-8", errors="replace").strip()


def available(model_dir: str = MODEL_DIR) -> bool:
    if not ONNX_AVAILABLE:
        return False
    return all(os.path.exists(os.path.join(model_dir, f))
               for f in (ENCODER_FILE, DECODER_FILE, TOKENIZER_FILE))


class TrOCR:
    """Lazily-loaded encoder/decoder pair plus its tokenizer."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._encoder = None
        self._decoder = None
        self._tokenizer: Optional[Tokenizer] = None

    def _load(self) -> None:
        if self._encoder is not None:
            return
        options = _ort.SessionOptions()
        # One thread per core is the wrong default here: a line crop is a
        # small graph and the thread pool costs more than it saves when many
        # crops are run back to back.
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        provider = ["CPUExecutionProvider"]
        self._encoder = _ort.InferenceSession(
            os.path.join(self.model_dir, ENCODER_FILE), options, providers=provider)
        self._decoder = _ort.InferenceSession(
            os.path.join(self.model_dir, DECODER_FILE), options, providers=provider)
        self._tokenizer = Tokenizer(os.path.join(self.model_dir, TOKENIZER_FILE))

    # ---------------------------------------------------------------- input

    @staticmethod
    def preprocess(gray):
        """
        One line crop as the (1, 3, 384, 384) tensor the ViT expects.

        The aspect ratio is NOT preserved, deliberately: TrOCR was trained on
        line crops squashed to a square, so letterboxing them here - which
        looks more respectful of the image - would present the model with
        geometry it never saw and degrade the transcription.
        """
        if _cv2 is None or _np is None:
            raise RuntimeError("OpenCV/numpy required for TrOCR preprocessing.")
        if gray is None or getattr(gray, "size", 0) == 0:
            raise ValueError("empty crop")
        if gray.ndim == 2:
            rgb = _cv2.cvtColor(gray, _cv2.COLOR_GRAY2RGB)
        else:
            rgb = _cv2.cvtColor(gray, _cv2.COLOR_BGR2RGB)
        resized = _cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE),
                              interpolation=_cv2.INTER_LINEAR)
        array = resized.astype(_np.float32) / 255.0
        array = (array - _IMAGE_MEAN) / _IMAGE_STD
        return _np.transpose(array, (2, 0, 1))[_np.newaxis, ...]

    # -------------------------------------------------------------- decode

    def transcribe(self, gray, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
        """
        Read one line crop.

        Greedy decoding, and without a key/value cache: the decoder graph is
        re-run over the whole prefix at each step. That is O(n^2) where a
        cached decoder is O(n), and for a line of a few dozen tokens it costs
        little while removing a large amount of state-plumbing that could be
        silently wrong. Correctness first; the cached graph is available in
        the same repository if throughput ever matters.
        """
        self._load()
        pixel_values = self.preprocess(gray)
        encoder_states = self._encoder.run(
            ["last_hidden_state"], {"pixel_values": pixel_values})[0]

        ids: List[int] = [DECODER_START_TOKEN]
        scores: List[float] = []
        for _ in range(max_new_tokens):
            logits = self._decoder.run(
                ["logits"],
                {"input_ids": _np.array([ids], dtype=_np.int64),
                 "encoder_hidden_states": encoder_states})[0]
            step = logits[0, -1]
            # Softmax only over the winning logit's neighbourhood is enough
            # for a confidence figure, but the full one is cheap and honest.
            shifted = step - step.max()
            probabilities = _np.exp(shifted)
            probabilities /= probabilities.sum()
            token = int(step.argmax())
            scores.append(float(probabilities[token]))
            if token == EOS_TOKEN:
                break
            ids.append(token)

        text = self._tokenizer.decode(ids[1:])
        # Mean token probability. This is the MODEL's certainty about its own
        # output and says nothing about whether the output is right - on
        # Devanagari it is routinely high and completely wrong.
        confidence = float(sum(scores) / len(scores)) if scores else 0.0
        return {
            "text": text,
            "confidence": round(confidence, 4),
            "tokens": len(ids) - 1,
            "truncated": len(ids) - 1 >= max_new_tokens,
        }


_INSTANCE: Optional[TrOCR] = None


def _engine(model_dir: str = MODEL_DIR) -> Optional[TrOCR]:
    global _INSTANCE
    if not available(model_dir):
        return None
    if _INSTANCE is None or _INSTANCE.model_dir != model_dir:
        _INSTANCE = TrOCR(model_dir)
    return _INSTANCE


def reset_cache() -> None:
    global _INSTANCE
    _INSTANCE = None


def transcribe(gray, model_dir: str = MODEL_DIR) -> Optional[dict]:
    """
    Transcribe one handwritten line crop, or None when unavailable.

    None means "no transcription was attempted" and must never be shown as an
    empty reading.
    """
    engine = _engine(model_dir)
    if engine is None:
        return None
    try:
        return engine.transcribe(gray)
    except Exception as exc:
        return {"text": "", "confidence": 0.0, "error": str(exc)}


# Scripts this checkpoint can actually read. TrOCR's handwritten weights are
# IAM-trained, so the decoder's vocabulary is English BPE and Devanagari is
# not merely unseen - it is unrepresentable.
SUPPORTED_SCRIPTS = {"latin"}


# TrOCR keeps generating after the content ends on a short crop. Measured:
# "Ramesh Kumar Yadav 1991", "4474.", "1/2NISSA.000652" - the reading is
# right and then a tail of invented tokens follows it. The model was trained
# on IAM lines that fill their crop, so a three-character value gives it far
# more decode budget than there is ink to justify.
#
# The bound is the crop's own aspect ratio: handwriting averages roughly half
# a character per crop-height of width, so a crop 8 heights wide cannot hold
# thirty characters. This trims the tail without touching the reading.
_CHARS_PER_HEIGHT = 2.2
_MIN_EXPECTED_CHARS = 3


def expected_max_chars(gray) -> int:
    """How many characters a crop of this shape could plausibly hold."""
    if gray is None or getattr(gray, "size", 0) == 0:
        return _MIN_EXPECTED_CHARS
    h, w = gray.shape[:2]
    if h <= 0:
        return _MIN_EXPECTED_CHARS
    return max(_MIN_EXPECTED_CHARS, int((w / float(h)) * _CHARS_PER_HEIGHT) + 2)


def trim_overrun(text: str, gray) -> str:
    """
    Drop generation past what the crop can hold, on a word boundary.

    Cutting mid-word would turn a correct reading into a wrong one, so the
    last whole word that still fits is kept; if even the first word overruns
    the text is returned untouched rather than mangled.
    """
    if not text:
        return text
    limit = expected_max_chars(gray)
    if len(text) <= limit:
        return text
    kept = []
    used = 0
    for word in text.split():
        if kept and used + 1 + len(word) > limit:
            break
        used += (1 if kept else 0) + len(word)
        kept.append(word)
    return " ".join(kept) if kept else text


def transcribe_line(gray, ocr_text: str = "", model_dir: str = MODEL_DIR) -> Optional[dict]:
    """
    Transcribe a handwritten line, REFUSING scripts this model cannot read.

    The script is judged from what Tesseract already read on the same line,
    via field_extractor.detect_script - the cheapest reliable signal available
    and one the pipeline computes anyway.

    This guard is the whole reason the module is safe to switch on. Measured
    on a real Rajasthan certificate, the Devanagari entries came back as
    "MRB Protected Authority Commission Decem" and "sentirement ," at
    confidences of 0.36 and 0.43 - fluent English for text containing no
    English at all. A recogniser that cannot say "I cannot read this" and
    instead invents an owner's name is worse than no recogniser, so the
    refusal happens BEFORE the model runs rather than being left to a
    confidence threshold that these outputs would sometimes pass.
    """
    import field_extractor

    script = field_extractor.detect_script(ocr_text or "")
    if ocr_text and script not in SUPPORTED_SCRIPTS:
        return {
            "text": "", "confidence": 0.0, "script": script,
            "script_supported": False,
            "reason": (f"TrOCR's handwritten model reads Latin script only "
                       f"(trained on IAM English); this line is {script}. "
                       f"No transcription was attempted."),
        }
    result = transcribe(gray, model_dir)
    if result is None:
        return None
    result["script"] = script or "unknown"
    result["script_supported"] = True
    # Never auto-accepted, whatever the confidence: this is a suggestion for
    # a human, exactly like the cloud-LLM suggestions in llm_extractor.py.
    result["needs_review"] = True
    return result


def describe(model_dir: str = MODEL_DIR) -> dict:
    if not ONNX_AVAILABLE:
        return {"available": False,
                "reason": "onnxruntime/numpy not installed"}
    missing = [f for f in (ENCODER_FILE, DECODER_FILE, TOKENIZER_FILE)
               if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        return {"available": False,
                "reason": "model files missing: " + ", ".join(missing),
                "hint": "tools/fetch_trocr.py downloads them."}
    return {
        "available": True,
        "model": "Xenova/trocr-base-handwritten (int8 ONNX)",
        "script": "Latin only - trained on IAM English handwriting",
        "auto_accept": False,
    }
