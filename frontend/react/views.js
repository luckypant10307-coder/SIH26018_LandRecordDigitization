/* Intelligent Land Record Digitization & Validation System - SIH 2026, PS 26018
 *
 * React port: the seven tab views.
 */

"use strict";

/* ================================================================== *
 * Ingest
 * ================================================================== */

function CapabilityList({ caps, masterLoaded, samples }) {
  caps = caps || {};
  const items = [
    { on: caps.pdf_text_layer, name: "Native PDF text layer",
      note: caps.pdf_text_layer ? "PyMuPDF available" : "PyMuPDF missing" },
    { on: caps.image_preprocessing, name: "Image quality assessment",
      note: caps.image_preprocessing ? "OpenCV deskew + denoise" : "OpenCV missing" },
    { on: caps.tesseract, name: "Tesseract OCR",
      note: caps.tesseract
        ? "Languages: " + (caps.tesseract_languages || []).slice(0, 6).join(", ")
        : "Not installed — scans queue for manual entry" },
    { on: masterLoaded, name: "LGD administrative master",
      note: masterLoaded ? "District / tehsil cross-check active" : "Master data missing" },
    { on: (samples || []).length > 0, name: "Sample corpus",
      note: (samples || []).length + " documents bundled" },
  ];
  return h("ul", { className: "cap-list" }, items.map((i, k) =>
    h("li", { key: k },
      h("span", { className: "cap-dot " + (i.on ? "on" : "off") }),
      h("span", { className: "cap-name" }, i.name),
      h("span", { className: "cap-note" }, i.note))));
}

function LastRunTable({ out }) {
  const rows = out.processed || [];
  const errors = out.errors || [];
  if (!rows.length && !errors.length) return null;
  return h("div", { className: "card" },
    h("h3", null, "Last ingestion run"),
    h("div", { className: "table-wrap" },
      h("table", { className: "data-table" },
        h("thead", null, h("tr", null,
          ["Document", "Engine", "Fields", "Trust", "Issues", "Outcome"].map((c, i) =>
            h("th", { key: i, className: i >= 2 && i <= 4 ? "num" : null }, c)))),
        h("tbody", null,
          rows.map((r, i) => {
            const s = r.summary || {};
            return h("tr", { key: "p" + i },
              h("td", { className: "cell-trunc" }, r.filename),
              h("td", null, h(Badge, { gray: true }, r.engine)),
              h("td", { className: "num" },
                (s.fields_extracted || 0) + "/" + (s.fields_total || 0)),
              h("td", { className: "num" }, (r.trust_score || 0).toFixed(0)),
              h("td", { className: "num" },
                (r.error_count || 0) + "e / " + (r.warning_count || 0) + "w"),
              h("td", null, h(Badge, { status: r.decision })));
          }).concat(errors.map((e, i) =>
            h("tr", { key: "e" + i },
              h("td", { className: "cell-trunc" }, e.filename),
              h("td", { colSpan: 5 },
                h("span", { className: "badge red" }, "Failed"), " ",
                h("span", { className: "muted small" }, e.error)))))))));
}

function IngestView({ session, onIngested, goTo }) {
  const { can, toast, username } = useApp();
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [lastRun, setLastRun] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const addFiles = (list) => {
    setFiles((prev) => prev.concat(Array.from(list || [])));
    setError(null);
  };

  const run = async (label, request, after) => {
    setBusy(label);
    setError(null);
    try {
      const out = await request();
      setLastRun(out);
      await onIngested();
      if (after) after(out);
    } catch (err) {
      setError(err.message);
      toast(err.message, "err");
    } finally {
      setBusy(null);
    }
  };

  const upload = () => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    run("Extracting and validating…",
        () => apiCall("/api/upload", { method: "POST", body: fd }, username),
        (out) => {
          setFiles([]);
          toast(`Processed ${out.count} document(s)`, "ok");
        });
  };

  const seed = () => {
    run("Ingesting the bundled sample corpus…",
        () => apiCall("/api/seed", { method: "POST" }, username),
        (out) => {
          toast(`Ingested ${out.count} sample records`, "ok");
          goTo("queue");
        });
  };

  return h(Fragment, null,
    h("div", { className: "card" },
      h("h3", null, "Ingest documents"),
      h("div", {
        className: "dropzone" + (dragging ? " dragover" : ""),
        onClick: () => inputRef.current && inputRef.current.click(),
        onDragOver: (e) => { e.preventDefault(); setDragging(true); },
        onDragLeave: () => setDragging(false),
        onDrop: (e) => {
          e.preventDefault();
          setDragging(false);
          addFiles(e.dataTransfer.files);
        },
      },
        h("p", null, "Drop scanned khatauni / 7-12 / jamabandi files here"),
        h("p", { className: "muted small" }, "PDF, PNG, JPG or TIFF"),
        h("button", {
          className: "btn",
          onClick: (e) => { e.stopPropagation(); inputRef.current.click(); },
        }, "Browse…"),
        h("input", {
          ref: inputRef, type: "file", multiple: true, hidden: true,
          accept: ".pdf,.png,.jpg,.jpeg,.tif,.tiff",
          onChange: (e) => addFiles(e.target.files),
        })),

      files.length
        ? h("ul", { className: "file-list" }, files.map((f, i) =>
            h("li", { key: i },
              h("span", { className: "fname" }, f.name),
              h("span", { className: "fsize" }, fmtBytes(f.size)),
              h("button", {
                className: "link-btn",
                onClick: () => setFiles((prev) => prev.filter((_, j) => j !== i)),
              }, "remove"))))
        : null,

      h("div", { className: "row-actions" },
        h("button", {
          className: "btn primary",
          disabled: !can("upload") || !files.length || !!busy,
          onClick: upload,
        }, "Extract and validate"),
        h("button", {
          className: "btn",
          disabled: !can("upload") || !!busy,
          onClick: seed,
        }, "Load sample corpus")),

      busy ? h(Spinner, { label: busy }) : null,
      error ? h("p", { className: "status-note" }, error) : null),

    h("div", { className: "card" },
      h("h3", null, "Pipeline capabilities on this machine"),
      h(CapabilityList, {
        caps: session.capabilities,
        masterLoaded: session.admin_master_loaded,
        samples: session.samples_available,
      })),

    lastRun ? h(LastRunTable, { out: lastRun }) : null);
}

/* ================================================================== *
 * Verification queue
 * ================================================================== */

function QueueView({ docs, state, filters, setFilters, onOpen }) {
  const { can, toast, username } = useApp();

  const exportCsv = () => {
    fetch("/api/export/csv", { headers: { "X-User": username } })
      .then((r) => (r.ok ? r.blob()
        : r.json().then((j) => Promise.reject(new Error(j.error)))))
      .then((blob) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "land_records_export.csv";
        a.click();
        URL.revokeObjectURL(a.href);
      })
      .catch((e) => toast(e.message, "err"));
  };

  const columns = [
    { label: "#", num: true, cell: (d) => d.id },
    { label: "Document", trunc: true, title: (d) => d.filename, cell: (d) => d.filename },
    { label: "Owner", trunc: true, cell: (d) => d.owner_name || EM_DASH },
    { label: "Khasra", cell: (d) => d.khasra_number || EM_DASH },
    { label: "District", cell: (d) => d.district || EM_DASH },
    { label: "Engine", cell: (d) => h(Badge, { gray: true }, d.ocr_engine || EM_DASH) },
    { label: "Trust", num: true,
      cell: (d) => (d.trust_score === null ? EM_DASH : Number(d.trust_score).toFixed(0)) },
    { label: "Issues", num: true,
      cell: (d) => (d.error_count || 0) + "e / " + (d.warning_count || 0) + "w" },
    { label: "Status", cell: (d) => h(Badge, { status: d.status }) },
    { label: "", cell: (d) => h("button", {
        className: "btn tiny", onClick: () => onOpen(d.id),
      }, "Open") },
  ];

  return h("div", { className: "card" },
    h("div", { className: "card-head" },
      h("h3", null, "Verification queue"),
      h("div", { className: "row-actions" },
        h("input", {
          className: "input", type: "search", placeholder: "Search owner, khasra, village…",
          value: filters.search,
          onChange: (e) => setFilters(Object.assign({}, filters, { search: e.target.value })),
        }),
        h("select", {
          className: "input", value: filters.status,
          onChange: (e) => setFilters(Object.assign({}, filters, { status: e.target.value })),
        },
          h("option", { value: "" }, "All statuses"),
          Object.keys(STATUS_META).map((k) =>
            h("option", { key: k, value: k }, STATUS_META[k].label))),
        h("button", {
          className: "btn", disabled: !can("export"), onClick: exportCsv,
        }, "Export CSV"))),
    h(Async, { state, empty: null }, () =>
      h(DataTable, {
        columns, rows: docs, rowKey: (d) => d.id,
        empty: "No documents yet. Ingest a file or load the sample corpus.",
      })));
}

/* ================================================================== *
 * Verification workspace
 * ================================================================== */

/* One editable field row.
 *
 * The vanilla app makes the value div contentEditable and reads textContent
 * back on blur. That works but puts the value in the DOM as the source of
 * truth, which is exactly what React is not able to share. Here an edit
 * swaps in a controlled <input>, so the draft lives in component state and
 * the committed value comes from the server's response - one owner each. */
function FieldRow({ field, required, onSave }) {
  const { can, toast } = useApp();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const conf = Number(field.ai_confidence || 0);
  const blank = field.value === null || field.value === undefined || field.value === "";

  const begin = () => {
    if (!can("correct")) { toast("Your role cannot edit field values.", "err"); return; }
    setDraft(blank ? "" : String(field.value));
    setEditing(true);
  };

  const commit = () => {
    setEditing(false);
    const next = draft.trim();
    if (next !== (blank ? "" : String(field.value).trim())) onSave(field.field_key, next, false);
  };

  return h("div", { className: "field-row st-" + field.status },
    h("div", { className: "field-label" },
      field.display || field.field_key,
      required ? h("span", { className: "req" }, "*") : null),
    editing
      ? h("input", {
          className: "field-value editing", autoFocus: true, value: draft,
          onChange: (e) => setDraft(e.target.value),
          onBlur: commit,
          onKeyDown: (e) => {
            if (e.key === "Enter") { e.preventDefault(); e.target.blur(); }
            if (e.key === "Escape") { e.preventDefault(); setEditing(false); }
          },
        })
      : h("div", {
          className: "field-value" + (blank ? " blank" : ""), tabIndex: 0,
          onClick: begin,
          onKeyDown: (e) => { if (e.key === "Enter") { e.preventDefault(); begin(); } },
        }, blank ? "not found" : field.value),
    h("div", { className: "field-meta" },
      h("div", { className: "conf-bar" },
        h("span", { style: { width: (conf * 100).toFixed(0) + "%", background: confColor(conf) } })),
      h("span", { className: "conf-num" }, (conf * 100).toFixed(0) + "%"),
      h("button", {
        className: "tick-btn", title: "Confirm this value is correct",
        onClick: () => onSave(field.field_key, null, true),
      }, "✓")));
}

function IssueList({ issues }) {
  if (!issues || !issues.length) {
    return h("div", { className: "issue sev-info" },
      h("span", { className: "issue-rule" }, "CLEAR"),
      h("div", null, h("div", { className: "issue-msg" },
        "No validation issues. This record satisfies every rule.")));
  }
  const order = { error: 0, warning: 1, info: 2 };
  const sorted = issues.slice().sort(
    (a, b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3));
  return h(Fragment, null, sorted.map((i, k) =>
    h("div", { className: "issue sev-" + i.severity, key: k },
      h("span", { className: "issue-rule" }, i.rule),
      h("div", null,
        h("div", { className: "issue-msg" }, i.message),
        i.suggestion
          ? h("div", { className: "issue-fix" }, "Suggested fix: " + i.suggestion)
          : null))));
}

function DocumentPreview({ docId }) {
  const [showText, setShowText] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const text = useAsync(
    () => apiCall(`/api/documents/${docId}/text`),
    [docId], showText);

  useEffect(() => { setShowText(false); setImageFailed(false); }, [docId]);

  return h("div", { className: "preview-pane" },
    h("div", { className: "row-actions" },
      h("button", { className: "btn tiny", onClick: () => setShowText((v) => !v) },
        showText ? "Show document" : "Show extracted text")),
    showText || imageFailed
      ? h("pre", { className: "preview-text" },
          imageFailed && !showText
            ? "No image preview available for this document."
            : h(Async, { state: text }, (t) => t.full_text || "(no text extracted)"))
      : h("img", {
          className: "preview-img", alt: "Document preview",
          src: `/api/documents/${docId}/preview`,
          onError: () => setImageFailed(true),
        }));
}

function WorkspaceView({ docId, schema, onChanged }) {
  const { can, toast, username } = useApp();
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const doc = useAsync(
    () => apiCall(`/api/documents/${docId}`, {}, username),
    [docId, username], docId != null);

  const required = useMemo(() => {
    const map = {};
    (schema || []).forEach((s) => { map[s.key] = s.required; });
    return map;
  }, [schema]);

  if (docId == null) {
    return h("div", { className: "card" },
      h("p", { className: "mini-empty" },
        "Open a document from the verification queue to review it."));
  }

  const saveField = async (key, value, confirmOnly) => {
    try {
      await apiCall(`/api/documents/${docId}/fields`, {
        method: "POST",
        json: { field_key: key, value: value, confirm: !!confirmOnly },
      }, username);
      toast(confirmOnly ? "Value confirmed"
                        : "Correction saved and record revalidated", "ok");
      doc.reload();
      onChanged();
    } catch (err) {
      toast(err.message, "err");
      doc.reload();
    }
  };

  const decide = async (action, body) => {
    try {
      await apiCall(`/api/documents/${docId}/${action}`,
                    { method: "POST", json: body }, username);
      toast(action === "approve"
        ? "Record approved and committed to the repository"
        : "Record rejected", "ok");
      setRejecting(false);
      setReason("");
      doc.reload();
      onChanged();
    } catch (err) { toast(err.message, "err"); }
  };

  return h(Async, { state: doc }, (d) => {
    const q = d.quality || {};
    const trust = Number(d.trust_score || 0);
    const chips = [];
    if (q.legibility_score != null) chips.push(["Legibility", Number(q.legibility_score).toFixed(0) + "/100"]);
    if (q.sharpness !== undefined) chips.push(["Sharpness", Number(q.sharpness).toFixed(0)]);
    if (q.contrast !== undefined) chips.push(["Contrast", Number(q.contrast).toFixed(0)]);
    if (q.skew_deg !== undefined) chips.push(["Skew", Number(q.skew_deg).toFixed(1) + "°"]);
    if (d.mean_ocr_conf != null) chips.push(["Mean OCR conf", (Number(d.mean_ocr_conf) * 100).toFixed(0) + "%"]);
    if (d.processing_ms) chips.push(["Processed in", d.processing_ms + " ms"]);

    return h(Fragment, null,
      h("div", { className: "card" },
        h("div", { className: "ws-head" },
          h("div", null,
            h("h3", null, d.filename),
            h("p", { className: "muted small" },
              `Document #${d.id} · ${d.ocr_engine || EM_DASH} · `
              + `${d.page_count || 1} page(s) · uploaded ${fmtTime(d.uploaded_at)} · `),
            h(Badge, { status: d.status })),
          h("div", { className: "trust-badge " + trustClass(trust) },
            h("span", { className: "tb-value" }, trust.toFixed(0)),
            h("span", { className: "tb-label" }, "trust"))),
        h("div", { className: "quality-strip" },
          chips.map(([k, v], i) =>
            h("span", { className: "qchip", key: i }, k, " ", h("b", null, v))),
          (d.warnings || []).map((w, i) =>
            h("span", { className: "qchip", key: "w" + i }, w)))),

      h("div", { className: "ws-grid" },
        h("div", { className: "card" },
          h("h3", null, "Extracted fields"),
          h("div", { className: "field-list" },
            (d.fields || []).map((f) =>
              h(FieldRow, {
                key: f.field_key, field: f,
                required: required[f.field_key], onSave: saveField,
              })))),
        h("div", null,
          h("div", { className: "card" },
            h("h3", null, "Document"),
            h(DocumentPreview, { docId: d.id })),
          h("div", { className: "card" },
            h("h3", null, "Validation"),
            h(IssueList, { issues: d.issues || [] }),
            h("div", { className: "row-actions" },
              h("button", {
                className: "btn primary", disabled: !can("approve"),
                onClick: () => decide("approve", null),
              }, "Approve record"),
              h("button", {
                className: "btn danger", disabled: !can("reject"),
                onClick: () => setRejecting(true),
              }, "Reject"))))),

      h(Modal, {
        open: rejecting, title: "Reject this record", danger: true,
        confirmLabel: "Reject record",
        onCancel: () => setRejecting(false),
        onConfirm: () => decide("reject", { reason: reason.trim() }),
      },
        h("p", { className: "muted small" },
          "The reason is written to the audit trail against your name."),
        h("textarea", {
          className: "input", rows: 3, autoFocus: true, value: reason,
          placeholder: "Why is this record being rejected?",
          onChange: (e) => setReason(e.target.value),
        })));
  });
}

/* ================================================================== *
 * Dashboard
 * ================================================================== */

function DashboardView() {
  const { username } = useApp();
  const stats = useAsync(() => apiCall("/api/stats", {}, username), [username]);

  const statusColors = {
    auto_approved: "var(--green)", approved: "var(--green)",
    needs_review: "var(--orange)", blocked: "var(--red)",
    rejected: "var(--red)", processing: "var(--text-2)",
  };

  return h(Async, { state: stats }, (s) => {
    const byStatus = s.by_status || {};
    const kpis = [
      { v: s.documents_total || 0, l: "Documents processed",
        s: `${s.avg_processing_ms || 0} ms average` },
      { v: s.pending_verification || 0, l: "Pending verification",
        s: `${byStatus.blocked || 0} blocked, ${byStatus.needs_review || 0} in review` },
      { v: (s.avg_trust_score || 0).toFixed(0), l: "Average trust score",
        s: `Legibility ${(s.avg_legibility || 0).toFixed(0)}/100` },
      { v: s.extraction_precision === null ? EM_DASH
           : (s.extraction_precision * 100).toFixed(1) + "%",
        l: "Extraction precision",
        s: s.fields_reviewed
          ? `${s.fields_reviewed} field${s.fields_reviewed === 1 ? "" : "s"} human-reviewed`
          : "Awaiting human review" },
      { v: s.corrections_total || 0, l: "Corrections captured",
        s: "Training signal for the model" },
    ];

    return h(Fragment, null,
      h("div", { className: "kpi-row" },
        kpis.map((k, i) => h(Kpi, { key: i, value: k.v, label: k.l, sub: k.s }))),
      h("div", { className: "chart-grid" },
        h("div", { className: "card" }, h("h3", null, "By outcome"),
          h(BarChart, {
            rows: Object.keys(byStatus).map((k) => ({
              label: (STATUS_META[k] || { label: k }).label, value: byStatus[k], key: k })),
            colorFn: (r) => statusColors[r.key] || "var(--blue)" })),
        h("div", { className: "card" }, h("h3", null, "By district"),
          h(BarChart, {
            rows: (s.by_district || []).slice(0, 8).map((d) => ({
              label: d.district, value: d.c,
              display: `${d.c} · ${Number(d.avg_trust || 0).toFixed(0)}` })) })),
        h("div", { className: "card" }, h("h3", null, "Most frequent issues"),
          h(BarChart, {
            rows: (s.top_issues || []).slice(0, 8).map((i) => ({
              label: i.rule, value: i.count, sev: i.severity })),
            colorFn: (r) => r.sev === "error" ? "var(--red)"
              : r.sev === "warning" ? "var(--orange)" : "var(--blue)" })),
        h("div", { className: "card" }, h("h3", null, "By OCR engine"),
          h(BarChart, {
            rows: (s.by_engine || []).map((e) => ({
              label: e.engine, value: e.c,
              display: `${e.c} · ${Number(e.avg_trust || 0).toFixed(0)}` })) }))),
      h("div", { className: "card" },
        h("h3", null, "Per-field quality"),
        h(DataTable, {
          rows: s.field_accuracy || [],
          rowKey: (f) => f.field_key,
          empty: "No fields recorded yet.",
          columns: [
            { label: "Field", cell: (f) => f.display || f.field_key },
            { label: "Present", num: true, cell: (f) => (f.total || 0) - (f.missing || 0) },
            { label: "Missing", num: true, cell: (f) => f.missing || 0 },
            { label: "Corrected", num: true, cell: (f) => f.corrected || 0 },
            { label: "Avg confidence", num: true,
              cell: (f) => ((f.avg_conf || 0) * 100).toFixed(0) + "%" },
            { label: "Precision", num: true,
              cell: (f) => f.precision == null
                ? h("span", { className: "muted" }, EM_DASH)
                : (f.precision * 100).toFixed(0) + "%" },
          ] })));
  });
}

/* ================================================================== *
 * Learning
 * ================================================================== */

function LearningView() {
  const { can, toast, username } = useApp();
  const learning = useAsync(() => apiCall("/api/learning", {}, username), [username]);

  const retrain = async () => {
    try {
      const out = await apiCall("/api/learning/retrain", { method: "POST" }, username);
      toast(`Model retrained on ${out.model.samples} correction sample(s)`, "ok");
      learning.reload();
    } catch (err) { toast(err.message, "err"); }
  };

  return h(Async, { state: learning }, (out) => {
    const m = out.model || {};
    const th = out.thresholds || {};
    return h(Fragment, null,
      h("div", { className: "card-head" },
        h("h3", null, "AI learning loop"),
        h("button", { className: "btn primary", disabled: !can("retrain"), onClick: retrain },
          "Retrain from corrections")),
      h("div", { className: "kpi-row" },
        h(Kpi, { value: m.samples || 0, label: "Correction samples" }),
        h(Kpi, { value: (m.confusions || []).length, label: "Character confusion rules" }),
        h(Kpi, { value: (m.aliases || []).length, label: "Value aliases" }),
        h(Kpi, { value: (m.calibration || []).length, label: "Calibrated fields" })),
      h("div", { className: "chart-grid" },
        h("div", { className: "card" }, h("h3", null, "Character confusions"),
          (m.confusions || []).length
            ? m.confusions.map((c, i) => h(MiniItem, { key: i, note: `seen ${c.support}×` },
                h("code", null, c.from || c.wrong), " → ",
                h("code", null, c.to || c.right)))
            : h("p", { className: "mini-empty" },
                `No confusion rule has reached the evidence threshold yet `
                + `(needs ${th.min_confusion_support || 4} sightings).`)),
        h("div", { className: "card" }, h("h3", null, "Value aliases"),
          (m.aliases || []).length
            ? m.aliases.map((a, i) => h(MiniItem, { key: i, note: `${a.support}×` },
                h("code", null, a.wrong_value || a.wrong), " → ",
                h("code", null, a.corrected_value || a.right), " ",
                h("span", { className: "muted small" }, a.field_key)))
            : h("p", { className: "mini-empty" },
                `No alias learned yet (needs ${th.min_alias_support || 3} identical corrections).`)),
        h("div", { className: "card" }, h("h3", null, "Confidence calibration"),
          (m.calibration || []).length
            ? m.calibration.map((c, i) => h(MiniItem, { key: i,
                note: `confidence × ${Number(c.multiplier).toFixed(2)} `
                  + `(${c.reviewed} reviewed, `
                  + `${(Number(c.observed_precision) * 100).toFixed(0)}% precise)` },
                c.field_key))
            : h("p", { className: "mini-empty" },
                `Calibration starts after ${th.min_calibration_sample || 8} `
                + `reviewed fields per field type.`))),
      h("div", { className: "card" },
        h("h3", null, "Recent correction signals"),
        h(DataTable, {
          rows: out.recent_corrections || [],
          empty: "No corrections recorded yet.",
          columns: [
            { label: "When", muted: true, cell: (c) => fmtTime(c.at) },
            { label: "Field", cell: (c) => c.field_key },
            { label: "Extracted", trunc: true, cell: (c) => c.ai_value || EM_DASH },
            { label: "Corrected to", trunc: true, cell: (c) => c.human_value || EM_DASH },
            { label: "Conf", num: true,
              cell: (c) => c.ai_confidence == null ? EM_DASH
                : (Number(c.ai_confidence) * 100).toFixed(0) + "%" },
            { label: "Engine", muted: true, cell: (c) => c.ocr_engine || EM_DASH },
          ] })));
  });
}

/* ================================================================== *
 * Audit trail
 * ================================================================== */

function AuditView() {
  const { username } = useApp();
  const audit = useAsync(() => apiCall("/api/audit?limit=200", {}, username), [username]);

  return h("div", { className: "card" },
    h("h3", null, "Audit trail"),
    h("p", { className: "muted small" },
      "Append-only. Every ingestion, correction, approval and rejection, "
      + "with the actor and their role at the time."),
    h(Async, { state: audit }, (out) =>
      h(DataTable, {
        rows: out.entries || [],
        empty: "No audit entries yet.",
        columns: [
          { label: "When", muted: true, cell: (e) => fmtTime(e.at) },
          { label: "Actor", cell: (e) => e.username || "system" },
          { label: "Role", cell: (e) => h(Badge, { gray: true }, e.role || EM_DASH) },
          { label: "Action", cell: (e) => e.action },
          { label: "Doc", num: true, cell: (e) => e.document_id || EM_DASH },
          { label: "Detail", muted: true, cell: (e) =>
              [e.field_key,
               e.old_value && e.new_value ? `${e.old_value} → ${e.new_value}` : null,
               e.detail].filter(Boolean).join(" · ") },
        ] })));
}
