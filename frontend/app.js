/* Intelligent Land Record Digitization & Validation System - SIH 2026, PS 26018
 * Vanilla JS, no build step, no framework. Talks to the stdlib Python API.
 */

"use strict";

const state = {
  user: null,
  rights: [],
  users: [],
  schema: [],
  docs: [],
  currentId: null,
  currentDoc: null,
  pendingFiles: [],
  showingText: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ------------------------------------------------------------------ *
 * API helper
 * ------------------------------------------------------------------ */

/**
 * Headers that identify the caller to the API.
 *
 * The Bearer token is what the server actually verifies. X-User is still
 * sent because an offline install with no JWT secret configured has nothing
 * to verify a token against and falls back to it; a server that IS
 * configured ignores it entirely, so sending both is not a way around the
 * check.
 *
 * The token is read from the live session on every call rather than cached
 * at sign-in: Supabase access tokens expire after about an hour, and the
 * client refreshes them in the background, so a cached copy would start
 * returning 401 partway through a long verification session.
 */
async function authHeaders() {
  const headers = {};
  if (state.user) headers["X-User"] = state.user.username;
  try {
    if (typeof SupabaseAuth !== "undefined" && SupabaseAuth.getSession) {
      const session = await SupabaseAuth.getSession();
      if (session && session.access_token) {
        headers["Authorization"] = `Bearer ${session.access_token}`;
      }
    }
  } catch (e) {
    // No session to attach. The request still goes out, and the server
    // decides whether that is allowed - the client does not pre-judge it.
  }
  return headers;
}

async function api(path, opts = {}) {
  const headers = Object.assign({}, await authHeaders(), opts.headers || {});
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
    delete opts.json;
  }
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  const ctype = res.headers.get("Content-Type") || "";
  let payload = null;
  if (ctype.includes("application/json")) {
    payload = await res.json();
  } else {
    payload = { raw: await res.text() };
  }
  if (!res.ok) throw new Error((payload && payload.error) || `Request failed (${res.status})`);
  return payload;
}

function can(right) {
  return state.rights.indexOf(right) !== -1;
}

/* ------------------------------------------------------------------ *
 * Small UI utilities
 * ------------------------------------------------------------------ */

function toast(msg, kind) {
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = msg;
  $("#toastHost").appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
}

function fmtTime(s) {
  if (!s) return "\u2014";
  const d = new Date(s.replace(" ", "T") + (s.endsWith("Z") ? "" : "Z"));
  if (isNaN(d)) return s;
  return d.toLocaleString(undefined, {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
  });
}

const STATUS_META = {
  auto_approved: { label: "Auto-approved", cls: "green" },
  approved:      { label: "Approved",      cls: "green" },
  needs_review:  { label: "Needs review",  cls: "orange" },
  blocked:       { label: "Blocked",       cls: "red" },
  rejected:      { label: "Rejected",      cls: "red" },
  processing:    { label: "Processing",     cls: "gray" },
};

/* ------------------------------------------------------------------ *
 * How a document was read, in words an officer can act on.
 *
 * The backend records the engine that read each document as an internal
 * identifier - "tesseract:hin+eng", "pdf_text_layer", "degraded_no_ocr".
 * Those are the right names in a log and the wrong names on screen: they
 * name our implementation instead of answering the question the reader
 * actually has, which is "how much should I trust this text, and do I need
 * to look at the original?"
 *
 * So the identifier is translated once, here, and nothing else in the UI
 * prints a raw engine string.
 * ------------------------------------------------------------------ */

const LANGUAGE_NAMES = {
  hin: "Hindi",    eng: "English",  mar: "Marathi",  ben: "Bengali",
  tam: "Tamil",    tel: "Telugu",   kan: "Kannada",  mal: "Malayalam",
  guj: "Gujarati", pan: "Punjabi",  ori: "Odia",     asm: "Assamese",
  urd: "Urdu",     nep: "Nepali",   san: "Sanskrit", sat: "Santali",
  kok: "Konkani",  mni: "Manipuri", div: "Dhivehi",  sin: "Sinhala",
};

function languageNames(codes) {
  return (codes || [])
    .map((c) => LANGUAGE_NAMES[String(c).trim()] || String(c).trim())
    .filter(Boolean);
}

/**
 * Split installed languages into the ones this system exists to read and
 * everything else.
 *
 * A machine with the full language set installed has ~124 packs, and listing
 * the alphabetically-first six gives "afr, amh, ara, aze" - true, useless,
 * and it buries the fact that Hindi is supported. Indian languages are
 * listed by name in their own order; the rest are summarised as a count,
 * because "and 118 other languages" is the whole of what they are worth
 * saying here.
 */
function indianLanguages(codes) {
  const order = Object.keys(LANGUAGE_NAMES);
  const set = new Set((codes || []).map((c) => String(c).trim()));
  const indian = order.filter((c) => set.has(c)).map((c) => LANGUAGE_NAMES[c]);
  return { indian, otherCount: set.size - indian.length };
}

/**
 * Translate an internal engine identifier into a source description.
 *
 * Returns { label, detail, cls } - label is short enough for a table cell,
 * detail expands it for a tooltip, and cls marks the ones that mean "a human
 * still has to look at this".
 */
function readingMethod(engine) {
  const raw = String(engine || "").trim();
  if (!raw || raw === "none") {
    return { label: "Not read", cls: "red",
             detail: "No text could be extracted from this file." };
  }
  if (raw.startsWith("tesseract")) {
    const codes = (raw.split(":")[1] || "").split("+").filter(Boolean);
    const names = languageNames(codes);
    return {
      label: "Scanned",
      cls: "orange",
      detail: names.length
        ? "Read by character recognition (" + names.join(", ") + ")."
        : "Read by character recognition from a scanned image.",
    };
  }
  if (raw === "trocr" || raw.startsWith("trocr")) {
    return { label: "Handwritten", cls: "orange",
             detail: "Handwriting recognition - verify against the original." };
  }
  if (raw === "pdf_text_layer" || raw === "pdf_text" || raw === "native_text") {
    return { label: "Digital PDF", cls: "green",
             detail: "Text taken directly from the file, not recognised from an image." };
  }
  if (raw === "plain_text") {
    return { label: "Digital text", cls: "green",
             detail: "Text taken directly from the file." };
  }
  // Office and web documents. All are read as text rather than recognised
  // from an image, so all are equally reliable - but naming the kind of
  // document is more use to the reader than a single "Office document".
  const DIRECT = {
    word_docx:          "Word document",
    opendocument_text:  "OpenDocument text",
    excel_xlsx:         "Excel workbook",
    opendocument_sheet: "OpenDocument sheet",
    powerpoint_pptx:    "Presentation",
    delimited_text:     "Spreadsheet export",
    html:               "Web page",
    rich_text:          "Rich text",
  };
  if (DIRECT[raw]) {
    return { label: DIRECT[raw], cls: "green",
             detail: "Text taken directly from the document, "
                   + "not recognised from an image." };
  }
  if (raw === "degraded_no_ocr") {
    return { label: "Manual entry needed", cls: "red",
             detail: "No recognition engine was available for this file, so it is queued "
                   + "for manual entry rather than recorded as empty." };
  }
  // An identifier we have no translation for is shown as-is rather than
  // hidden: an unexplained blank in this column would be worse than a name
  // the reader does not recognise.
  return { label: raw, cls: "gray", detail: "" };
}

/**
 * Issue counts as two labelled figures rather than "0e / 18w".
 *
 * The shorthand is compact and unreadable: nothing on screen says what "e"
 * and "w" stand for, and errors and warnings route a record differently, so
 * the distinction has to survive being displayed.
 */
function issueCounts(errors, warnings) {
  const e = errors || 0, w = warnings || 0;
  if (!e && !w) return `<span class="ic-clear">None</span>`;
  const parts = [];
  if (e) parts.push(`<span class="ic-error">${e} error${e === 1 ? "" : "s"}</span>`);
  if (w) parts.push(`<span class="ic-warn">${w} warning${w === 1 ? "" : "s"}</span>`);
  return parts.join('<span class="ic-sep">·</span>');
}

function readingBadge(engine) {
  const m = readingMethod(engine);
  return `<span class="badge ${m.cls}"${m.detail ? ` title="${esc(m.detail)}"` : ""}>`
       + `${esc(m.label)}</span>`;
}

function statusBadge(status) {
  const m = STATUS_META[status] || { label: status || "\u2014", cls: "gray" };
  return `<span class="badge ${m.cls}">${esc(m.label)}</span>`;
}

function confColor(c) {
  if (c >= 0.85) return "var(--green)";
  if (c >= 0.6) return "var(--orange)";
  return "var(--red)";
}

/* ------------------------------------------------------------------ *
 * Tabs
 * ------------------------------------------------------------------ */

function showTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel-view").forEach((v) => v.classList.toggle("active", v.id === "view-" + name));
  if (name === "queue") loadDocuments();
  if (name === "cadastral") loadCadastralMap();
  if (name === "dashboard") loadDashboard();
  if (name === "learning") loadLearning();
  if (name === "audit") loadAudit();
}

$$(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    showTab(t.dataset.tab);
    // Deep-link the active tab so a reviewer can bookmark or share a view,
    // and so the browser back button behaves the way people expect.
    if (history.replaceState) history.replaceState(null, "", "#" + t.dataset.tab);
  }));

// A hash may address a tab ("#dashboard") or a specific record
// ("#workspace/7"), so a reviewer can be sent straight to one document.
function routeHash() {
  const parts = location.hash.replace("#", "").split("/");
  const name = parts[0];
  if (!name || !$(`.tab[data-tab="${name}"]`)) return false;
  showTab(name);
  if (name === "workspace" && parts[1]) openDocument(Number(parts[1]));
  return true;
}

window.addEventListener("hashchange", routeHash);

/* ------------------------------------------------------------------ *
 * Session
 * ------------------------------------------------------------------ */

async function loadSession(username) {
  if (username) state.user = { username: username };
  const s = await api("/api/session");
  state.user = s.user;
  state.rights = s.rights || [];
  state.users = s.users || [];

  const sel = $("#userSelect");
  sel.innerHTML = state.users
    .map((u) => `<option value="${esc(u.username)}"${u.username === s.user.username ? " selected" : ""}>${esc(u.full_name)}</option>`)
    .join("");
  $("#roleChip").textContent = s.user.role.charAt(0).toUpperCase() + s.user.role.slice(1);

  renderCapabilities(s.capabilities, s.admin_master_loaded, s.samples_available);
  applyRights();
  return s;
}

function applyRights() {
  $("#uploadBtn").disabled = !can("upload") || state.pendingFiles.length === 0;
  $("#seedBtn").disabled = !can("upload");
  $("#retrainBtn").disabled = !can("retrain");
  $("#exportCsvBtn").disabled = !can("export");
  $("#approveBtn").disabled = !can("approve");
  $("#rejectBtn").disabled = !can("reject");
}

$("#userSelect").addEventListener("change", async (e) => {
  await loadSession(e.target.value);
  toast(`Switched to ${state.user.full_name} (${state.user.role})`);
  if (state.currentId) openDocument(state.currentId);
});

function renderCapabilities(caps, masterLoaded, samples) {
  caps = caps || {};
  const langs = indianLanguages(caps.tesseract_languages || []);
  // Named by what the office gets, not by the library behind it - a
  // dependency name answers a question an operator did not ask.
  const items = [
    { on: caps.pdf_text_layer, name: "Digital PDF records" },
    { on: caps.image_preprocessing, name: "Scan restoration" },
    { on: caps.tesseract, name: "Scanned document recognition",
      note: langs.indian.length
        ? langs.indian.slice(0, 3).join(", ")
          + (langs.indian.length > 3 ? ` +${langs.indian.length - 3}` : "")
        : "" },
    { on: masterLoaded, name: "Administrative directory" },
    { on: (samples || []).length > 0, name: "Demonstration records",
      note: (samples || []).length ? String((samples || []).length) : "" },
  ];
  // The summary carries the count so the list does not have to carry a
  // sentence explaining itself.
  const ready = items.filter((i) => i.on).length;
  const summary = $("#capSummary");
  if (summary) {
    summary.textContent = ready === items.length
      ? "all capabilities available"
      : `${ready} of ${items.length} available`;
  }
  $("#capList").innerHTML = items.map((i) => `
    <li>
      <span class="cap-dot ${i.on ? "on" : "off"}"></span>
      <span class="cap-name">${esc(i.name)}</span>
      <span class="cap-note">${i.on ? esc(i.note || "") : "unavailable"}</span>
    </li>`).join("");
}

/* ------------------------------------------------------------------ *
 * Ingest
 * ------------------------------------------------------------------ */

const dz = $("#dropzone");
const fileInput = $("#fileInput");

dz.addEventListener("click", () => fileInput.click());
$("#browseBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("dragover");
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => addFiles(fileInput.files));

function addFiles(list) {
  state.pendingFiles = state.pendingFiles.concat(Array.from(list || []));
  renderFileList();
  applyRights();
}

function renderFileList() {
  $("#fileList").innerHTML = state.pendingFiles.map((f, i) => `
    <li>
      <span class="fname">${esc(f.name)}</span>
      <span class="fsize">${fmtBytes(f.size)}</span>
      <button class="link-btn" data-rm="${i}">remove</button>
    </li>`).join("");
  $$("#fileList [data-rm]").forEach((b) =>
    b.addEventListener("click", () => {
      state.pendingFiles.splice(Number(b.dataset.rm), 1);
      renderFileList();
      applyRights();
    }));
}

$("#uploadBtn").addEventListener("click", async () => {
  if (!state.pendingFiles.length) return;
  const fd = new FormData();
  state.pendingFiles.forEach((f) => fd.append("files", f, f.name));
  $("#uploadStatus").innerHTML = `<p class="status-note"><span class="spinner"></span> Extracting and validating\u2026</p>`;
  try {
    const out = await api("/api/upload", { method: "POST", body: fd });
    state.pendingFiles = [];
    renderFileList();
    renderLastRun(out);
    $("#uploadStatus").innerHTML = "";
    toast(`Processed ${out.count} document(s)`, "ok");
    await loadDocuments();
  } catch (err) {
    $("#uploadStatus").innerHTML = `<p class="status-note">${esc(err.message)}</p>`;
    toast(err.message, "err");
  }
  applyRights();
});

$("#seedBtn").addEventListener("click", async () => {
  $("#uploadStatus").innerHTML = `<p class="status-note"><span class="spinner"></span> Loading the sample records\u2026</p>`;
  try {
    const out = await api("/api/seed", { method: "POST" });
    renderLastRun(out);
    $("#uploadStatus").innerHTML = "";
    toast(`Ingested ${out.count} sample records`, "ok");
    await loadDocuments();
    showTab("queue");
  } catch (err) {
    $("#uploadStatus").innerHTML = `<p class="status-note">${esc(err.message)}</p>`;
    toast(err.message, "err");
  }
});

function renderLastRun(out) {
  const rows = out.processed || [];
  if (!rows.length && !(out.errors || []).length) return;
  $("#lastRunCard").hidden = false;
  let html = `<thead><tr><th>Document</th><th>Engine</th><th class="num">Fields</th>
    <th class="num">Trust</th><th class="num">Issues</th><th>Outcome</th></tr></thead><tbody>`;
  rows.forEach((r) => {
    const s = r.summary || {};
    html += `<tr>
      <td class="cell-trunc">${esc(r.filename)}</td>
      <td>${readingBadge(r.engine)}</td>
      <td class="num">${s.fields_extracted || 0}/${s.fields_total || 0}</td>
      <td class="num">${(r.trust_score || 0).toFixed(0)}</td>
      <td class="issues-cell">${issueCounts(r.error_count, r.warning_count)}</td>
      <td>${statusBadge(r.decision)}</td>
    </tr>`;
  });
  (out.errors || []).forEach((e) => {
    html += `<tr><td class="cell-trunc">${esc(e.filename)}</td>
      <td colspan="5"><span class="badge red">Failed</span> <span class="muted small">${esc(e.error)}</span></td></tr>`;
  });
  $("#lastRunTable").innerHTML = html + "</tbody>";
}

/* ------------------------------------------------------------------ *
 * Queue
 * ------------------------------------------------------------------ */

let searchTimer = null;
$("#searchInput").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(loadDocuments, 220);
});
$("#statusFilter").addEventListener("change", loadDocuments);

$("#exportCsvBtn").addEventListener("click", async () => {
  const url = "/api/export/csv";
  fetch(url, { headers: await authHeaders() })
    .then((r) => (r.ok ? r.blob() : r.json().then((j) => Promise.reject(new Error(j.error)))))
    .then((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "land_records_export.csv";
      a.click();
      URL.revokeObjectURL(a.href);
    })
    .catch((e) => toast(e.message, "err"));
});

async function loadDocuments() {
  const q = encodeURIComponent($("#searchInput").value.trim());
  const st = $("#statusFilter").value;
  const out = await api(`/api/documents?status=${st}&search=${q}&limit=200`);
  state.docs = out.documents || [];
  renderQueue();
}

function renderQueue() {
  const tbody = $("#queueTable tbody");
  const pending = state.docs.filter(
    (d) => d.status === "needs_review" || d.status === "blocked").length;
  $("#queueCount").textContent = pending;
  $("#queueEmpty").hidden = state.docs.length > 0;

  tbody.innerHTML = state.docs.map((d) => `
    <tr>
      <td class="num">${d.id}</td>
      <td class="cell-trunc" title="${esc(d.filename)}">${esc(d.filename)}</td>
      <td class="cell-trunc">${esc(d.owner_name || "\u2014")}</td>
      <td>${esc(d.khasra_number || "\u2014")}</td>
      <td>${esc(d.district || "\u2014")}</td>
      <td>${readingBadge(d.ocr_engine)}</td>
      <td class="num">${d.trust_score === null ? "\u2014" : Number(d.trust_score).toFixed(0)}</td>
      <td class="issues-cell">${issueCounts(d.error_count, d.warning_count)}</td>
      <td>${statusBadge(d.status)}</td>
      <td><button class="btn tiny" data-open="${d.id}">Open</button></td>
    </tr>`).join("");

  $$("#queueTable [data-open]").forEach((b) =>
    b.addEventListener("click", () => {
      openDocument(Number(b.dataset.open));
      showTab("workspace");
    }));
}

/* ------------------------------------------------------------------ *
 * Verification workspace
 * ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ *
 * What the record carries beyond its 17 fields
 *
 * Each of these was already being computed and stored and had nowhere to
 * appear. The co-owner list is the starkest: a parcel with sixteen
 * claimants was being shown as one name, because the schema holds one
 * owner_name and the list lived only in the database.
 *
 * All three hide when empty. A parcel with a single owner shows no
 * co-owner table and a document with no map shows no map panel - an empty
 * panel is furniture, and furniture is what makes a working screen look
 * like a brochure.
 * ------------------------------------------------------------------ */

function renderGeotag(geo) {
  const el = $("#wsGeotag");
  if (!geo || geo.lat == null || geo.lon == null) { el.hidden = true; return; }
  el.hidden = false;
  const metres = Number(geo.accuracy_m || 0);
  // Stated in the unit a reader can act on. "300000 m" invites nobody to
  // notice that the record has been placed 300 km from the parcel.
  const spread = metres >= 1000
    ? `±${Math.round(metres / 1000)} km`
    : `±${metres} m`;
  // Parcel-grade and place-name are different claims and must never look
  // alike: one is the plot, the other is the district it sits in.
  const grade = metres <= 100 ? "parcel" : "approximate";
  el.innerHTML =
    `<span class="geo-grade ${grade}">${grade === "parcel" ? "Parcel" : "Approximate"}</span>`
    + `<span class="geo-coord">${Number(geo.lat).toFixed(5)}, ${Number(geo.lon).toFixed(5)}</span>`
    + `<span class="geo-spread">${spread}</span>`
    + (geo.name ? `<span class="geo-name">${esc(geo.name)}</span>` : "");
}

function renderOwners(owners) {
  const card = $("#ownersCard");
  if (!owners || owners.length < 2) { card.hidden = true; return; }
  card.hidden = false;
  $("#ownersCount").textContent =
    `${owners.length} claimants on this parcel`;
  $("#ownersTable").innerHTML =
    `<thead><tr><th>#</th><th>Name</th><th>Father / guardian</th></tr></thead><tbody>`
    + owners.map((o, i) => `<tr>
        <td class="num">${i + 1}</td>
        <td>${esc(o.name || "—")}</td>
        <td>${o.father_name ? esc(o.father_name)
                            : `<span class="muted">—</span>`}</td>
      </tr>`).join("")
    + `</tbody>`;
}

function renderParcelMap(pm) {
  const card = $("#parcelCard");
  if (!pm || (!pm.subject_label && !(pm.neighbour_labels || []).length)) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  $("#parcelSource").textContent = "read from the document's own map";
  const rows = [];
  if (pm.subject_label) rows.push(["Khasra on the map", esc(pm.subject_label)]);
  if (pm.subject_area_px) {
    // Pixels, said plainly. These maps carry no scale, so converting to
    // square metres would be inventing a measurement.
    rows.push(["Traced area", `${Math.round(pm.subject_area_px).toLocaleString()} px`]);
  }
  if ((pm.subject_polygon || []).length) {
    rows.push(["Boundary", `${pm.subject_polygon.length} vertices`]);
  }
  if ((pm.neighbour_labels || []).length) {
    rows.push(["Adjoining plots",
      pm.neighbour_labels.map((n) => `<span class="pill">${esc(n)}</span>`).join("")]);
  }
  $("#parcelFacts").innerHTML = rows
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
}

async function openDocument(id) {
  state.currentId = id;
  const doc = await api(`/api/documents/${id}`);
  state.currentDoc = doc;
  $("#wsEmpty").hidden = true;
  $("#wsBody").hidden = false;

  $("#wsTitle").textContent = doc.filename;
  const q = doc.quality || {};
  $("#wsMeta").innerHTML =
    `Document #${doc.id} &middot; ${esc(readingMethod(doc.ocr_engine).label)} &middot; ` +
    `${doc.page_count || 1} page(s) &middot; uploaded ${fmtTime(doc.uploaded_at)} ` +
    `&middot; ${statusBadge(doc.status)}`;

  const trust = Number(doc.trust_score || 0);
  const tb = $("#wsTrust");
  tb.className = "trust-badge " + (trust >= 80 ? "good" : trust >= 50 ? "mid" : "bad");
  tb.querySelector(".tb-value").textContent = trust.toFixed(0);

  state.showingText = false;
  $("#previewImg").hidden = false;
  $("#previewText").hidden = true;
  $("#toggleTextBtn").textContent = "Show extracted text";
  $("#previewImg").src = `/api/documents/${id}/preview`;
  $("#previewImg").onerror = function () {
    this.hidden = true;
    $("#previewText").hidden = false;
    $("#previewText").textContent = "No image preview available for this document.";
  };

  renderGeotag(doc.geotag);
  renderOwners(doc.owners);
  renderParcelMap(doc.parcel_map);

  const chips = [];
  if (q.legibility_score !== undefined && q.legibility_score !== null)
    chips.push(["Legibility", Number(q.legibility_score).toFixed(0) + "/100"]);
  if (q.sharpness !== undefined) chips.push(["Sharpness", Number(q.sharpness).toFixed(0)]);
  if (q.contrast !== undefined) chips.push(["Contrast", Number(q.contrast).toFixed(0)]);
  if (q.skew_angle !== undefined) chips.push(["Skew", Number(q.skew_angle).toFixed(1) + "\u00b0"]);
  if (doc.mean_ocr_conf !== null && doc.mean_ocr_conf !== undefined)
    chips.push(["Mean OCR conf", (Number(doc.mean_ocr_conf) * 100).toFixed(0) + "%"]);
  if (doc.processing_ms) chips.push(["Processed in", doc.processing_ms + " ms"]);
  $("#qualityStrip").innerHTML = chips
    .map(([k, v]) => `<span class="qchip">${esc(k)} <b>${esc(v)}</b></span>`).join("") +
    (doc.warnings && doc.warnings.length
      ? doc.warnings.map((w) => `<span class="qchip">${esc(w)}</span>`).join("")
      : "");

  renderFields(doc.fields || []);
  renderIssues(doc.issues || []);
  applyRights();
}

$("#toggleTextBtn").addEventListener("click", async () => {
  state.showingText = !state.showingText;
  $("#toggleTextBtn").textContent = state.showingText ? "Show document" : "Show extracted text";
  $("#previewImg").hidden = state.showingText;
  $("#previewText").hidden = !state.showingText;
  if (state.showingText && !$("#previewText").dataset.loaded) {
    const t = await api(`/api/documents/${state.currentId}/text`);
    $("#previewText").textContent = t.full_text || "(no text extracted)";
    $("#previewText").dataset.loaded = "1";
  }
});

function renderFields(fields) {
  const req = {};
  state.schema.forEach((s) => { req[s.key] = s.required; });

  $("#fieldList").innerHTML = fields.map((f) => {
    const conf = Number(f.ai_confidence || 0);
    const blank = f.value === null || f.value === undefined || f.value === "";
    return `
      <div class="field-row st-${esc(f.status)}" data-key="${esc(f.field_key)}">
        <div class="field-label">${esc(f.display || f.field_key)}${req[f.field_key] ? '<span class="req">*</span>' : ""}</div>
        <div class="field-value ${blank ? "blank" : ""}" data-edit="${esc(f.field_key)}" tabindex="0">${blank ? "not found" : esc(f.value)}</div>
        <div class="field-meta">
          <div class="conf-bar"><span style="width:${(conf * 100).toFixed(0)}%;background:${confColor(conf)}"></span></div>
          <span class="conf-num">${(conf * 100).toFixed(0)}%</span>
          <button class="tick-btn" data-confirm="${esc(f.field_key)}" title="Confirm this value is correct">\u2713</button>
        </div>
      </div>`;
  }).join("");

  $$("#fieldList [data-edit]").forEach((el) => {
    el.addEventListener("click", () => beginEdit(el));
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); beginEdit(el); } });
  });
  $$("#fieldList [data-confirm]").forEach((b) =>
    b.addEventListener("click", () => saveField(b.dataset.confirm, null, true)));
}

function beginEdit(el) {
  if (!can("correct")) { toast("Your role cannot edit field values.", "err"); return; }
  if (el.dataset.editing) return;
  const key = el.dataset.edit;
  const original = el.classList.contains("blank") ? "" : el.textContent;
  el.dataset.editing = "1";
  el.classList.add("editing");
  el.contentEditable = "true";
  el.textContent = original;
  el.focus();
  document.getSelection().selectAllChildren(el);

  const finish = async (commit) => {
    el.contentEditable = "false";
    el.classList.remove("editing");
    delete el.dataset.editing;
    const next = el.textContent.trim();
    if (!commit || next === original.trim()) {
      el.textContent = original || "not found";
      return;
    }
    await saveField(key, next, false);
  };

  el.addEventListener("blur", () => finish(true), { once: true });
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); el.blur(); }
    if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
}

async function saveField(key, value, confirmOnly) {
  try {
    const out = await api(`/api/documents/${state.currentId}/fields`, {
      method: "POST",
      json: { field_key: key, value: value, confirm: !!confirmOnly },
    });
    renderFields(out.fields || []);
    renderIssues((out.validation && out.validation.issues) || []);
    const trust = Number((out.validation && out.validation.trust_score) || 0);
    const tb = $("#wsTrust");
    tb.className = "trust-badge " + (trust >= 80 ? "good" : trust >= 50 ? "mid" : "bad");
    tb.querySelector(".tb-value").textContent = trust.toFixed(0);
    toast(confirmOnly ? "Value confirmed" : "Correction saved and record revalidated", "ok");
    loadDocuments();
  } catch (err) {
    toast(err.message, "err");
    openDocument(state.currentId);
  }
}

function renderIssues(issues) {
  if (!issues || !issues.length) {
    $("#issueList").innerHTML =
      `<div class="issue sev-info"><span class="issue-rule">CLEAR</span>
       <div><div class="issue-msg">No validation issues. This record satisfies every rule.</div></div></div>`;
    return;
  }
  const order = { error: 0, warning: 1, info: 2 };
  issues = issues.slice().sort((a, b) => (order[a.severity] || 3) - (order[b.severity] || 3));
  $("#issueList").innerHTML = issues.map((i) => `
    <div class="issue sev-${esc(i.severity)}">
      <span class="issue-rule">${esc(i.rule)}</span>
      <div>
        <div class="issue-msg">${esc(i.message)}</div>
        ${i.suggestion ? `<div class="issue-fix">Suggested fix: ${esc(i.suggestion)}</div>` : ""}
      </div>
    </div>`).join("");
}

$("#approveBtn").addEventListener("click", async () => {
  try {
    await api(`/api/documents/${state.currentId}/approve`, { method: "POST" });
    toast("Record approved and committed to the repository", "ok");
    await loadDocuments();
    openDocument(state.currentId);
  } catch (err) { toast(err.message, "err"); }
});

$("#rejectBtn").addEventListener("click", () => { $("#rejectModal").hidden = false; $("#rejectReason").focus(); });
$("#rejectCancel").addEventListener("click", () => { $("#rejectModal").hidden = true; });
$("#rejectConfirm").addEventListener("click", async () => {
  const reason = $("#rejectReason").value.trim();
  try {
    await api(`/api/documents/${state.currentId}/reject`, { method: "POST", json: { reason } });
    $("#rejectModal").hidden = true;
    $("#rejectReason").value = "";
    toast("Record rejected", "ok");
    await loadDocuments();
    openDocument(state.currentId);
  } catch (err) { toast(err.message, "err"); }
});

/* ------------------------------------------------------------------ *
 * Dashboard
 * ------------------------------------------------------------------ */

function barChart(el, rows, colorFn) {
  const max = Math.max.apply(null, rows.map((r) => r.value).concat([1]));
  el.innerHTML = rows.length ? rows.map((r) => `
    <div class="bar-row">
      <span class="bar-label" title="${esc(r.label)}">${esc(r.label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(r.value / max * 100).toFixed(1)}%;background:${colorFn ? colorFn(r) : "var(--blue)"}"></span></span>
      <span class="bar-value">${esc(r.display !== undefined ? r.display : r.value)}</span>
    </div>`).join("") : `<p class="mini-empty">No data yet.</p>`;
}

/* ------------------------------------------------------------------ *
 * Cadastral map (S12a vectorization/georeferencing, rendered with a
 * locally-vendored Leaflet - see frontend/vendor/leaflet/, not a CDN, so
 * this keeps working with no internet connection).
 * ------------------------------------------------------------------ */

let cadastralMap = null;
let cadastralLayers = null;
let cadastralLayerControl = null;

// Matches STATUS_META's colour classes (see statusBadge above) so a parcel's
// fill colour always means the same thing the queue/workspace badges do.
/**
 * Resolve a CSS custom property to the literal colour it currently holds.
 *
 * Leaflet writes `color` and `fillColor` into the SVG as `stroke` and `fill`
 * ATTRIBUTES, and an SVG presentation attribute does not resolve var() - it
 * is simply invalid, so the browser falls back to the default fill, which is
 * black. That bug was survivable on a light theme (a black parcel is ugly but
 * visible) and became invisible parcels on a dark one.
 *
 * Reading the variable here keeps the stylesheet as the single source of
 * truth for the palette while handing Leaflet something SVG understands, so
 * the map follows any future theme change with no edit to this file.
 */
const cssColor = (() => {
  const cache = new Map();
  return (name, fallback) => {
    if (cache.has(name)) return cache.get(name);
    let value = "";
    try {
      value = getComputedStyle(document.documentElement)
        .getPropertyValue(name).trim();
    } catch (e) { /* non-browser context */ }
    const resolved = value || fallback;
    cache.set(name, resolved);
    return resolved;
  };
})();

function parcelColor(linked) {
  // Grey, and deliberately not a status colour: a parcel with no record is
  // not "pending", it is absent from the register entirely.
  if (!linked) return cssColor("--text-2", "#94A89C");
  const cls = (STATUS_META[linked.status] || {}).cls;
  if (cls === "green") return cssColor("--green", "#4FC98A");
  if (cls === "orange") return cssColor("--orange", "#E9B45C");
  if (cls === "red") return cssColor("--red", "#F2796B");
  return cssColor("--blue", "#5AB0F2");
}

/* Popup geometry, shared by every layer on the map.
 *
 * Leaflet pans the map to bring a popup fully into view. With the default
 * 5px padding a popup opened near an edge slides the whole sheet far enough
 * that the parcels end up against the opposite border - the map appears to
 * jump away from the parcel the user just clicked. Padding it away from the
 * edges keeps that pan small, and clear of the zoom buttons in the corner.
 */
const PARCEL_POPUP_OPTS = {
  maxWidth: 300,
  minWidth: 210,
  autoPanPadding: [58, 34],
  closeButton: true,
};

function parcelPopup(props) {
  const linked = props.linked_document;
  const rows = [];
  const add = (k, v) => rows.push(
    `<dt>${esc(k)}</dt><dd>${v}</dd>`);

  add("Khasra", props.khasra_number
    ? esc(props.khasra_number)
    : `<span class="pp-absent">Label not legible</span>`);

  if (props.area_m2) {
    add("Area", `${(props.area_m2 / 10000).toFixed(3)} ha`
      + `<span class="pp-sub">${Number(props.area_m2).toLocaleString()} m&sup2;</span>`);
  }
  if (linked) {
    add("Owner", linked.owner_name
      ? esc(linked.owner_name)
      : `<span class="pp-absent">Not recorded</span>`);
    add("Status", statusBadge(linked.status));
    if (linked.trust_score != null) {
      add("Trust", `${Math.round(linked.trust_score)}<span class="pp-sub">of 100</span>`);
    }
  }
  if (props.centroid_lat != null) {
    add("Centre", `<span class="pp-coord">${props.centroid_lat.toFixed(5)}, `
      + `${props.centroid_lon.toFixed(5)}</span>`);
  }

  const head = `<div class="pp-head">`
    + `<span class="pp-id">Parcel ${esc(props.parcel_id)}</span></div>`;
  const foot = linked
    ? `<a class="pp-link" href="#workspace/${linked.document_id}">`
      + `Open in verification workspace &rarr;</a>`
    : `<p class="pp-none">No digitised record is linked to this parcel yet.</p>`;

  return `<div class="parcel-popup">${head}`
    + `<dl class="pp-grid">${rows.join("")}</dl>${foot}</div>`;
}

/* ---- Thematic layer: land classification -------------------------------
 * Classification values arrive as free text in whatever script the record
 * was written in (सिंचित / असिंचित / irrigated / barren ...), so they are
 * bucketed by keyword rather than matched exactly - an unrecognised value
 * gets its own colour and is still shown, never silently dropped, because
 * "we could not classify this" is itself information a revenue officer
 * wants on the map.
 */
const LAND_CLASS_BUCKETS = [
  { key: "irrigated",   color: "#5FD08A", test: /सिंचित|सिचित|irrigat|बागायत/i, label: "Irrigated" },
  { key: "unirrigated", color: "#E0A867", test: /असिंचित|असचिति|unirrigat|जिरायत|dry/i, label: "Unirrigated" },
  { key: "barren",      color: "#A8B6AD", test: /बंजर|barren|waste/i, label: "Barren / waste" },
  { key: "residential", color: "#B18CE0", test: /आवासीय|residen|abadi|आबादी/i, label: "Residential" },
];

function landClassBucket(value) {
  if (!value) return null;
  // Unirrigated must be tested before irrigated: "असिंचित" contains "सिंचित".
  for (const b of LAND_CLASS_BUCKETS) {
    if (b.key === "unirrigated" && b.test.test(value)) return b;
  }
  for (const b of LAND_CLASS_BUCKETS) {
    if (b.key !== "unirrigated" && b.test.test(value)) return b;
  }
  return { key: "other", color: "#5AB0F2", label: "Other / unclassified" };
}

function buildLegend(entries) {
  const el = $("#cadastralLegend");
  if (!entries.length) { el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML = entries.map((e) =>
    `<span class="legend-item"><span class="legend-swatch" style="background:${e.color}"></span>${esc(e.label)}</span>`
  ).join("");
}

// Refetched every time this tab is opened, not cached like the queue/
// dashboard would be tempted to: the whole point is that a document's link
// to its parcel is live - approve, correct, or reject it elsewhere and the
// map must reflect that on the next visit, not the state from page load.
let cadastralSelectedMap = null;

// The village list can grow at runtime (an admin adds a real map), so it is
// refreshed on every visit rather than built once at page load.
async function refreshCadastralMapList() {
  const sel = $("#cadastralMapSelect");
  let maps = [];
  try {
    maps = (await api("/api/cadastral/maps")).maps || [];
  } catch (e) {
    maps = [];
  }
  sel.innerHTML = maps.map((m) =>
    `<option value="${esc(m.id)}">${esc(m.village)}${m.district ? " — " + esc(m.district) : ""}`
    + `${m.bundled ? " (demo)" : ""}</option>`).join("");
  if (maps.length && !maps.some((m) => m.id === cadastralSelectedMap)) {
    cadastralSelectedMap = maps[0].id;
  }
  if (cadastralSelectedMap) sel.value = cadastralSelectedMap;
  sel.parentElement.hidden = maps.length < 2;   // pointless chooser for one map
  return maps;
}

async function loadCadastralMap() {
  await refreshCadastralMapList();
  const qs = cadastralSelectedMap ? `?map=${encodeURIComponent(cadastralSelectedMap)}` : "";
  const geojson = await api("/api/cadastral/parcels" + qs);
  const metaEl = $("#cadastralMeta");
  const discEl = $("#cadastralDisclaimer");

  if (geojson._error) {
    metaEl.textContent = "Unavailable";
    discEl.hidden = false;
    discEl.textContent = geojson._error;
    return;
  }
  discEl.hidden = !geojson._disclaimer;
  if (geojson._disclaimer) discEl.textContent = geojson._disclaimer;

  const features = geojson.features || [];
  const linkedCount = features.filter((f) => f.properties.linked_document).length;
  const geo = geojson._georeferencing || {};
  // Three facts, in the order a revenue officer needs them: how big the sheet
  // is, how much of it is digitised, and how well it is placed on the earth.
  const parts = [
    `${features.length} ${features.length === 1 ? "parcel" : "parcels"}`,
    `${linkedCount} linked to a record`,
  ];
  if (geo.rms_metres != null) {
    parts.push(`placed to ±${geo.rms_metres.toFixed(1)} m`);
  } else if (geo.method === "imported") {
    // An imported transform carries no residuals to report - it was fitted
    // elsewhere - so the honest statement is how it was placed, not a
    // precision figure we do not have.
    parts.push("aligned to a georeferenced sheet");
  }
  metaEl.textContent = parts.join(" · ");

  if (!features.length) return;

  if (!cadastralMap) {
    cadastralMap = L.map("cadastralMap", { attributionControl: true });
  }
  // Every layer is rebuilt from the fresh response rather than mutated, so a
  // status change elsewhere can never leave a stale polygon behind.
  if (cadastralLayers) {
    Object.values(cadastralLayers).forEach((l) => cadastralMap.removeLayer(l));
  }
  if (cadastralLayerControl) cadastralMap.removeControl(cadastralLayerControl);

  // --- Layer 1: record status (the default view) ---
  const statusLayer = L.geoJSON(geojson, {
    style: (f) => {
      const linked = f.properties.linked_document;
      const c = parcelColor(linked);
      return { color: c, weight: 2, fillColor: c, fillOpacity: linked ? 0.35 : 0.1 };
    },
    onEachFeature: (f, layer) =>
      layer.bindPopup(parcelPopup(f.properties || {}), PARCEL_POPUP_OPTS),
  });

  // --- Layer 2: land classification, from the linked record's own field ---
  const classesSeen = new Map();
  const landUseLayer = L.geoJSON(geojson, {
    style: (f) => {
      const linked = f.properties.linked_document;
      const bucket = linked ? landClassBucket(linked.land_classification) : null;
      if (bucket) classesSeen.set(bucket.key, bucket);
      const c = bucket ? bucket.color : cssColor("--border", "#24362D");
      return { color: c, weight: 2, fillColor: c, fillOpacity: bucket ? 0.45 : 0.08 };
    },
    onEachFeature: (f, layer) => {
      const linked = f.properties.linked_document;
      const cls = linked && linked.land_classification;
      layer.bindPopup(`<div class="parcel-popup">`
        + `<div class="pp-head"><span class="pp-id">Parcel `
        + `${esc(f.properties.parcel_id)}</span></div>`
        + `<dl class="pp-grid"><dt>Classification</dt><dd>`
        + (cls ? esc(cls) : `<span class="pp-absent">No linked record</span>`)
        + `</dd></dl></div>`, PARCEL_POPUP_OPTS);
    },
  });

  // --- Layer 3: parcels with no digitised record yet ---
  // The operationally useful inverse of the status layer: a revenue office
  // needs to see the gaps in its own coverage, not just what it has done.
  const missing = features.filter((f) => !f.properties.linked_document);
  const missingLayer = L.geoJSON(
    { type: "FeatureCollection", features: missing },
    {
      style: { color: cssColor("--red", "#F2796B"), weight: 2,
               fillColor: cssColor("--red", "#F2796B"),
               fillOpacity: 0.3, dashArray: "5,4" },
      onEachFeature: (f, layer) => layer.bindPopup(
        `<div class="parcel-popup">`
        + `<div class="pp-head"><span class="pp-id">Parcel `
        + `${esc(f.properties.parcel_id)}</span></div>`
        + `<dl class="pp-grid"><dt>Khasra</dt><dd>`
        + (f.properties.khasra_number
            ? esc(f.properties.khasra_number)
            : `<span class="pp-absent">Label not legible</span>`)
        + `</dd></dl>`
        + `<p class="pp-none">No digitised record for this parcel yet.</p></div>`,
        PARCEL_POPUP_OPTS),
    });

  // --- Layer 4: khasra number labels ---
  const labelLayer = L.layerGroup(
    features.filter((f) => f.properties.khasra_number).map((f) => {
      const layer = L.geoJSON(f);
      const c = layer.getBounds().getCenter();
      return L.marker(c, {
        icon: L.divIcon({ className: "parcel-label",
                          html: esc(f.properties.khasra_number) }),
        interactive: false,
      });
    }));

  // --- Layer 5: the ground control points the georeferencing was fitted to ---
  const gcps = geojson._control_points || [];
  const gcpLayer = L.layerGroup(gcps.map((p, i) =>
    L.circleMarker([p.lat, p.lon], {
      radius: 6, color: "#B18CE0", fillColor: "#B18CE0", fillOpacity: 0.9, weight: 2,
    }).bindPopup(`<div class="parcel-popup">`
      + `<div class="pp-head"><span class="pp-id">Control point ${i + 1}</span></div>`
      + `<dl class="pp-grid"><dt>Position</dt><dd><span class="pp-coord">`
      + `${p.lat}, ${p.lon}</span></dd></dl>`
      + `<p class="pp-none">Demonstration anchor, not a surveyed monument.</p></div>`,
        PARCEL_POPUP_OPTS)));

  // --- Optional basemap. Off by default and labelled as such: every other
  // part of this project works with no internet, and a tile layer silently
  // reaching out to a third party would break that promise without saying
  // so. Turning it on is the viewer's explicit choice.
  // SATELLITE IMAGERY is the one a revenue officer actually wants under a
  // parcel. OpenStreetMap is a street map, and over rural India it is mostly
  // empty - the villages in this corpus are not in it at all. Esri's World
  // Imagery shows the fields themselves, which is what makes a traced
  // boundary checkable by eye and what a georeferencing pass needs to match
  // corners against.
  //
  // These are Esri's public tile endpoints and take no API key.
  const basemaps = {
    "Satellite (Esri)": L.tileLayer(
      "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
      + "MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19,
        attribution: "Imagery &copy; Esri (loaded only when enabled)" }),
    "Topographic (Esri)": L.tileLayer(
      "https://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/"
      + "MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, className: "tiles-drawn",
        attribution: "&copy; Esri (loaded only when enabled)" }),
    "Street map (OpenStreetMap)": L.tileLayer(
      "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 19, className: "tiles-drawn",
        attribution: "&copy; OpenStreetMap contributors (loaded only when enabled)" }),
  };

  // Note the {y}/{x} order on the Esri URLs, against OSM's {x}/{y}. Esri's
  // REST tile scheme addresses by row then column; swapping them returns
  // real tiles of the wrong place, which is worse than an error because the
  // map still looks plausible.

  cadastralLayers = {
    ...basemaps,
    status: statusLayer, landUse: landUseLayer, missing: missingLayer,
    labels: labelLayer, gcp: gcpLayer,
  };

  statusLayer.addTo(cadastralMap);
  // Basemaps go in the FIRST argument, so Leaflet renders them as radio
  // buttons: they are alternatives, not toggles, and stacking two raster
  // basemaps just hides one behind the other. "None" is the default and
  // stays selectable, because every other part of this page works with no
  // internet and a tile layer silently reaching out would break that
  // promise without saying so.
  cadastralLayerControl = L.control.layers(
    { "None (offline)": L.layerGroup().addTo(cadastralMap), ...basemaps },
    {
      "Record status": statusLayer,
      "Land classification": landUseLayer,
      [`Missing records (${missing.length})`]: missingLayer,
      "Khasra numbers": labelLayer,
      [`Control points (${gcps.length})`]: gcpLayer,
    },
    { collapsed: false }
  ).addTo(cadastralMap);

  // The legend follows whichever thematic layer is actually showing.
  const statusLegend = [
    { color: "var(--green)", label: "Approved" },
    { color: "var(--orange)", label: "Needs review" },
    { color: "var(--red)", label: "Blocked" },
    { color: "var(--text-2)", label: "No record yet" },
  ];
  buildLegend(statusLegend);
  cadastralMap.on("overlayadd", (e) => {
    if (e.layer === landUseLayer) {
      buildLegend([...classesSeen.values()].map((b) => ({ color: b.color, label: b.label })));
    } else if (e.layer === statusLayer) {
      buildLegend(statusLegend);
    } else if (e.layer === missingLayer) {
      buildLegend([{ color: "#E56458", label: "Parcel with no digitised record" }]);
    }
  });

  // invalidateSize BEFORE fitBounds, not after. This tab is display:none
  // until it is opened, so on first visit Leaflet still believes the
  // container is the size it was when hidden; fitting to a stale size picks
  // the wrong zoom and centre, and invalidating afterwards keeps that wrong
  // view. Measuring first means the fit is computed against the real box.
  cadastralMap.invalidateSize();
  cadastralMap.fitBounds(statusLayer.getBounds(), { padding: [24, 24] });
}

$("#cadastralMapSelect").addEventListener("change", (e) => {
  cadastralSelectedMap = e.target.value;
  loadCadastralMap();     // fitBounds re-centres, so switching village moves the view
});

async function loadDashboard() {
  const s = await api("/api/stats");

  const byStatus = s.by_status || {};
  const total = s.documents_total || 0;
  const kpis = [
    { v: total, l: "Documents processed", s: `${s.avg_processing_ms || 0} ms average` },
    { v: s.pending_verification || 0, l: "Pending verification",
      s: `${byStatus.blocked || 0} blocked, ${byStatus.needs_review || 0} in review` },
    { v: (s.avg_trust_score || 0).toFixed(0), l: "Average trust score",
      s: `Legibility ${(s.avg_legibility || 0).toFixed(0)}/100` },
    { v: s.extraction_precision === null ? "\u2014" : (s.extraction_precision * 100).toFixed(1) + "%",
      l: "Extraction precision",
      s: s.fields_reviewed ? `${s.fields_reviewed} field${s.fields_reviewed === 1 ? "" : "s"} human-reviewed` : "Awaiting human review" },
    { v: s.corrections_total || 0, l: "Corrections captured", s: "Training signal for the model" },
  ];
  $("#kpiRow").innerHTML = kpis.map((k) => `
    <div class="kpi"><div class="kpi-value">${esc(k.v)}</div>
    <div class="kpi-label">${esc(k.l)}</div><div class="kpi-sub">${esc(k.s)}</div></div>`).join("");

  const statusColors = {
    auto_approved: "var(--green)", approved: "var(--green)",
    needs_review: "var(--orange)", blocked: "var(--red)",
    rejected: "var(--red)", processing: "var(--text-2)",
  };
  barChart($("#chartStatus"),
    Object.keys(byStatus).map((k) => ({
      label: (STATUS_META[k] || { label: k }).label, value: byStatus[k], key: k,
    })),
    (r) => statusColors[r.key] || "var(--blue)");

  barChart($("#chartDistrict"),
    (s.by_district || []).slice(0, 8).map((d) => ({
      label: d.district, value: d.c,
      display: `${d.c} \u00b7 ${Number(d.avg_trust || 0).toFixed(0)}`,
    })));

  barChart($("#chartIssues"),
    (s.top_issues || []).slice(0, 8).map((i) => ({ label: i.rule, value: i.count, sev: i.severity })),
    (r) => r.sev === "error" ? "var(--red)" : r.sev === "warning" ? "var(--orange)" : "var(--blue)");

  barChart($("#chartEngine"),
    (s.by_engine || []).map((e) => ({
      label: readingMethod(e.engine).label, value: e.c,
      display: `${e.c} \u00b7 ${Number(e.avg_trust || 0).toFixed(0)}`,
    })));

  const fa = s.field_accuracy || [];
  $("#fieldQualityTable").innerHTML =
    `<thead><tr><th>Field</th><th class="num">Present</th><th class="num">Missing</th>
     <th class="num">Corrected</th><th class="num">Avg confidence</th><th class="num">Precision</th></tr></thead><tbody>` +
    fa.map((f) => `<tr>
      <td>${esc(f.display || f.field_key)}</td>
      <td class="num">${(f.total || 0) - (f.missing || 0)}</td>
      <td class="num">${f.missing || 0}</td>
      <td class="num">${f.corrected || 0}</td>
      <td class="num">${((f.avg_conf || 0) * 100).toFixed(0)}%</td>
      <td class="num">${f.precision === null || f.precision === undefined
        ? '<span class="muted">\u2014</span>' : (f.precision * 100).toFixed(0) + "%"}</td>
    </tr>`).join("") + "</tbody>";
}

/* ------------------------------------------------------------------ *
 * Learning
 * ------------------------------------------------------------------ */

async function loadLearning() {
  const out = await api("/api/learning");
  const m = out.model || {};

  $("#learnKpis").innerHTML = [
    { v: m.samples || 0, l: "Correction samples" },
    { v: (m.confusions || []).length, l: "Character confusion rules" },
    { v: (m.aliases || []).length, l: "Value aliases" },
    { v: (m.calibration || []).length, l: "Calibrated fields" },
  ].map((k) => `<div class="kpi"><div class="kpi-value">${esc(k.v)}</div>
      <div class="kpi-label">${esc(k.l)}</div></div>`).join("");

  const th = out.thresholds || {};
  $("#confusionList").innerHTML = (m.confusions || []).length
    ? m.confusions.map((c) => `<div class="mini-item">
        <span><code>${esc(c.wrong)}</code> \u2192 <code>${esc(c.right)}</code></span>
        <span class="muted small">seen ${c.support}\u00d7</span></div>`).join("")
    : `<p class="mini-empty">No confusion rule has reached the evidence threshold yet (needs ${th.min_confusion_support || 4} sightings).</p>`;

  $("#aliasList").innerHTML = (m.aliases || []).length
    ? m.aliases.map((a) => `<div class="mini-item">
        <span><code>${esc(a.wrong)}</code> \u2192 <code>${esc(a.right)}</code>
        <span class="muted small">${esc(a.field_key)}</span></span>
        <span class="muted small">${a.support}\u00d7</span></div>`).join("")
    : `<p class="mini-empty">No alias learned yet (needs ${th.min_alias_support || 3} identical corrections).</p>`;

  $("#calibList").innerHTML = (m.calibration || []).length
    ? m.calibration.map((c) => `<div class="mini-item">
        <span>${esc(c.field_key)}</span>
        <span class="muted small">confidence \u00d7 ${Number(c.multiplier).toFixed(2)}
        (${c.reviewed} reviewed, ${(Number(c.observed_precision) * 100).toFixed(0)}% precise)</span></div>`).join("")
    : `<p class="mini-empty">Calibration starts after ${th.min_calibration_sample || 8} reviewed fields per field type.</p>`;

  const rc = out.recent_corrections || [];
  $("#correctionTable").innerHTML =
    `<thead><tr><th>When</th><th>Field</th><th>Extracted</th><th>Corrected to</th><th>Conf</th><th>Engine</th></tr></thead><tbody>` +
    (rc.length ? rc.map((c) => `<tr>
      <td class="muted small">${fmtTime(c.at)}</td>
      <td>${esc(c.field_key)}</td>
      <td class="cell-trunc">${esc(c.ai_value || "\u2014")}</td>
      <td class="cell-trunc">${esc(c.human_value || "\u2014")}</td>
      <td class="num">${c.ai_confidence === null ? "\u2014" : (Number(c.ai_confidence) * 100).toFixed(0) + "%"}</td>
      <td class="muted small">${esc(readingMethod(c.ocr_engine).label)}</td></tr>`).join("")
      : `<tr><td colspan="6" class="muted">No corrections recorded yet.</td></tr>`) + "</tbody>";
}

$("#retrainBtn").addEventListener("click", async () => {
  try {
    const out = await api("/api/learning/retrain", { method: "POST" });
    toast(`Model retrained on ${out.model.samples} correction sample(s)`, "ok");
    loadLearning();
  } catch (err) { toast(err.message, "err"); }
});

/* ------------------------------------------------------------------ *
 * Audit
 * ------------------------------------------------------------------ */

async function loadAudit() {
  const out = await api("/api/audit?limit=200");
  const rows = out.entries || [];
  $("#auditTable").innerHTML =
    `<thead><tr><th>When</th><th>Actor</th><th>Role</th><th>Action</th><th>Doc</th><th>Detail</th></tr></thead><tbody>` +
    (rows.length ? rows.map((e) => `<tr>
      <td class="muted small">${fmtTime(e.at)}</td>
      <td>${esc(e.username || "system")}</td>
      <td><span class="badge gray">${esc(e.role || "\u2014")}</span></td>
      <td>${esc(e.action)}</td>
      <td class="num">${e.document_id || "\u2014"}</td>
      <td class="muted small">${esc(
        [e.field_key, e.old_value && e.new_value ? `${e.old_value} \u2192 ${e.new_value}` : null, e.detail]
          .filter(Boolean).join(" \u00b7 "))}</td>
    </tr>`).join("") : `<tr><td colspan="6" class="muted">No audit entries yet.</td></tr>`) + "</tbody>";
}

/* ------------------------------------------------------------------ *
 * Authentication with Supabase
 * ------------------------------------------------------------------ */

/**
 * Does this server actually require a signed-in user?
 *
 * Asked before the sign-in gate runs. A server with no JWT secret
 * configured cannot verify a token and accepts the request anyway, so
 * sending the operator to a login page it will not check is a gate with
 * nothing behind it - and on a laptop with no Supabase project reachable
 * it makes the whole application unopenable.
 *
 * Fails CLOSED: if the question cannot be answered, the gate runs. An
 * unreachable server is not evidence that sign-in is unnecessary.
 */
async function authIsEnforced() {
  try {
    const res = await fetch("/api/auth/status", { cache: "no-store" });
    if (!res.ok) return true;
    const status = await res.json();
    return status.enforced !== false;
  } catch (e) {
    return true;
  }
}

async function checkAuthentication() {
  if (!(await authIsEnforced())) {
    console.log("Auth is not enforced by this server (no JWT secret "
              + "configured); continuing without sign-in.");
    return true;
  }
  try {
    // Wait for Supabase to load
    let attempts = 0;
    while (typeof window.supabase === 'undefined' && attempts < 50) {
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }

    // Check if Supabase is available after waiting
    if (typeof window.supabase === 'undefined') {
      console.error('Supabase library failed to load');
      window.location.href = 'login.html';
      return false;
    }

    // Initialize Supabase if not already done
    if (typeof window.initSupabase === 'function') {
      window.initSupabase();
    }

    // Check if SupabaseAuth is available
    if (typeof window.SupabaseAuth === 'undefined') {
      console.error('SupabaseAuth not initialized');
      window.location.href = 'login.html';
      return false;
    }

    const session = await SupabaseAuth.getSession();

    if (!session) {
      // Not logged in, redirect to login page
      console.log('No active session found');
      window.location.href = 'login.html';
      return false;
    }

    const user = await SupabaseAuth.getUser();

    // Store user info in localStorage
    localStorage.setItem('supabase_user', JSON.stringify(user));
    localStorage.setItem('user_email', user.email);
    localStorage.setItem('user_role', user.user_metadata?.role || 'operator');

    console.log('Authentication successful:', user.email);
    return true;
  } catch (error) {
    console.error('Authentication check failed:', error);
    // Clear any stale data
    localStorage.removeItem('supabase_user');
    localStorage.removeItem('user_email');
    localStorage.removeItem('user_role');
    window.location.href = 'login.html';
    return false;
  }
}

// Add logout functionality
function setupLogout() {
  const topbarRight = document.querySelector('.topbar-right');

  // Add logout button
  const logoutBtn = document.createElement('button');
  logoutBtn.className = 'btn';
  logoutBtn.textContent = 'Logout';
  logoutBtn.style.marginLeft = '12px';

  logoutBtn.addEventListener('click', async () => {
    try {
      await SupabaseAuth.signOut();
      localStorage.clear();
      window.location.href = 'login.html';
    } catch (error) {
      toast('Logout failed: ' + error.message, 'err');
    }
  });

  topbarRight.appendChild(logoutBtn);
}

// Listen for auth state changes
if (typeof window.SupabaseAuth !== 'undefined') {
  SupabaseAuth.onAuthStateChange((event, session) => {
    if (event === 'SIGNED_OUT') {
      localStorage.clear();
      window.location.href = 'login.html';
    } else if (event === 'TOKEN_REFRESHED') {
      console.log('Token refreshed');
    }
  });
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

(async function boot() {
  // Check authentication first
  const isAuthenticated = await checkAuthentication();

  if (!isAuthenticated) {
    return; // Will redirect to login
  }

  try {
    await loadSession();
    setupLogout(); // Add logout button after session loads
    const sc = await api("/api/schema");
    state.schema = sc.fields || [];
    await loadDocuments();
    routeHash();
  } catch (err) {
    toast("Could not reach the API: " + err.message, "err");
  }
})();
