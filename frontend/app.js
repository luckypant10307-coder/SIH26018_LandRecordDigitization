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

async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {});
  if (state.user) headers["X-User"] = state.user.username;
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
  const items = [
    { on: caps.pdf_text_layer, name: "Native PDF text layer",
      note: caps.pdf_text_layer ? "PyMuPDF available" : "PyMuPDF missing" },
    { on: caps.image_preprocessing, name: "Image quality assessment",
      note: caps.image_preprocessing ? "OpenCV deskew + denoise" : "OpenCV missing" },
    { on: caps.tesseract, name: "Tesseract OCR",
      note: caps.tesseract
        ? "Languages: " + (caps.tesseract_languages || []).slice(0, 6).join(", ")
        : "Not installed \u2014 scans queue for manual entry" },
    { on: masterLoaded, name: "LGD administrative master",
      note: masterLoaded ? "District / tehsil cross-check active" : "Master data missing" },
    { on: (samples || []).length > 0, name: "Sample corpus",
      note: (samples || []).length + " documents bundled" },
  ];
  $("#capList").innerHTML = items.map((i) => `
    <li>
      <span class="cap-dot ${i.on ? "on" : "off"}"></span>
      <span class="cap-name">${esc(i.name)}</span>
      <span class="cap-note">${esc(i.note)}</span>
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
  $("#uploadStatus").innerHTML = `<p class="status-note"><span class="spinner"></span> Ingesting the bundled sample corpus\u2026</p>`;
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
      <td><span class="badge gray">${esc(r.engine)}</span></td>
      <td class="num">${s.fields_extracted || 0}/${s.fields_total || 0}</td>
      <td class="num">${(r.trust_score || 0).toFixed(0)}</td>
      <td class="num">${(r.error_count || 0)}e / ${(r.warning_count || 0)}w</td>
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

$("#exportCsvBtn").addEventListener("click", () => {
  const url = "/api/export/csv";
  fetch(url, { headers: { "X-User": state.user.username } })
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
      <td><span class="badge gray">${esc(d.ocr_engine || "\u2014")}</span></td>
      <td class="num">${d.trust_score === null ? "\u2014" : Number(d.trust_score).toFixed(0)}</td>
      <td class="num">${d.error_count || 0}e / ${d.warning_count || 0}w</td>
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

async function openDocument(id) {
  state.currentId = id;
  const doc = await api(`/api/documents/${id}`);
  state.currentDoc = doc;
  $("#wsEmpty").hidden = true;
  $("#wsBody").hidden = false;

  $("#wsTitle").textContent = doc.filename;
  const q = doc.quality || {};
  $("#wsMeta").innerHTML =
    `Document #${doc.id} &middot; ${esc(doc.ocr_engine || "\u2014")} &middot; ` +
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
      label: e.engine, value: e.c,
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
      <td class="muted small">${esc(c.ocr_engine || "\u2014")}</td></tr>`).join("")
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

async function checkAuthentication() {
  try {
    // Check if Supabase is available
    if (typeof window.SupabaseAuth === 'undefined' || typeof window.supabase === 'undefined') {
      console.error('Supabase not loaded');
      window.location.href = 'login.html';
      return false;
    }

    const session = await SupabaseAuth.getSession();

    if (!session) {
      // Not logged in, redirect to login page
      window.location.href = 'login.html';
      return false;
    }

    const user = await SupabaseAuth.getUser();

    // Store user info in localStorage
    localStorage.setItem('supabase_user', JSON.stringify(user));
    localStorage.setItem('user_email', user.email);
    localStorage.setItem('user_role', user.user_metadata?.role || 'operator');

    return true;
  } catch (error) {
    console.error('Authentication check failed:', error);
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
if (window.SupabaseAuth) {
  SupabaseAuth.onAuthStateChange((event, session) => {
    if (event === 'SIGNED_OUT') {
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
