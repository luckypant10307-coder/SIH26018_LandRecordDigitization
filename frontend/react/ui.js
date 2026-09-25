/* Intelligent Land Record Digitization & Validation System - SIH 2026, PS 26018
 *
 * React port: the small shared components.
 *
 * These exist because the vanilla app builds the same markup by string
 * concatenation in eight different places - a status badge is written out
 * five times, a table header four. Each one is a chance for two views to
 * drift apart, which they had already started to do.
 */

"use strict";

/* A status pill. The single source of colour truth for the whole UI: the
 * cadastral map's parcel fill reads STATUS_META too, so a parcel and its
 * queue row can never disagree about what "blocked" looks like. */
function Badge({ status, gray, children }) {
  if (gray) return h("span", { className: "badge gray" }, children);
  const meta = STATUS_META[status] || { label: status || EM_DASH, cls: "gray" };
  return h("span", { className: "badge " + meta.cls }, meta.label);
}

function Spinner({ label }) {
  return h("p", { className: "status-note" },
    h("span", { className: "spinner" }), " ", label || "Loading…");
}

/* One place that decides what a view looks like while it is loading, when it
 * has failed, and when it is legitimately empty - so no view can forget to
 * handle one of the three. `empty` is only shown once loading has finished
 * AND no error occurred, which is the case most easily got wrong by hand. */
function Async({ state, empty, children }) {
  if (state.loading && !state.data) return h(Spinner, null);
  if (state.error) {
    return h("div", { className: "issue sev-error" },
      h("span", { className: "issue-rule" }, "UNAVAILABLE"),
      h("div", null, h("div", { className: "issue-msg" }, state.error)));
  }
  if (!state.data) return null;
  const body = children(state.data);
  if (empty && (body === null || (Array.isArray(body) && body.length === 0))) {
    return h("p", { className: "mini-empty" }, empty);
  }
  return body;
}

/* A data table from column definitions, so header and body cannot fall out
 * of step - a real bug class in the vanilla version, where the colspan on an
 * empty-state row is a hand-maintained number. */
function DataTable({ columns, rows, empty, rowKey }) {
  return h("div", { className: "table-wrap" },
    h("table", { className: "data-table" },
      h("thead", null, h("tr", null, columns.map((c, i) =>
        h("th", { key: i, className: c.num ? "num" : null }, c.label)))),
      h("tbody", null,
        rows.length
          ? rows.map((row, r) => h("tr", { key: rowKey ? rowKey(row, r) : r },
              columns.map((c, i) => h("td", {
                key: i,
                className: [c.num ? "num" : null, c.trunc ? "cell-trunc" : null,
                            c.muted ? "muted small" : null].filter(Boolean).join(" ") || null,
                title: c.trunc ? String(c.title ? c.title(row) : "") : null,
              }, c.cell(row)))))
          : h("tr", null, h("td", { colSpan: columns.length, className: "muted" },
              empty || "Nothing to show.")))));
}

function Kpi({ value, label, sub }) {
  return h("div", { className: "kpi" },
    h("div", { className: "kpi-value" }, value),
    h("div", { className: "kpi-label" }, label),
    sub ? h("div", { className: "kpi-sub" }, sub) : null);
}

/* Horizontal bar chart. Ported from barChart() in app.js, including the
 * max-of-values-and-1 guard that stops a division by zero from producing
 * NaN-width bars on an empty dataset. */
function BarChart({ rows, colorFn, empty }) {
  if (!rows.length) return h("p", { className: "mini-empty" }, empty || "No data yet.");
  const max = Math.max.apply(null, rows.map((r) => r.value).concat([1]));
  return h(Fragment, null, rows.map((r, i) =>
    h("div", { className: "bar-row", key: i },
      h("span", { className: "bar-label", title: String(r.label) }, r.label),
      h("span", { className: "bar-track" },
        h("span", { className: "bar-fill", style: {
          width: (r.value / max * 100).toFixed(1) + "%",
          background: colorFn ? colorFn(r) : "var(--blue)",
        } })),
      h("span", { className: "bar-value" },
        r.display !== undefined ? r.display : r.value))));
}

function MiniItem({ children, note }) {
  return h("div", { className: "mini-item" },
    h("span", null, children),
    note ? h("span", { className: "muted small" }, note) : null);
}

/* The toast host. Each toast removes itself after 4.2s, exactly as the
 * vanilla version does with setTimeout + el.remove(). */
function ToastHost({ toasts }) {
  return h("div", { className: "toast-host", id: "toastHost" },
    toasts.map((t) =>
      h("div", { key: t.id, className: "toast" + (t.kind ? " " + t.kind : "") }, t.msg)));
}

/* A modal that renders nothing at all when closed.
 *
 * The vanilla app keeps the reject dialog in the DOM and toggles `hidden`,
 * which leaves its textarea focusable by keyboard while invisible. Not
 * rendering it removes that whole class of problem. */
function Modal({ open, title, children, onCancel, onConfirm, confirmLabel, danger }) {
  if (!open) return null;
  return h("div", { className: "modal-backdrop" },
    h("div", { className: "modal", role: "dialog", "aria-modal": "true" },
      h("h3", null, title),
      children,
      h("div", { className: "modal-actions" },
        h("button", { className: "btn", onClick: onCancel }, "Cancel"),
        h("button", {
          className: "btn " + (danger ? "danger" : "primary"),
          onClick: onConfirm,
        }, confirmLabel || "Confirm"))));
}
