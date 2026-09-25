/* Intelligent Land Record Digitization & Validation System - SIH 2026, PS 26018
 *
 * React port: shared helpers, API client and hooks.
 *
 * WHY THERE IS NO BUILD STEP AND NO JSX
 * ------------------------------------
 * The whole project's claim is that it runs with nothing to install - stdlib
 * Python, no npm, no bundler, and Leaflet vendored locally rather than pulled
 * from a CDN so it works with no internet. Introducing React the usual way
 * (npm + Vite + node_modules) would quietly retract that.
 *
 * So React is vendored the same way Leaflet already is - the UMD builds sit in
 * frontend/vendor/react/ - and components are written with the `h` helper
 * below instead of JSX. That costs a little readability at the call site and
 * buys the ability to open this file in a browser and have it work. Babel in
 * the browser would have preserved JSX, but it compiles on every page load,
 * which is the wrong trade for a demo that has to start instantly on someone
 * else's laptop.
 *
 * The vanilla frontend in ../app.js is untouched and still served at "/".
 * This port lives at /react.html so the two can be compared side by side and
 * nothing that currently works is put at risk.
 */

"use strict";

/* React.createElement, shortened. `h("div", {className: "x"}, child, child)`
 * is the JSX-free form of `<div className="x">…</div>`. */
const h = React.createElement;
const { useState, useEffect, useMemo, useRef, useCallback, useContext,
        createContext, Fragment } = React;

/* ------------------------------------------------------------------ *
 * API client
 *
 * Identical contract to the vanilla app: the backend identifies the caller
 * from an X-User header, so every request carries the active username.
 * ------------------------------------------------------------------ */

async function apiCall(path, opts = {}, username = null) {
  const headers = Object.assign({}, opts.headers || {});
  if (username) headers["X-User"] = username;
  const options = Object.assign({}, opts);
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.json);
    delete options.json;
  }
  const res = await fetch(path, Object.assign(options, { headers }));
  const ctype = res.headers.get("Content-Type") || "";
  const payload = ctype.includes("application/json")
    ? await res.json()
    : { raw: await res.text() };
  if (!res.ok) {
    throw new Error((payload && payload.error) || `Request failed (${res.status})`);
  }
  return payload;
}

/* ------------------------------------------------------------------ *
 * Formatters - ported unchanged so both frontends render identically.
 *
 * Note there is no esc() here and that is the point: React escapes text
 * children by itself, so the manual escaping the vanilla app needs on every
 * interpolation is structurally unnecessary. Any innerHTML-equivalent
 * (dangerouslySetInnerHTML) is deliberately absent from this port.
 * ------------------------------------------------------------------ */

const EM_DASH = "—";

function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
}

function fmtTime(s) {
  if (!s) return EM_DASH;
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

function confColor(c) {
  if (c >= 0.85) return "var(--green)";
  if (c >= 0.6) return "var(--orange)";
  return "var(--red)";
}

function trustClass(trust) {
  return trust >= 80 ? "good" : trust >= 50 ? "mid" : "bad";
}

/* ------------------------------------------------------------------ *
 * App context: session, rights and the toast queue.
 *
 * A context rather than prop-drilling because `can(right)` and `toast()` are
 * needed at almost every leaf - the queue's Open button, each field row's
 * edit guard, every failed request.
 * ------------------------------------------------------------------ */

const AppContext = createContext(null);

function useApp() {
  return useContext(AppContext);
}

/* ------------------------------------------------------------------ *
 * useAsync - load-on-mount with explicit loading and error states.
 *
 * The vanilla app awaits a fetch and writes innerHTML, so a slow or failed
 * request shows the PREVIOUS tab's contents until it resolves. Modelling
 * loading and error as first-class state is most of what this port buys:
 * every view can say "loading", "failed, here is why", or show data, and can
 * never silently display something stale as if it were current.
 * ------------------------------------------------------------------ */

function useAsync(fn, deps, enabled = true) {
  const [state, setState] = useState({ loading: enabled, error: null, data: null });
  const seq = useRef(0);

  const run = useCallback(() => {
    if (!enabled) return;
    const ticket = ++seq.current;
    setState((s) => ({ loading: true, error: null, data: s.data }));
    Promise.resolve()
      .then(fn)
      .then((data) => {
        // A response that arrives after a newer request was issued is
        // discarded. Without this, switching villages twice quickly can leave
        // the map showing the first village's parcels.
        if (ticket === seq.current) setState({ loading: false, error: null, data });
      })
      .catch((err) => {
        if (ticket === seq.current) {
          setState({ loading: false, error: err.message || String(err), data: null });
        }
      });
  }, deps.concat([enabled]));

  useEffect(run, deps.concat([enabled]));
  return Object.assign({}, state, { reload: run });
}
