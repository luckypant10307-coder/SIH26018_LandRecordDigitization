/* Intelligent Land Record Digitization & Validation System - SIH 2026, PS 26018
 *
 * React port: application root, tab routing and mount.
 */

"use strict";

const TABS = [
  { id: "ingest",    label: "Ingest" },
  { id: "queue",     label: "Verification Queue", count: true },
  { id: "workspace", label: "Verification Workspace" },
  { id: "cadastral", label: "Cadastral Map" },
  { id: "dashboard", label: "Dashboard" },
  { id: "learning",  label: "Learning" },
  { id: "audit",     label: "Audit Trail" },
];

const TAB_IDS = TABS.map((t) => t.id);

/* Hash routing, ported from routeHash().
 *
 * A hash addresses a tab ("#dashboard") or a specific record
 * ("#workspace/7"), so a reviewer can be sent straight to one document. The
 * cadastral map's popups link this way, which is why it has to keep working.
 */
function parseHash() {
  const parts = location.hash.replace("#", "").split("/");
  const tab = TAB_IDS.indexOf(parts[0]) !== -1 ? parts[0] : "ingest";
  const docId = parts[1] ? Number(parts[1]) : null;
  return { tab, docId: Number.isFinite(docId) ? docId : null };
}

function useHashRoute() {
  const [route, setRoute] = useState(parseHash);
  useEffect(() => {
    const onChange = () => setRoute(parseHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((tab, docId) => {
    const hash = "#" + tab + (docId != null ? "/" + docId : "");
    if (location.hash !== hash) location.hash = hash;   // fires hashchange
    else setRoute({ tab, docId: docId != null ? docId : null });
  }, []);
  return [route, navigate];
}

/* ------------------------------------------------------------------ *
 * Toasts
 * ------------------------------------------------------------------ */

function useToasts() {
  const [toasts, setToasts] = useState([]);
  const nextId = useRef(1);
  const toast = useCallback((msg, kind) => {
    const id = nextId.current++;
    setToasts((prev) => prev.concat([{ id, msg, kind }]));
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4200);
  }, []);
  return [toasts, toast];
}

/* ------------------------------------------------------------------ *
 * Root
 * ------------------------------------------------------------------ */

function App() {
  const [route, navigate] = useHashRoute();
  const [toasts, toast] = useToasts();
  const [username, setUsername] = useState(null);
  const [filters, setFilters] = useState({ search: "", status: "" });
  const [debounced, setDebounced] = useState({ search: "", status: "" });

  // Session first: everything else needs the username for its X-User header.
  const session = useAsync(() => apiCall("/api/session", {}, username), [username]);
  const schema = useAsync(() => apiCall("/api/schema", {}, username), [username]);

  // 220 ms, same as the vanilla app's searchTimer, so typing does not fire a
  // request per keystroke.
  useEffect(() => {
    const t = setTimeout(() => setDebounced(filters), 220);
    return () => clearTimeout(t);
  }, [filters.search, filters.status]);

  const docsState = useAsync(
    () => apiCall(`/api/documents?status=${debounced.status}`
      + `&search=${encodeURIComponent(debounced.search.trim())}&limit=200`, {}, username),
    [debounced.search, debounced.status, username]);

  const docs = (docsState.data && docsState.data.documents) || [];
  const rights = (session.data && session.data.rights) || [];
  const user = session.data && session.data.user;

  const can = useCallback((right) => rights.indexOf(right) !== -1, [rights]);
  const ctx = useMemo(
    () => ({ can, toast, username: user ? user.username : null, user }),
    [can, toast, user]);

  const pending = docs.filter(
    (d) => d.status === "needs_review" || d.status === "blocked").length;

  if (session.error) {
    return h("div", { className: "boot-error" },
      h("h2", null, "Could not reach the API"),
      h("p", null, session.error),
      h("p", { className: "muted small" },
        "Start the backend with: python backend/server.py"));
  }
  if (!session.data || !schema.data) return h(Spinner, { label: "Starting…" });

  const openDoc = (id) => navigate("workspace", id);

  return h(AppContext.Provider, { value: ctx },
    h("header", { className: "topbar" },
      h("div", { className: "topbar-left" },
        h("h1", null, "Land Record Digitization & Validation"),
        h("span", { className: "muted small" }, "SIH 2026 · PS 26018 · React UI")),
      h("div", { className: "topbar-right" },
        h("select", {
          className: "input", value: user.username,
          onChange: (e) => {
            setUsername(e.target.value);
            toast("Switched user");
          },
        }, (session.data.users || []).map((u) =>
          h("option", { key: u.username, value: u.username }, u.full_name))),
        h("span", { className: "role-chip" },
          user.role.charAt(0).toUpperCase() + user.role.slice(1)))),

    h("nav", { className: "tabs", role: "tablist" }, TABS.map((t) =>
      h("button", {
        key: t.id,
        className: "tab" + (route.tab === t.id ? " active" : ""),
        role: "tab",
        "aria-selected": route.tab === t.id,
        onClick: () => navigate(t.id, t.id === "workspace" ? route.docId : null),
      }, t.label,
        t.count ? h("span", { className: "tab-count" }, pending) : null))),

    h("main", { className: "panel" },
      route.tab === "ingest" ? h(IngestView, {
        session: session.data, onIngested: docsState.reload,
        goTo: (tab) => navigate(tab),
      }) : null,
      route.tab === "queue" ? h(QueueView, {
        docs, state: docsState, filters, setFilters, onOpen: openDoc,
      }) : null,
      route.tab === "workspace" ? h(WorkspaceView, {
        docId: route.docId, schema: schema.data.fields || [],
        onChanged: docsState.reload,
      }) : null,
      route.tab === "cadastral" ? h(CadastralView, null) : null,
      route.tab === "dashboard" ? h(DashboardView, null) : null,
      route.tab === "learning" ? h(LearningView, null) : null,
      route.tab === "audit" ? h(AuditView, null) : null),

    h(ToastHost, { toasts }));
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App, null));
