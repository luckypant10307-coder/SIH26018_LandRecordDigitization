/* Intelligent Land Record Digitization & Validation System - SIH 2026, PS 26018
 *
 * React port: the cadastral map.
 *
 * THE ONE PLACE REACT MUST NOT BE IN CHARGE
 * -----------------------------------------
 * Leaflet creates and mutates its own DOM - tile panes, SVG paths, popups -
 * inside the container it is given. React's whole contract is that it owns
 * the children of the elements it renders, so the two cannot both manage the
 * same subtree: React would discard Leaflet's nodes on the next reconcile.
 *
 * So the map <div> is rendered ONCE with no React children, handed to Leaflet
 * via a ref, and everything inside it is driven imperatively from effects.
 * This is the documented escape hatch, not a workaround. The parts React can
 * own - the village selector, the metadata line, the legend - stay as normal
 * components, so only the genuinely imperative part is imperative.
 *
 * Layers are rebuilt wholesale from each response rather than mutated, which
 * is the same decision the vanilla version made and for the same reason: a
 * document approved or rejected in another tab must not be able to leave a
 * stale polygon colour behind.
 */

"use strict";

/* Matches STATUS_META's colour classes so a parcel's fill always means the
 * same thing as the queue and workspace badges. */
function parcelColor(linked) {
  if (!linked) return "#B9B7B2";           // no uploaded document matched yet
  const cls = (STATUS_META[linked.status] || {}).cls;
  if (cls === "green") return "var(--green)";
  if (cls === "orange") return "var(--orange)";
  if (cls === "red") return "var(--red)";
  return "var(--blue)";
}

/* Classification arrives as free text in whatever script the record was
 * written in (सिंचित / असिंचित / irrigated / जिरायत …), so values are
 * bucketed by keyword. An unrecognised value gets its own colour and is
 * still shown - "we could not classify this" is itself information a revenue
 * officer wants on the map, never something to drop silently. */
const LAND_CLASS_BUCKETS = [
  { key: "irrigated",   color: "#2E7D32", test: /सिंचित|सिचित|irrigat|बागायत/i, label: "Irrigated" },
  { key: "unirrigated", color: "#C0864B", test: /असिंचित|असचिति|unirrigat|जिरायत|dry/i, label: "Unirrigated" },
  { key: "barren",      color: "#9E9E9E", test: /बंजर|barren|waste/i, label: "Barren / waste" },
  { key: "residential", color: "#6A4C93", test: /आवासीय|residen|abadi|आबादी/i, label: "Residential" },
];

function landClassBucket(value) {
  if (!value) return null;
  // Unirrigated is tested first on purpose: "असिंचित" CONTAINS "सिंचित", so
  // testing in declaration order would classify every unirrigated parcel as
  // irrigated - the exact inversion this project has already been bitten by
  // once in the extractor's land-class table.
  const un = LAND_CLASS_BUCKETS.find((b) => b.key === "unirrigated");
  if (un.test.test(value)) return un;
  const hit = LAND_CLASS_BUCKETS.find((b) => b.key !== "unirrigated" && b.test.test(value));
  return hit || { key: "other", color: "#2783DE", label: "Other / unclassified" };
}

/* Popup bodies are HTML strings because that is Leaflet's API. They are the
 * only place in this port where markup is assembled by hand, so values are
 * escaped explicitly here - React is not doing it for us inside a popup. */
function escapeHtml(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function statusBadgeHtml(status) {
  const m = STATUS_META[status] || { label: status || EM_DASH, cls: "gray" };
  return `<span class="badge ${m.cls}">${escapeHtml(m.label)}</span>`;
}

function parcelPopup(props) {
  const linked = props.linked_document;
  let body = `<b>Parcel ${escapeHtml(props.parcel_id)}</b><br>`
    + `Khasra: ${escapeHtml(props.khasra_number || "— (label not read)")}`;
  if (props.area_m2) {
    body += `<br>Measured area: ${Number(props.area_m2).toLocaleString()} m²`
      + ` (${(props.area_m2 / 10000).toFixed(3)} ha)`;
  }
  if (props.centroid_lat != null) {
    body += `<br><span class="muted small">${props.centroid_lat.toFixed(5)},`
      + ` ${props.centroid_lon.toFixed(5)}</span>`;
  }
  if (linked) {
    body += `<br>${statusBadgeHtml(linked.status)}`
      + `<br>Owner: ${escapeHtml(linked.owner_name || EM_DASH)}`
      + `<br>Trust: ${linked.trust_score != null ? Math.round(linked.trust_score) : EM_DASH}`
      + `<br><a href="#workspace/${linked.document_id}">Open in verification workspace &rarr;</a>`;
  } else {
    body += `<br><span class="muted small">No uploaded document matched to `
      + `this parcel yet.</span>`;
  }
  return body;
}

const STATUS_LEGEND = [
  { color: "var(--green)", label: "Approved" },
  { color: "var(--orange)", label: "Needs review" },
  { color: "var(--red)", label: "Blocked" },
  { color: "#B9B7B2", label: "No record yet" },
];

function Legend({ entries }) {
  if (!entries || !entries.length) return null;
  return h("div", { className: "legend" }, entries.map((e, i) =>
    h("span", { className: "legend-item", key: i },
      h("span", { className: "legend-swatch", style: { background: e.color } }),
      e.label)));
}

function CadastralView() {
  const { username, can, toast } = useApp();
  const [mapId, setMapId] = useState(null);
  const [legend, setLegend] = useState(STATUS_LEGEND);
  const [mapRefresh, setMapRefresh] = useState(0);
  const [village, setVillage] = useState("");
  const [district, setDistrict] = useState("");
  const [cityjsonFile, setCityjsonFile] = useState(null);
  const [importBusy, setImportBusy] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);
  const cityjsonInputRef = useRef(null);
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef(null);
  const controlRef = useRef(null);

  const maps = useAsync(
    () => apiCall("/api/cadastral/maps", {}, username).then((o) => o.maps || [])
            .catch(() => []),
    [username, mapRefresh]);

  // Default to the first map once the list arrives, without clobbering a
  // choice the viewer has already made.
  useEffect(() => {
    const list = maps.data || [];
    if (list.length && !list.some((m) => m.id === mapId)) setMapId(list[0].id);
  }, [maps.data]);

  const parcels = useAsync(
    () => apiCall("/api/cadastral/parcels"
      + (mapId ? "?map=" + encodeURIComponent(mapId) : ""), {}, username),
    [mapId, username], true);

  async function importCityJSON(event) {
    event.preventDefault();
    if (!cityjsonFile || !can("retrain")) return;
    const form = new FormData();
    form.append("village", village.trim());
    form.append("district", district.trim());
    form.append("cityjson", cityjsonFile, cityjsonFile.name);
    setImportBusy(true);
    try {
      const result = await apiCall(
        "/api/cadastral/cityjson", { method: "POST", body: form }, username);
      setMapId(result.id);
      setMapRefresh((value) => value + 1);
      setVillage("");
      setDistrict("");
      setCityjsonFile(null);
      if (cityjsonInputRef.current) cityjsonInputRef.current.value = "";
      toast(`Imported ${result.parcels} CityJSON parcel(s)`,
            result.warnings && result.warnings.length ? "warn" : "ok");
    } catch (error) {
      toast(error.message, "err");
    } finally {
      setImportBusy(false);
    }
  }

  async function exportCityJSON() {
    if (!mapId || !can("export")) return;
    setExportBusy(true);
    try {
      const response = await fetch(
        "/api/cadastral/cityjson?map=" + encodeURIComponent(mapId),
        { headers: username ? { "X-User": username } : {} });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.error || `Request failed (${response.status})`);
      }
      const blob = await response.blob();
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `cityjson_${mapId}.city.json`;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (error) {
      toast(error.message, "err");
    } finally {
      setExportBusy(false);
    }
  }

  // Build the Leaflet layers. Depends on the geojson only; the container is
  // rendered unconditionally below so the ref is always attached by the time
  // this runs.
  useEffect(() => {
    const geojson = parcels.data;
    if (!geojson || geojson._error || !containerRef.current) return;
    const features = geojson.features || [];
    if (!features.length) return;

    if (!mapRef.current) {
      mapRef.current = L.map(containerRef.current, { attributionControl: true });
    }
    const map = mapRef.current;
    if (layersRef.current) {
      Object.values(layersRef.current).forEach((l) => map.removeLayer(l));
    }
    if (controlRef.current) map.removeControl(controlRef.current);

    const statusLayer = L.geoJSON(geojson, {
      style: (f) => {
        const linked = f.properties.linked_document;
        const c = parcelColor(linked);
        return { color: c, weight: 2, fillColor: c, fillOpacity: linked ? 0.35 : 0.1 };
      },
      onEachFeature: (f, layer) => layer.bindPopup(parcelPopup(f.properties || {})),
    });

    const classesSeen = new Map();
    const landUseLayer = L.geoJSON(geojson, {
      style: (f) => {
        const linked = f.properties.linked_document;
        const bucket = linked ? landClassBucket(linked.land_classification) : null;
        if (bucket) classesSeen.set(bucket.key, bucket);
        const c = bucket ? bucket.color : "#E6E5E3";
        return { color: c, weight: 2, fillColor: c, fillOpacity: bucket ? 0.45 : 0.08 };
      },
      onEachFeature: (f, layer) => {
        const linked = f.properties.linked_document;
        layer.bindPopup(`<b>Parcel ${escapeHtml(f.properties.parcel_id)}</b><br>`
          + `Land classification: `
          + `${escapeHtml((linked && linked.land_classification) || "— no linked record")}`);
      },
    });

    // The operationally useful inverse of the status layer: a revenue office
    // needs to see the gaps in its own coverage, not only what it has done.
    const missing = features.filter((f) => !f.properties.linked_document);
    const missingLayer = L.geoJSON(
      { type: "FeatureCollection", features: missing },
      {
        style: { color: "#E56458", weight: 2, fillColor: "#E56458",
                 fillOpacity: 0.3, dashArray: "5,4" },
        onEachFeature: (f, layer) => layer.bindPopup(
          `<b>Parcel ${escapeHtml(f.properties.parcel_id)}</b><br>`
          + `Khasra: ${escapeHtml(f.properties.khasra_number || "— (label not read)")}<br>`
          + `<span class="muted small">No digitised record for this parcel yet.</span>`),
      });

    const labelLayer = L.layerGroup(
      features.filter((f) => f.properties.khasra_number).map((f) => {
        const centre = L.geoJSON(f).getBounds().getCenter();
        return L.marker(centre, {
          icon: L.divIcon({ className: "parcel-label",
                            html: escapeHtml(f.properties.khasra_number) }),
          interactive: false,
        });
      }));

    const gcps = geojson._control_points || [];
    const gcpLayer = L.layerGroup(gcps.map((p, i) =>
      L.circleMarker([p.lat, p.lon], {
        radius: 6, color: "#6A4C93", fillColor: "#6A4C93", fillOpacity: 0.9, weight: 2,
      }).bindPopup(`<b>Ground control point ${i + 1}</b><br>`
        + `lat ${p.lat}, lon ${p.lon}<br>`
        + `<span class="muted small">Illustrative demo anchor, not a real `
        + `survey point.</span>`)));

    // Off by default and labelled as such: every other part of this project
    // works with no internet, and a tile layer silently reaching out to a
    // third party would retract that without telling anyone.
    const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors (loaded only when enabled)",
    });

    layersRef.current = { statusLayer, landUseLayer, missingLayer, labelLayer, gcpLayer, osm };
    statusLayer.addTo(map);

    controlRef.current = L.control.layers(null, {
      "Record status": statusLayer,
      "Land classification": landUseLayer,
      [`Missing records (${missing.length})`]: missingLayer,
      "Khasra labels": labelLayer,
      [`Ground control points (${gcps.length})`]: gcpLayer,
      "OpenStreetMap basemap (needs internet)": osm,
    }, { collapsed: false }).addTo(map);

    setLegend(STATUS_LEGEND);
    const onOverlayAdd = (e) => {
      if (e.layer === landUseLayer) {
        setLegend([...classesSeen.values()].map((b) => ({ color: b.color, label: b.label })));
      } else if (e.layer === statusLayer) {
        setLegend(STATUS_LEGEND);
      } else if (e.layer === missingLayer) {
        setLegend([{ color: "#E56458", label: "Parcel with no digitised record" }]);
      }
    };
    map.on("overlayadd", onOverlayAdd);

    map.fitBounds(statusLayer.getBounds(), { padding: [20, 20] });
    map.invalidateSize();

    return () => { map.off("overlayadd", onOverlayAdd); };
  }, [parcels.data]);

  // Leaflet measures its container on creation. Mounted inside a tab that was
  // display:none, it reads a zero size and renders a grey box; this is why
  // invalidateSize exists and why it is called again whenever this view is
  // shown, not only when the data changes.
  useEffect(() => {
    if (mapRef.current) mapRef.current.invalidateSize();
  });

  useEffect(() => () => {
    if (mapRef.current) { mapRef.current.remove(); mapRef.current = null; }
  }, []);

  const list = maps.data || [];
  const geojson = parcels.data;
  const features = (geojson && geojson.features) || [];
  const linkedCount = features.filter((f) => f.properties.linked_document).length;
  const geo = (geojson && geojson._georeferencing) || {};

  let meta = "";
  if (parcels.loading) meta = "Loading…";
  else if (geojson && geojson._error) meta = "Unavailable";
  else if (geojson) {
    meta = `${features.length} parcel(s), ${linkedCount} linked to an uploaded document`
      + (geo.max_residual_deg != null
          ? ` · max residual ${geo.max_residual_deg.toFixed(6)}°` : "")
      + (geo.method ? ` · georeferencing: ${geo.method}` : "");
  }

  return h("div", { className: "card" },
    h("div", { className: "card-head" },
      h("h3", null, "Cadastral map"),
      h("div", { className: "row-actions" },
        list.length > 1
          ? h("select", {
              className: "input", value: mapId || "",
              onChange: (e) => setMapId(e.target.value),
            }, list.map((m) =>
              h("option", { key: m.id, value: m.id },
                m.village + (m.district ? " — " + m.district : "")
                + (m.bundled ? " (demo)" : ""))))
          : null,
        h("span", { className: "muted small" }, meta))),

    h("form", { className: "row gap cadastral-cityjson-tools",
      onSubmit: importCityJSON },
      h("label", null,
        h("span", { className: "muted small" }, "Village"),
        h("input", {
          className: "input", required: true, maxLength: 120,
          placeholder: "Village name", value: village,
          onChange: (event) => setVillage(event.target.value),
        })),
      h("label", null,
        h("span", { className: "muted small" }, "District (optional)"),
        h("input", {
          className: "input", maxLength: 120, placeholder: "District",
          value: district, onChange: (event) => setDistrict(event.target.value),
        })),
      h("label", null,
        h("span", { className: "muted small" }, "CityJSON parcel layer"),
        h("input", {
          className: "input", type: "file", required: true,
          ref: cityjsonInputRef,
          accept: ".city.json,.json,application/json",
          onChange: (event) => setCityjsonFile(event.target.files[0] || null),
        })),
      h("button", {
        className: "btn",
        type: "submit",
        disabled: !can("retrain") || !cityjsonFile || importBusy,
      }, importBusy ? "Importing…" : "Import CityJSON"),
      h("button", {
        className: "btn",
        type: "button",
        disabled: !can("export") || !mapId || exportBusy,
        onClick: exportCityJSON,
      }, exportBusy ? "Exporting…" : "Export selected map as CityJSON")),

    geojson && geojson._error
      ? h("p", { className: "status-note" }, geojson._error)
      : null,
    geojson && geojson._disclaimer
      ? h("p", { className: "status-note" }, geojson._disclaimer)
      : null,
    (geojson && !geojson._error && !features.length)
      ? h("p", { className: "mini-empty" },
          "No parcels in this map. Run tools/make_cadastral_map.py for the "
          + "demo village, or add a real map under storage/cadastral/.")
      : null,

    // Rendered unconditionally and with NO React children - Leaflet owns
    // everything inside it. Conditionally rendering this div would destroy
    // the map instance every time the data reloaded.
    h("div", { className: "cadastral-map", ref: containerRef }),
    h(Legend, { entries: legend }));
}
