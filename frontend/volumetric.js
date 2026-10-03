/* ------------------------------------------------------------------ *
 * 3D Property View (PS 26011)
 *
 * Each declared volume becomes one extruded slab. No geometry is computed
 * here, and that is the point: a VerticalParcel already carries a footprint
 * ring plus base_m and top_m, which is exactly the pair that
 * `fill-extrusion-base` and `fill-extrusion-height` take. What you see is a
 * direct render of the stored model, not a separate 3D reconstruction of it.
 *
 * ONLY LON/LAT FOOTPRINTS ARE DRAWN. Most footprints in this system come from
 * a parcel map that carried no control points, so they are in PIXELS of that
 * image. Those locate nothing on the earth - (246, 20) is a perfectly valid
 * coordinate pair in the Atlantic - so they are filtered out and the panel
 * explains itself rather than placing a building at sea.
 *
 * Loaded after app.js, and uses its $, $$, esc, api and byLevel helpers.
 * ------------------------------------------------------------------ */

const VOL_SOURCE = "volumes";
const VOL_LAYER = "volume-extrusions";

const volState = {
  map: null,
  byParcel: {},
  current: null,      // selected parcel ULPIN
  level: null,        // selected level code, or null for the whole stack
  shift: 0,           // metres added so basements stay above zero
  imagery: [],        // locally-held drone orthomosaics, with their corners
};

const VOL_COLOURS = {
  B: "#E9B45C",       // basement
  S: "#B98BE0",       // subsurface utility
  G: "#4FC98A",       // ground
  F: "#5AB0F2",       // floor
  A: "#F2796B",       // air rights
};

const VOL_KINDS = {
  B: "basement", S: "subsurface utility", G: "ground",
  F: "floor", A: "air rights",
};

function volColour(code) {
  // Coloured by level KIND rather than by owner: the question this view
  // answers is "what is at this height", and a basement reads differently
  // from an air-rights envelope.
  return VOL_COLOURS[code[0]] || "#94A89C";
}

function volLooksGeographic(ring) {
  if (!ring || ring.length < 3) return false;
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  if (!lons.every((x) => x >= -180 && x <= 180)) return false;
  if (!lats.every((y) => y >= -90 && y <= 90)) return false;
  // Half a degree is about 55 km. Anything wider is a pixel ring that happens
  // to fall inside the valid range, not a plot of land.
  return (Math.max.apply(null, lons) - Math.min.apply(null, lons)) < 0.5
      && (Math.max.apply(null, lats) - Math.min.apply(null, lats)) < 0.5;
}

function volFeatures() {
  const list = volState.byParcel[volState.current] || [];
  const shown = volState.level
    ? list.filter((p) => p.level_code === volState.level)
    : list;
  return {
    type: "FeatureCollection",
    features: shown.map((p) => ({
      type: "Feature",
      properties: {
        ulpin_3d: p.ulpin_3d,
        level_code: p.level_code,
        unit: p.unit,
        base: p.base_m + volState.shift,
        top: p.top_m + volState.shift,
        colour: volColour(p.level_code),
      },
      geometry: {
        type: "Polygon",
        // GeoJSON wants a closed ring; the stored footprint is not closed.
        coordinates: [p.footprint.concat([p.footprint[0]])],
      },
    })),
  };
}

function volRedraw() {
  const map = volState.map;
  if (!map || !map.getSource || !map.getSource(VOL_SOURCE)) return;
  map.getSource(VOL_SOURCE).setData(volFeatures());
}

function volFit() {
  const list = volState.byParcel[volState.current] || [];
  if (!list.length || !volState.map) return;
  const ring = list[0].footprint;
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  volState.map.fitBounds(
    [[Math.min.apply(null, lons), Math.min.apply(null, lats)],
     [Math.max.apply(null, lons), Math.max.apply(null, lats)]],
    { padding: 150, pitch: 55, bearing: -20, duration: 700, maxZoom: 19 });
}

async function volAddLocalImagery() {
  /*
   * Lay any locally-held drone orthomosaic over the satellite basemap.
   *
   * This is the answer to a blurry close-up rather than a decoration. Esri's
   * imagery runs out at zoom 18 over rural India, so a 23 m building is a
   * handful of stretched pixels. The drone survey of that same ground is
   * 2.2 cm - around five hundred times finer - and it is the imagery these
   * parcels were digitised from, so the boundaries line up with what is
   * underneath them instead of merely being near it.
   *
   * A MapLibre `image` source takes four corner coordinates, which the
   * GeoTIFF already gives exactly, so nothing is being positioned by hand.
   * Missing entirely when the imagery has not been fetched - the satellite
   * basemap is always underneath, so this only ever adds.
   */
  let index;
  try {
    const response = await fetch("imagery/index.json");
    if (!response.ok) return;
    index = await response.json();
  } catch (e) {
    return;                       // no local imagery; the basemap stands alone
  }

  Object.keys(index).forEach((name) => {
    const scene = index[name];
    const id = "drone-" + name;
    if (volState.map.getSource(id)) return;
    volState.map.addSource(id, {
      type: "image",
      url: scene.url,
      coordinates: scene.coordinates,
    });
    volState.map.addLayer({
      id: id, type: "raster", source: id,
      paint: { "raster-opacity": 1, "raster-fade-duration": 200 },
    });
    volState.imagery.push({ id: id, coordinates: scene.coordinates,
                            gsd_cm: scene.gsd_cm });
  });
}

function volImageryUnder(ring) {
  /* Which local orthomosaic, if any, covers this parcel. */
  if (!ring || !ring.length) return null;
  const lon = ring.reduce((a, p) => a + p[0], 0) / ring.length;
  const lat = ring.reduce((a, p) => a + p[1], 0) / ring.length;
  return volState.imagery.find((scene) => {
    const lons = scene.coordinates.map((p) => p[0]);
    const lats = scene.coordinates.map((p) => p[1]);
    return lon >= Math.min.apply(null, lons) && lon <= Math.max.apply(null, lons)
        && lat >= Math.min.apply(null, lats) && lat <= Math.max.apply(null, lats);
  }) || null;
}

function volEnsureMap() {
  if (volState.map) return volState.map;

  volState.map = new maplibregl.Map({
    container: "volMap",
    // Esri's public imagery - the same endpoint the 2D map already offers and
    // the same bargain: a tile request carries coordinates and nothing from
    // the record.
    style: {
      version: 8,
      sources: {
        satellite: {
          type: "raster",
          tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/"
                  + "World_Imagery/MapServer/tile/{z}/{y}/{x}"],
          tileSize: 256,
          // Without this, MapLibre keeps requesting deeper tiles and Esri
          // answers with its "Map data not yet available" placeholder, so
          // zooming in on a building made the ground vanish. Declaring the
          // source's real limit makes MapLibre stretch the last good tile
          // instead: blurry and correct beats blank.
          //
          // 18 rather than the 19 the 2D map uses, because coverage is not
          // uniform and this project's subjects are rural. Measured by
          // fetching tiles and comparing them against Esri's placeholder
          // (2,521 bytes, one identical md5 at every depth):
          //
          //     Vansar, the drone site     real imagery to z18
          //     Jaunpur, our own parcels   real imagery to z18
          //     Hinjewadi, Pune            real imagery to z19
          //     Lucknow centre             real imagery to z19
          //
          // At 19 the cities stay sharp and the villages go blank, which is
          // exactly backwards for a rural land-records system. The cost of 18
          // is a softer city view; the cost of 19 is no ground at all over the
          // places this project is actually about.
          maxzoom: 18,
          attribution: "Imagery &copy; Esri",
        },
      },
      layers: [{ id: "satellite", type: "raster", source: "satellite" }],
    },
    center: [78.96, 20.59],
    zoom: 3.2,
    pitch: 55,
    bearing: -20,
  });

  volState.map.addControl(
    new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");

  volState.map.on("load", () => {
    volAddLocalImagery();
    volState.map.addSource(VOL_SOURCE, { type: "geojson", data: volFeatures() });
    volState.map.addLayer({
      id: VOL_LAYER,
      type: "fill-extrusion",
      source: VOL_SOURCE,
      paint: {
        "fill-extrusion-color": ["get", "colour"],
        "fill-extrusion-base": ["get", "base"],
        "fill-extrusion-height": ["get", "top"],
        "fill-extrusion-opacity": 0.82,
      },
    });
    volFit();
  });

  volState.map.on("click", VOL_LAYER, (e) => {
    const f = e.features && e.features[0];
    if (f) volSelectLevel(f.properties.level_code);
  });
  volState.map.on("mouseenter", VOL_LAYER, () => {
    volState.map.getCanvas().style.cursor = "pointer";
  });
  volState.map.on("mouseleave", VOL_LAYER, () => {
    volState.map.getCanvas().style.cursor = "";
  });

  return volState.map;
}

function volSelectLevel(code) {
  volState.level = code;
  volRedraw();
  $$("#volLevelList .vl-row").forEach((b) => {
    b.classList.toggle("active", b.dataset.level === code);
  });

  const list = (volState.byParcel[volState.current] || [])
    .filter((p) => p.level_code === code)
    .sort((a, b) => a.unit - b.unit);

  $("#volDetailCard").hidden = list.length === 0;
  if (!list.length) return;

  const first = list[0];
  $("#volDetailTitle").textContent =
    "Level " + code + " · " + (first.level_kind || "");
  $("#volDetailNote").textContent =
    first.base_m + "–" + first.top_m + " m above ground"
    + (first.surveyed ? "" : " · declared, not surveyed");

  $("#volDetailTable").innerHTML =
    "<thead><tr><th>3D ULPIN</th><th>Unit</th><th>Owner</th></tr></thead><tbody>"
    + list.map((u) => "<tr>"
        + '<td class="mono">' + esc(u.ulpin_3d) + "</td>"
        + '<td class="num">' + u.unit + "</td>"
        + "<td>" + (u.owner_name ? esc(u.owner_name)
                                 : '<span class="muted">—</span>') + "</td>"
        + "</tr>").join("")
    + "</tbody>";
}

function volRenderLevels() {
  const list = volState.byParcel[volState.current] || [];
  const codes = [];
  list.forEach((p) => {
    if (codes.indexOf(p.level_code) === -1) codes.push(p.level_code);
  });
  // byLevel comes from app.js, where it already mirrors the server's
  // level_sort_key. Reversed so the top storey is drawn at the top.
  codes.sort(byLevel).reverse();

  $("#volLevels").hidden = codes.length === 0;
  $("#volLevelList").innerHTML = codes.map((code) => {
    const units = list.filter((p) => p.level_code === code).length;
    return '<button class="vl-row" data-level="' + esc(code) + '">'
      + '<span class="vl-swatch" style="background:' + volColour(code) + '"></span>'
      + '<span class="vl-c">' + esc(code) + "</span>"
      + '<span class="vl-u">' + units + "</span>"
      + "</button>";
  }).join("");
  $$("#volLevelList .vl-row").forEach((b) => {
    b.addEventListener("click", () => volSelectLevel(b.dataset.level));
  });

  const kinds = [];
  list.forEach((p) => {
    if (kinds.indexOf(p.level_code[0]) === -1) kinds.push(p.level_code[0]);
  });
  const scene = volImageryUnder((list[0] || {}).footprint);
  const ground = $("#volGround");
  if (ground) {
    ground.hidden = false;
    ground.textContent = scene
      ? "ground: drone orthomosaic, " + scene.gsd_cm + " cm/px"
      : "ground: Esri satellite, sharp to zoom 18 here";
  }

  $("#volLegend").hidden = kinds.length === 0;
  $("#volLegend").innerHTML = kinds.map((k) =>
    '<span class="vlg"><i style="background:' + volColour(k + "00") + '"></i>'
    + esc(VOL_KINDS[k] || k) + "</span>").join("")
    + (volState.shift
        ? '<span class="vlg-note">shifted +' + volState.shift
          + " m so basements stay visible</span>"
        : "");
}

function volSetParcel(ulpin) {
  volState.current = ulpin;
  volState.level = null;

  // fill-extrusion measures upward from zero, so a basement at -3 m would be
  // clipped away. The whole stack is lifted instead and the legend says by how
  // much: a stated offset beats a building that silently loses its basement.
  const list = volState.byParcel[ulpin] || [];
  let lowest = 0;
  list.forEach((p) => { if (p.base_m < lowest) lowest = p.base_m; });
  volState.shift = lowest < 0 ? -lowest : 0;

  volEnsureMap();
  volRenderLevels();
  $("#volDetailCard").hidden = true;
  volRedraw();
  volFit();
}

async function loadVolumetric() {
  let data;
  try {
    data = await api("/api/vertical");
  } catch (e) {
    $("#volEmpty").hidden = false;
    return;
  }

  const all = data.parcels || [];
  const usable = all.filter((p) => volLooksGeographic(p.footprint));

  volState.byParcel = {};
  usable.forEach((p) => {
    if (!volState.byParcel[p.parcel_ulpin]) volState.byParcel[p.parcel_ulpin] = [];
    volState.byParcel[p.parcel_ulpin].push(p);
  });

  const ulpins = Object.keys(volState.byParcel).sort();
  $("#volParcel").innerHTML = ulpins.map((u) =>
    '<option value="' + esc(u) + '">' + esc(u) + "</option>").join("");

  $("#volEmpty").hidden = ulpins.length > 0;
  if (!ulpins.length) {
    $("#volLevels").hidden = true;
    $("#volLegend").hidden = true;
    $("#volDetailCard").hidden = true;
    const head = $("#volEmpty").querySelector("p");
    head.textContent = all.length
      ? all.length + " volume" + (all.length === 1 ? " is" : "s are")
        + " stored, but none has a footprint in real coordinates."
      : "No vertical volumes have been declared yet.";
    return;
  }

  volSetParcel(ulpins[0]);
}

$("#volParcel").addEventListener("change", (e) => volSetParcel(e.target.value));
$("#volReload").addEventListener("click", () => loadVolumetric());
$("#volShowAll").addEventListener("click", () => {
  volState.level = null;
  volRedraw();
  $$("#volLevelList .vl-row").forEach((b) => b.classList.remove("active"));
  $("#volDetailCard").hidden = true;
});
