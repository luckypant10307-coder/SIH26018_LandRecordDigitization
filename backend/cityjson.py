"""CityJSON import and export for cadastral parcel layers.

The application stores parcel footprints as geographic GeoJSON. CityJSON is
handled here without optional GIS dependencies; unsupported coordinate
reference systems and geometry are rejected instead of being reinterpreted.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Tuple

import cadastral


class CityJSONError(ValueError):
    """Raised when CityJSON is invalid or cannot be represented as parcels."""


_WGS84_REFERENCE = re.compile(r"(?:EPSG/0/)(4326|4979)(?:$|[?#])", re.I)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CityJSONError(f"{label} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise CityJSONError(f"{label} must be finite.")
    return result


def _vertices_and_transform(document: dict) -> List[Tuple[float, float, float]]:
    vertices = document.get("vertices")
    if not isinstance(vertices, list) or not vertices:
        raise CityJSONError("CityJSON must contain a non-empty 'vertices' array.")

    transform = document.get("transform")
    scale = (1.0, 1.0, 1.0)
    translate = (0.0, 0.0, 0.0)
    if transform is not None:
        if not isinstance(transform, dict):
            raise CityJSONError("'transform' must be an object.")
        raw_scale = transform.get("scale")
        raw_translate = transform.get("translate")
        if not isinstance(raw_scale, list) or len(raw_scale) != 3:
            raise CityJSONError("'transform.scale' must contain three numbers.")
        if not isinstance(raw_translate, list) or len(raw_translate) != 3:
            raise CityJSONError("'transform.translate' must contain three numbers.")
        scale = tuple(_finite_number(v, "transform.scale") for v in raw_scale)
        translate = tuple(_finite_number(v, "transform.translate")
                          for v in raw_translate)
        if any(v == 0 for v in scale):
            raise CityJSONError("'transform.scale' values must be non-zero.")

    decoded = []
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, list) or len(vertex) != 3:
            raise CityJSONError(f"Vertex {index} must contain exactly three coordinates.")
        point = tuple(_finite_number(v, f"Vertex {index}") for v in vertex)
        point = tuple(point[i] * scale[i] + translate[i] for i in range(3))
        if not (-180 <= point[0] <= 180 and -90 <= point[1] <= 90):
            raise CityJSONError(
                "CityJSON coordinates are outside longitude/latitude bounds. "
                "Reproject the file to EPSG:4326 or EPSG:4979 before importing.")
        decoded.append(point)
    return decoded


def _closed(ring: List[list]) -> List[list]:
    """A ring with its first position repeated at the end, as GeoJSON wants."""
    return ring + [list(ring[0])] if ring and ring[0] != ring[-1] else ring


def _ring_area(ring: List[Tuple[float, float, float]]) -> float:
    return abs(sum(
        ring[i][0] * ring[(i + 1) % len(ring)][1]
        - ring[(i + 1) % len(ring)][0] * ring[i][1]
        for i in range(len(ring))) / 2.0)


def _surface_boundaries(geometry: dict) -> List[list]:
    geometry_type = geometry.get("type")
    boundaries = geometry.get("boundaries")
    if not isinstance(boundaries, list):
        raise CityJSONError("CityJSON geometry must have a 'boundaries' array.")
    if geometry_type in ("MultiSurface", "CompositeSurface"):
        return boundaries
    if geometry_type == "Solid":
        if not boundaries:
            return []
        shell = boundaries[0]
        if not isinstance(shell, list):
            raise CityJSONError("CityJSON solid has an invalid exterior shell.")
        return shell
    raise CityJSONError(
        f"Unsupported CityJSON geometry type '{geometry_type}'. "
        "Use MultiSurface, CompositeSurface, or Solid parcel geometry.")


def _object_polygon(geometry: dict, vertices: List[Tuple[float, float, float]]):
    candidates = []
    for surface in _surface_boundaries(geometry):
        if not isinstance(surface, list) or not surface:
            raise CityJSONError("CityJSON surface must contain an exterior ring.")
        rings = []
        for ring_indices in surface:
            if not isinstance(ring_indices, list) or len(ring_indices) < 3:
                raise CityJSONError(
                    "CityJSON polygon rings must have at least three indices.")
            ring = []
            for vertex_index in ring_indices:
                if (isinstance(vertex_index, bool)
                        or not isinstance(vertex_index, int)
                        or vertex_index < 0 or vertex_index >= len(vertices)):
                    raise CityJSONError("CityJSON boundary refers to an invalid vertex index.")
                ring.append(vertices[vertex_index])

            # CityJSON rings are IMPLICITLY CLOSED: the first vertex is NOT
            # repeated at the end. This is the exact opposite of GeoJSON, and
            # it is the single easiest thing to get backwards when moving
            # between the two - the spec's own cube example lists four indices
            # per square face, not five, and it inherits the convention from
            # Wavefront OBJ, which it cites.
            #
            # Requiring closure here rejected EVERY file a standard producer
            # emits - cjio, 3dfier, the 3D BAG, FME - while still round-
            # tripping happily with our own exporter, so the tests passed and
            # the feature could not actually exchange data with anything. A
            # repeated first vertex is tolerated rather than refused, because
            # some producers do emit one and dropping it loses nothing.
            if len(ring) > 3 and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                raise CityJSONError(
                    "CityJSON polygon ring collapses to fewer than three "
                    "distinct vertices.")
            rings.append(ring)
        if len(set(rings[0])) < 3:
            continue
        area = _ring_area(rings[0])
        if area > 0:
            candidates.append((area, rings))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], len(candidates)


def loads(data: Any) -> dict:
    """Parse and validate CityJSON input supplied as text, bytes, or a dict."""
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CityJSONError("CityJSON must be UTF-8 encoded.") from exc
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise CityJSONError(f"CityJSON is not valid JSON: {exc.msg}.") from exc
    if not isinstance(data, dict) or data.get("type") != "CityJSON":
        raise CityJSONError("The file is not a CityJSON document.")
    if data.get("version") not in ("1.0", "1.1", "2.0"):
        raise CityJSONError("Supported CityJSON versions are 1.0, 1.1, and 2.0.")
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise CityJSONError("CityJSON 'metadata' must be an object.")
    reference_system = metadata.get("referenceSystem")
    if not isinstance(reference_system, str) or not _WGS84_REFERENCE.search(reference_system):
        raise CityJSONError(
            "CityJSON must declare EPSG:4326 or EPSG:4979 in metadata.referenceSystem.")
    return data


def to_geojson(data: Any, source_file: str = "upload.city.json") -> dict:
    """Convert CityJSON parcel objects to a geographic GeoJSON FeatureCollection."""
    document = loads(data)
    vertices = _vertices_and_transform(document)
    objects = document.get("CityObjects")
    if not isinstance(objects, dict) or not objects:
        raise CityJSONError("CityJSON must contain a non-empty 'CityObjects' object.")

    features = []
    warnings = []
    for object_id, city_object in objects.items():
        if not isinstance(city_object, dict):
            raise CityJSONError(f"CityObject '{object_id}' must be an object.")
        geometries = city_object.get("geometry") or []
        if not isinstance(geometries, list):
            raise CityJSONError(f"CityObject '{object_id}' geometry must be an array.")
        selected = None
        candidate_count = 0
        for geometry in geometries:
            if not isinstance(geometry, dict):
                raise CityJSONError(f"CityObject '{object_id}' has invalid geometry.")
            polygon, count = _object_polygon(geometry, vertices)
            if polygon and (selected is None or _ring_area(polygon[0])
                            > _ring_area(selected[0])):
                selected = polygon
                candidate_count += count
        if selected is None:
            continue

        attributes = city_object.get("attributes")
        if attributes is None:
            attributes = {}
        if not isinstance(attributes, dict):
            raise CityJSONError(f"CityObject '{object_id}' attributes must be an object.")
        props = dict(attributes)
        normalized = {
            re.sub(r"[^a-z0-9]", "", key.casefold()): value
            for key, value in props.items()
        }
        khasra = next((normalized[key] for key in (
            "khasranumber", "khasrano", "khasra", "surveynumber", "surveyno",
            "survey", "plotnumber", "plotno", "parcelno", "parcelid")
                       if key in normalized), None)
        props["parcel_id"] = len(features) + 1
        props["khasra_number"] = (str(khasra).strip()
                                  if khasra is not None and str(khasra).strip()
                                  else None)
        props["cityjson_id"] = str(object_id)
        if "area_m2" in props:
            props["cityjson_declared_area_m2"] = props["area_m2"]
        ring_2d = [(position[0], position[1]) for position in selected[0]]
        centroid = cadastral.polygon_centroid(ring_2d)
        area_m2 = cadastral.polygon_area_m2(ring_2d)
        for hole in selected[1:]:
            area_m2 -= cadastral.polygon_area_m2(
                [(position[0], position[1]) for position in hole])
        props["area_px"] = 0.0
        props["area_m2"] = round(max(0.0, area_m2), 1)
        props["centroid_lon"] = round(centroid[0], 6) if centroid else None
        props["centroid_lat"] = round(centroid[1], 6) if centroid else None
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Polygon",
                # Rings are held unclosed internally, the CityJSON way, so
                # the closing vertex GeoJSON requires is added back here.
                # RFC 7946: "The first and last positions are equivalent, and
                # they MUST contain identical values."
                "coordinates": [_closed([list(position) for position in ring])
                                for ring in selected],
            },
        })
        if candidate_count > 1:
            warnings.append(
                f"CityObject '{object_id}' has multiple polygon surfaces; "
                "only its largest footprint was imported.")

    if not features:
        raise CityJSONError("CityJSON contains no usable parcel polygon surfaces.")
    result = {
        "type": "FeatureCollection",
        "features": features,
        "_georeferencing": {
            "method": "CityJSON",
            "source_file": source_file,
            "crs": document["metadata"]["referenceSystem"],
            "parcels": len(features),
        },
    }
    if warnings:
        result["_warnings"] = warnings
    return result


def to_cityjson(geojson: dict) -> dict:
    """Convert parcel GeoJSON to CityJSON 2.0, marking absent heights as placeholders."""
    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise CityJSONError("Cadastral export requires a GeoJSON FeatureCollection.")
    features = geojson.get("features")
    if not isinstance(features, list) or not features:
        raise CityJSONError("There are no parcel features to export.")

    vertices: List[List[float]] = []
    objects: Dict[str, dict] = {}
    extents = [math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf]
    ids = set()
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise CityJSONError(f"Feature {index} is invalid.")
        geometry = feature.get("geometry") or {}
        if not isinstance(geometry, dict):
            raise CityJSONError(f"Feature {index} geometry must be an object.")
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        polygons = [coordinates] if geometry_type == "Polygon" else coordinates
        if geometry_type not in ("Polygon", "MultiPolygon") or not isinstance(polygons, list):
            raise CityJSONError(f"Feature {index} must be a Polygon or MultiPolygon.")

        city_boundaries = []
        has_height = True
        for polygon in polygons:
            if not isinstance(polygon, list) or not polygon:
                raise CityJSONError(f"Feature {index} has an empty polygon.")
            city_surface = []
            for ring in polygon:
                if not isinstance(ring, list) or len(ring) < 3:
                    raise CityJSONError(f"Feature {index} has an invalid polygon ring.")
                coords = []
                for position in ring:
                    if not isinstance(position, (list, tuple)) or len(position) not in (2, 3):
                        raise CityJSONError(
                            f"Feature {index} positions must contain longitude, latitude, "
                            "and optional height.")
                    lon = _finite_number(position[0], "Longitude")
                    lat = _finite_number(position[1], "Latitude")
                    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                        raise CityJSONError("GeoJSON coordinates are outside WGS84 bounds.")
                    if len(position) == 3:
                        height = _finite_number(position[2], "Height")
                    else:
                        height = 0.0
                        has_height = False
                    coords.append([lon, lat, height])
                # GeoJSON rings arrive CLOSED and CityJSON rings must not be,
                # so the duplicate goes rather than getting another one added.
                # Emitting it repeated produced a doubled vertex in every face
                # - which our own importer then accepted, so nothing here
                # noticed, while other CityJSON readers see a degenerate edge.
                if len(coords) > 3 and coords[0] == coords[-1]:
                    coords.pop()
                if len(coords) < 3:
                    raise CityJSONError(
                        f"Feature {index} has a ring with fewer than three "
                        f"distinct positions.")
                indices = []
                for position in coords:
                    indices.append(len(vertices))
                    vertices.append(position)
                    for axis, value in enumerate(position):
                        extents[axis] = min(extents[axis], value)
                        extents[axis + 3] = max(extents[axis + 3], value)
                city_surface.append(indices)
            city_boundaries.append(city_surface)

        source_properties = feature.get("properties") or {}
        if not isinstance(source_properties, dict):
            raise CityJSONError(f"Feature {index} properties must be an object.")
        properties = dict(source_properties)
        base_id = properties.get("parcel_id", index)
        object_id = f"parcel-{base_id}"
        if object_id in ids:
            object_id = f"{object_id}-{index}"
        ids.add(object_id)
        if not has_height:
            properties["z_is_placeholder"] = True
        objects[object_id] = {
            "type": "GenericCityObject",
            "attributes": properties,
            "geometry": [{
                "type": "MultiSurface",
                "lod": "0",
                "boundaries": city_boundaries,
            }],
        }

    if not vertices:
        raise CityJSONError("There are no parcel vertices to export.")
    return {
        "type": "CityJSON",
        "version": "2.0",
        "metadata": {
            "referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4979",
            "geographicalExtent": extents,
        },
        "CityObjects": objects,
        "vertices": vertices,
    }
