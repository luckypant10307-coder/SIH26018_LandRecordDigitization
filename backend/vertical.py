"""
3D ULPIN generation and vertical parcel geometry.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26011.

THE PROBLEM THIS SOLVES

A 2D cadastre answers "who owns this ground". It cannot answer "who owns the
third floor", "who owns the parking two levels down", or "who owns the air
above the road" - and in a vertical city those are most of the questions. A
single surface parcel may carry a hundred separately-owned volumes stacked
on it, and the existing ULPIN identifies the parcel, not any of them.

WHAT A 3D ULPIN HAS TO BE

  DERIVED      Visibly a child of its surface parcel. An identifier that
               cannot be traced to the land it stands on is useless for
               settling a dispute about that land.
  DETERMINISTIC The same unit must produce the same identifier on every
               machine, every time, with no central allocator - a registry
               that needs a server to mint an ID cannot work in a tehsil
               office with intermittent power.
  REVERSIBLE   Parseable back into parcel, level and unit. An opaque hash
               would satisfy uniqueness and defeat every human who has to
               read it off a document.
  COLLISION-FREE within its parcel, which is all that is required: the
               parent ULPIN already separates parcels from each other.

THE FORMAT

    UP091223700412-F03-012
    └────────────┘ └─┘ └─┘
     parent ULPIN   |   unit within that level, 001-999
     14 chars,      |
     DILRMP 3.0     level: B.. basement, G00 ground, F.. floor,
                           A.. air rights above, S.. subsurface utility

21 characters, fixed width, A-Z0-9 and two hyphens.

The level letters are MNEMONIC, not ordinal: F is floor and B is basement
because that is what a revenue officer reads, and `F` sorts before `G` in
ASCII, so a plain lexicographic sort puts the third floor below the ground
floor. Readability was worth more than a free sort, so the format keeps the
letters and `level_sort_key` provides the physical order instead - bottom to
top, subsurface through air. Claiming the sort came free would have been a
property the format does not have.

THIS IS A PROPOSAL, NOT A PUBLISHED STANDARD, and the code says so wherever
it surfaces. DILRMP 3.0 specifies the 14-character parcel ULPIN; it does not
yet specify a vertical extension. Presenting an invented scheme as an
official one would be the same overclaim this project refuses everywhere
else - so the parent is taken from the official spec and only the suffix is
ours, deliberately separated by a hyphen so the official part stays
extractable.

GEOMETRY WITHOUT A SURVEY

With no LiDAR, no drone imagery and no floor plans, a unit's volume cannot
be measured - so it is DECLARED, from the footprint the cadastral map
already gives and a storey height, and every such volume is marked
`surveyed=False`. A declared volume is useful: it detects two units claiming
the same space, it supports a floor-wise register, and it is honest about
what it is. What it must never do is appear beside a surveyed volume as
though they were the same kind of fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

# The parent, straight from the existing extractor's rules.
PARCEL_ULPIN_PATTERN = re.compile(r"^[A-Z0-9]{14}$")

# Level kinds. The letter is the first character of the level code.
LEVEL_KINDS = {
    "B": "basement",
    "G": "ground",
    "F": "floor",
    "A": "air rights",
    "S": "subsurface utility",
}

# Default storey height in metres, used only when a volume is DECLARED rather
# than surveyed. 3.0 m is the figure the National Building Code uses for a
# habitable storey; it is a convention, not a measurement, and anything built
# on it carries surveyed=False.
DEFAULT_STOREY_M = 3.0

# Ground is the datum: its base is 0.0 and levels count outward from it.
GROUND_LEVEL_CODE = "G00"

_3D_ULPIN_PATTERN = re.compile(r"^([A-Z0-9]{14})-([BGFAS]\d{2})-(\d{3})$")


class VerticalError(ValueError):
    """A 3D ULPIN could not be formed or parsed."""


def level_code(kind: str, number: int = 0) -> str:
    """
    A three-character level code: a kind letter and two digits.

    Ground is always G00 and ignores `number`, because "ground floor 2" is
    not a thing and allowing it would create two spellings of one level -
    the fastest way to put the same volume in a register twice.
    """
    kind = (kind or "").strip().upper()[:1]
    if kind not in LEVEL_KINDS:
        raise VerticalError(
            f"Unknown level kind {kind!r}; expected one of "
            f"{', '.join(f'{k} ({v})' for k, v in LEVEL_KINDS.items())}.")
    if kind == "G":
        return GROUND_LEVEL_CODE
    if not 1 <= int(number) <= 99:
        raise VerticalError(
            f"Level number {number} out of range for kind {kind!r}; "
            f"1-99, and 0 is reserved for ground.")
    return f"{kind}{int(number):02d}"


def make_3d_ulpin(parcel_ulpin: str, level: str, unit: int) -> str:
    """
    The 3D ULPIN for one unit. Deterministic: same inputs, same output.

    Raises rather than guessing. A malformed parent is not repaired here -
    the extractor and the gazetteer already validate the 14-character ULPIN,
    and silently normalising one at this point would hide a reading error
    inside an identifier that then looks authoritative.
    """
    parcel = (parcel_ulpin or "").strip().upper()
    if not PARCEL_ULPIN_PATTERN.match(parcel):
        raise VerticalError(
            f"Parent ULPIN {parcel_ulpin!r} is not 14 alphanumeric characters. "
            f"A 3D ULPIN must descend from a real parcel identifier.")
    code = (level or "").strip().upper()
    if not re.match(r"^[BGFAS]\d{2}$", code):
        raise VerticalError(f"Level code {level!r} is malformed; expected e.g. F03.")
    if not 1 <= int(unit) <= 999:
        raise VerticalError(f"Unit {unit} out of range 1-999.")
    return f"{parcel}-{code}-{int(unit):03d}"


def parse_3d_ulpin(value: str) -> Dict[str, object]:
    """
    Split a 3D ULPIN back into its parts.

    Reversibility is a requirement, not a convenience: a reviewer holding a
    disputed identifier has to be able to say which parcel, which level and
    which unit it refers to without a lookup table.
    """
    match = _3D_ULPIN_PATTERN.match((value or "").strip().upper())
    if not match:
        raise VerticalError(
            f"{value!r} is not a 3D ULPIN. Expected 14 characters, a level "
            f"code and a unit, e.g. UP091223700412-F03-012.")
    parcel, code, unit = match.groups()
    return {
        "parcel_ulpin": parcel,
        "level_code": code,
        "level_kind": LEVEL_KINDS[code[0]],
        "level_number": int(code[1:]),
        "unit": int(unit),
    }


def level_elevation(code: str, storey_m: float = DEFAULT_STOREY_M) -> Tuple[float, float]:
    """
    (base, top) in metres relative to ground, for a DECLARED level.

    Ground is the datum at 0.0. Floors stack upward, basements downward, so a
    basement's base is NEGATIVE - which is the whole point: a parking level
    at -3 m and a shop at +0 m occupy the same footprint and must not collide.

    Air rights sit above the building and subsurface utilities below the
    basements; both are given a nominal extent, because an easement with no
    vertical extent cannot be tested for conflict with anything.
    """
    code = (code or "").strip().upper()
    if not re.match(r"^[BGFAS]\d{2}$", code):
        raise VerticalError(f"Level code {code!r} is malformed.")
    kind, number = code[0], int(code[1:])
    if kind == "G":
        return (0.0, storey_m)
    if kind == "F":
        return (number * storey_m, (number + 1) * storey_m)
    if kind == "B":
        return (-number * storey_m, -(number - 1) * storey_m)
    if kind == "A":
        # Air rights are measured from the top of the declared building, which
        # the caller knows and this function does not, so they are expressed
        # from a nominal 100 m datum and corrected by the caller when the
        # building height is known.
        return (100.0 + (number - 1) * storey_m, 100.0 + number * storey_m)
    # Subsurface utility corridors, below any basement.
    return (-100.0 - number * storey_m, -100.0 - (number - 1) * storey_m)


# Physical order, bottom to top. The letters are mnemonic and do not sort
# this way on their own - see the format note above.
_TIER = {"S": 0, "B": 1, "G": 2, "F": 3, "A": 4}


def level_sort_key(code: str) -> Tuple[int, int]:
    """
    Sort key putting levels in the order a building is read: subsurface,
    basement, ground, floor, air.

    Basements count DOWNWARD as their number rises - B02 is below B01 - so
    the number is negated for that tier. Without it a two-level basement
    lists its lower floor first and a reviewer reading top-to-bottom sees
    the stack inverted exactly where it matters.
    """
    code = (code or "").strip().upper()
    if not re.match(r"^[BGFAS]\d{2}$", code):
        raise VerticalError(f"Level code {code!r} is malformed.")
    kind, number = code[0], int(code[1:])
    return (_TIER[kind], -number if kind in ("B", "S") else number)


@dataclass
class VerticalParcel:
    """One separately-owned volume stacked on a surface parcel."""
    ulpin_3d: str
    parcel_ulpin: str
    level_code: str
    unit: int
    footprint: List[Tuple[float, float]]      # lon/lat ring, from the 2D parcel
    base_m: float
    top_m: float
    surveyed: bool = False                     # True only with real measurement
    footprint_is_parcel: bool = False
    # True when this unit's footprint is simply the WHOLE parcel, inherited
    # because no floor plan exists to say how the level is divided. It is not
    # a detail: two flats on one undivided level are geometrically identical
    # to two owners sold the same flat, so without this flag the detector
    # cannot tell a normal landing from a double allocation and reports every
    # multi-flat floor as a conflict. The difference is a declared fact, not
    # something recoverable from the coordinates.
    owner_name: Optional[str] = None
    document_id: Optional[int] = None
    notes: List[str] = dc_field(default_factory=list)

    @property
    def height_m(self) -> float:
        return round(self.top_m - self.base_m, 3)

    def to_dict(self) -> dict:
        d = {
            "ulpin_3d": self.ulpin_3d,
            "parcel_ulpin": self.parcel_ulpin,
            "level_code": self.level_code,
            "level_kind": LEVEL_KINDS.get(self.level_code[0], "unknown"),
            "level_number": int(self.level_code[1:]),
            "unit": self.unit,
            "base_m": self.base_m,
            "top_m": self.top_m,
            "height_m": self.height_m,
            "surveyed": self.surveyed,
            "footprint_is_parcel": self.footprint_is_parcel,
            "owner_name": self.owner_name,
            "document_id": self.document_id,
            "notes": list(self.notes),
        }
        if not self.surveyed:
            d["notes"] = d["notes"] + [
                "Volume DECLARED from the parcel footprint and a nominal "
                "storey height, not surveyed. Suitable for detecting "
                "conflicting claims; not a measurement of the unit."]
        return d


def stack(parcel_ulpin: str,
          footprint: Sequence[Tuple[float, float]],
          floors_above: int = 0,
          basements: int = 0,
          units_per_level: int = 1,
          storey_m: float = DEFAULT_STOREY_M,
          include_ground: bool = True) -> List[VerticalParcel]:
    """
    Build the declared vertical parcels for one surface parcel.

    The footprint is reused for every level, which is the honest simplification
    when no floor plan exists: a declared stack says "these units occupy this
    ground between these heights", and says nothing about their internal
    division. When a floor plan arrives the footprint per unit is replaced and
    `surveyed` becomes True - the structure does not change.
    """
    ring = [(float(x), float(y)) for x, y in (footprint or [])]
    if len(ring) < 3:
        raise VerticalError("A vertical parcel needs a footprint of at least 3 points.")

    levels: List[str] = []
    for n in range(basements, 0, -1):
        levels.append(level_code("B", n))
    if include_ground:
        levels.append(GROUND_LEVEL_CODE)
    for n in range(1, floors_above + 1):
        levels.append(level_code("F", n))

    out: List[VerticalParcel] = []
    for code in levels:
        base, top = level_elevation(code, storey_m)
        for unit in range(1, max(1, int(units_per_level)) + 1):
            out.append(VerticalParcel(
                ulpin_3d=make_3d_ulpin(parcel_ulpin, code, unit),
                parcel_ulpin=parcel_ulpin.strip().upper(),
                level_code=code, unit=unit,
                footprint=list(ring), base_m=base, top_m=top,
                surveyed=False, footprint_is_parcel=True))
    return out


# --------------------------------------------------------------------------
# Conflict detection in three dimensions
# --------------------------------------------------------------------------

def _rings_overlap(a: Sequence[Tuple[float, float]],
                   b: Sequence[Tuple[float, float]]) -> bool:
    """
    Whether two footprints share area. Bounding-box test only.

    Deliberately crude, and only ever used to find CANDIDATES. The real test
    is ST_Relate in PostGIS, which knows the difference between parcels that
    share an edge and parcels that share an interior - a distinction a
    bounding box cannot make, and getting it wrong reports every neighbour in
    a building as a conflict.
    """
    if not a or not b:
        return False
    ax = [p[0] for p in a]; ay = [p[1] for p in a]
    bx = [p[0] for p in b]; by = [p[1] for p in b]
    return not (max(ax) <= min(bx) or max(bx) <= min(ax)
                or max(ay) <= min(by) or max(by) <= min(ay))


def z_overlap(a: VerticalParcel, b: VerticalParcel) -> float:
    """
    Metres of shared height between two volumes, 0.0 when they do not meet.

    Touching is NOT overlapping. A flat's floor is the ceiling of the flat
    below, so `top == base` is the normal case for every storey in every
    building - treating it as a conflict would flag an entire tower.
    """
    low = max(a.base_m, b.base_m)
    high = min(a.top_m, b.top_m)
    return round(high - low, 3) if high > low else 0.0


def _share_an_undivided_level(a: VerticalParcel, b: VerticalParcel) -> bool:
    """Two units on one level of one parcel, neither with a real unit boundary."""
    return (a.parcel_ulpin == b.parcel_ulpin
            and a.level_code == b.level_code
            and a.footprint_is_parcel and b.footprint_is_parcel)


def _unpartitioned_levels(parcels: Sequence[VerticalParcel]) -> List[dict]:
    """
    One finding per level that holds several units of unknown extent.

    Grouped per level rather than per pair on purpose: ten flats on a landing
    make 45 pairs, and 45 copies of the same sentence is how a reviewer learns
    to scroll past findings.
    """
    groups: Dict[Tuple[str, str], List[VerticalParcel]] = {}
    for p in parcels:
        if p.footprint_is_parcel:
            groups.setdefault((p.parcel_ulpin, p.level_code), []).append(p)

    out: List[dict] = []
    for (parcel_ulpin, code), members in sorted(groups.items()):
        if len(members) < 2:
            continue                 # one unit holding the whole level is fine
        out.append({
            "rule": "LEVEL_NOT_PARTITIONED", "severity": "info",
            "parcel_ulpin": parcel_ulpin, "level_code": code,
            "ulpins": sorted(m.ulpin_3d for m in members),
            "message": (f"Level {code} of {parcel_ulpin} holds "
                        f"{len(members)} units, each recorded with the whole "
                        f"parcel as its footprint because the document carries "
                        f"no floor plan. Whether they overlap cannot be "
                        f"determined from this record."),
            "suggestion": ("Attach a floor plan or unit measurements to make "
                           "overlap on this level checkable. Until then the "
                           "units are neither confirmed separate nor "
                           "conflicting."),
        })
    return out


def find_conflicts(parcels: Sequence[VerticalParcel]) -> List[dict]:
    """
    Pairs of volumes claiming the same space.

    A conflict needs BOTH a shared footprint and a shared height range. Either
    alone is normal: the flats on one floor share a height range and sit side
    by side, and the flats in one column share a footprint and sit one above
    another. Only the two together mean two owners have been sold the same
    cubic metres.

    THE THIRD ANSWER. Several units on one level, each carrying the whole
    parcel as its footprint because no floor plan exists, are not a conflict
    and are not cleared either - the question is unanswerable from what the
    document gives. Calling that pair an overlap flags every ordinary block of
    flats; calling it clean asserts a separation nobody verified. So it is
    reported once per level as LEVEL_NOT_PARTITIONED, a stated gap rather than
    a verdict in either direction.
    """
    out: List[dict] = []
    for group in _unpartitioned_levels(parcels):
        out.append(group)
    for i in range(len(parcels)):
        for j in range(i + 1, len(parcels)):
            a, b = parcels[i], parcels[j]
            if _share_an_undivided_level(a, b):
                continue            # reported once per level, above
            if a.ulpin_3d == b.ulpin_3d:
                out.append({
                    "rule": "DUPLICATE_3D_ULPIN", "severity": "error",
                    "ulpins": [a.ulpin_3d, b.ulpin_3d],
                    "message": (f"Two volumes share the identifier "
                                f"{a.ulpin_3d}. A 3D ULPIN must be unique "
                                f"within its parcel."),
                    "suggestion": "Renumber one of the units.",
                })
                continue
            shared = z_overlap(a, b)
            if shared <= 0 or not _rings_overlap(a.footprint, b.footprint):
                continue
            out.append({
                "rule": "VERTICAL_OVERLAP", "severity": "error",
                "ulpins": [a.ulpin_3d, b.ulpin_3d],
                "overlap_m": shared,
                "message": (f"{a.ulpin_3d} and {b.ulpin_3d} occupy the same "
                            f"ground and overlap by {shared} m of height."),
                "suggestion": ("Two owners cannot hold the same volume - "
                               "check the floor levels on both records."),
            })
    return out


def describe() -> dict:
    """Capability line for run.py --check."""
    return {
        "format": "<14-char parcel ULPIN>-<level>-<unit>, 21 characters",
        "levels": dict(LEVEL_KINDS),
        "default_storey_m": DEFAULT_STOREY_M,
        "standard": ("DILRMP 3.0 defines the 14-character parcel ULPIN. The "
                     "vertical suffix is this project's proposal and is not a "
                     "published standard."),
    }
