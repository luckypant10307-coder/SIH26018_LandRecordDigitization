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

WHERE AN INTERNATIONAL STANDARD DOES APPLY

ISO 19152, the Land Administration Domain Model (LADM), is the international
standard for exactly this domain, and it already has the concept this module
needs: `LA_Level`, a grouping of spatial units sharing a coherent position in
the register. The level codes here are that idea in a fixed-width identifier,
and a "volume" here is LADM's 3D spatial unit.

Stated precisely, because the distinction is the whole point: this module is
ALIGNED with LADM's concepts and has NOT been tested for conformance against
the standard's schema. Claiming compliance would need the class structure,
the ISO 19152 code lists and a validation suite none of which exist here. The
honest sentence is "modelled on LADM's level and spatial-unit concepts", and
that is a stronger claim than "our own proposal" without being a false one.

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

# A sanity ceiling, not a building code. Its job is to catch a unit mix-up -
# centimetres or feet entered where metres were meant - before the figure
# becomes an elevation that nothing can collide with.
MAX_STOREY_M = 50.0

# How far a declared stack may stand above a remotely sensed envelope before
# it is reported. This is the measurement's own effective resolution; the
# caller adds one storey on top. Tightening it would flag honest records over
# a metre of satellite noise, which is how a check earns a reputation for
# crying wolf and then gets ignored on the day it is right.
MEASURED_TOLERANCE_M = 4.0

# Ground is the datum: its base is 0.0 and levels count outward from it.
GROUND_LEVEL_CODE = "G00"

# The building segment is OPTIONAL and numeric, and both facts are deliberate.
#
# Optional, because omitting it has an exact meaning - "the only building on
# this parcel" - which is the overwhelmingly common rural case, and because
# every identifier minted before the segment existed stays valid and keeps its
# meaning. A format change that invalidated stored identifiers would be a
# migration; this one is not.
#
# Numeric rather than lettered, because a letter would collide with the level
# alphabet. One of the other PS 26011 projects uses B01-B99 for the building
# AND B01-B99 for basements, so `B02` means two different things depending on
# where it sits. Digits cannot be confused with B/G/F/A/S, so the two segments
# stay unambiguous even quoted out of context.
_3D_ULPIN_PATTERN = re.compile(
    r"^([A-Z0-9]{14})(?:-(\d{2}))?-([BGFAS]\d{2})-(\d{3})(?:-([0-9A-Z*]))?$")

# ISO 7064 Mod 37,2 produces one check character for an ALPHANUMERIC body. It
# is opt-in, appended as a final segment, for the case it actually protects
# against: an identifier copied by hand off a document, where a transposition
# would otherwise yield another VALID identifier pointing at a different unit.
#
# Mod 37,2 and not Mod 11,2, which is the variant these schemes are usually
# quoted as. Mod 11,2 is defined over DIGITS with X as the check character,
# and a parcel ULPIN contains letters: feeding them in as values 10-35 pushes
# them past the radix and silently costs the standard's guarantee. Measured on
# this format before the fix, 2 of 183 single-character errors slipped through
# - the ones where a letter's value and a digit's were congruent mod 11. Mod
# 37,2 is the variant ISO specifies for an alphanumeric string, and it detects
# ALL single-character errors and ALL adjacent transpositions.
_CHECK_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ*"


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


def check_character(payload: str) -> str:
    """
    The ISO 7064 Mod 37,2 check character for an identifier body.

    Catches every single-character error and every transposition of adjacent
    characters - which is the failure that matters here, because a transposed
    3D ULPIN is usually still a WELL-FORMED one pointing at a different unit.
    Without a check character there is nothing to notice that.

    Separators are skipped so the character covers the information and not the
    punctuation: adding or dropping the optional building segment must not
    change the check for the parts that remain.
    """
    p = 0
    for ch in payload:
        if ch == "-":
            continue
        p = ((p + _value_of(ch)) * 2) % 37
    return _CHECK_ALPHABET[(38 - p) % 37]


def _value_of(ch: str) -> int:
    """Digits by value, letters as 10-35, which is Mod 37,2's own alphabet."""
    if ch.isdigit():
        return int(ch)
    value = ord(ch) - ord("A") + 10
    if not 10 <= value <= 35:
        raise VerticalError(f"Character {ch!r} is not alphanumeric.")
    return value


def make_3d_ulpin(parcel_ulpin: str, level: str, unit: int,
                  building: Optional[int] = None,
                  with_check: bool = False) -> str:
    """
    The 3D ULPIN for one unit. Deterministic: same inputs, same output.

    Raises rather than guessing. A malformed parent is not repaired here -
    the extractor and the gazetteer already validate the 14-character ULPIN,
    and silently normalising one at this point would hide a reading error
    inside an identifier that then looks authoritative.

    `building` is omitted for a parcel carrying one structure, which is most
    of them, and that omission is meaningful rather than merely shorter: it
    says "the only building here". Passing building=1 is therefore NOT the
    same statement, and the two forms are kept distinct instead of collapsed.
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

    if building is None:
        body = f"{parcel}-{code}-{int(unit):03d}"
    else:
        if not 1 <= int(building) <= 99:
            raise VerticalError(
                f"Building {building} out of range 1-99. A parcel carrying "
                f"more than 99 separate structures needs a survey, not a "
                f"wider field.")
        body = f"{parcel}-{int(building):02d}-{code}-{int(unit):03d}"

    return f"{body}-{check_character(body)}" if with_check else body


def parse_3d_ulpin(value: str) -> Dict[str, object]:
    """
    Split a 3D ULPIN back into its parts.

    Reversibility is a requirement, not a convenience: a reviewer holding a
    disputed identifier has to be able to say which parcel, which level and
    which unit it refers to without a lookup table.
    """
    text = (value or "").strip().upper()
    match = _3D_ULPIN_PATTERN.match(text)
    if not match:
        raise VerticalError(
            f"{value!r} is not a 3D ULPIN. Expected 14 characters, an optional "
            f"two-digit building, a level code and a unit, e.g. "
            f"UP091223700412-F03-012 or UP091223700412-02-F03-012.")
    parcel, building, code, unit, check = match.groups()

    # A check character is verified when present and never demanded, because
    # identifiers minted before the scheme existed do not carry one. Present
    # and wrong is a hard failure: that is precisely the mis-transcription the
    # character exists to catch, and accepting it would make it decoration.
    if check:
        body = text[:text.rindex("-")]
        expected = check_character(body)
        if check != expected:
            raise VerticalError(
                f"{value!r} fails its check character: expected {expected}, "
                f"found {check}. The identifier was most likely mis-copied - "
                f"a transposition usually still looks like a valid ULPIN, "
                f"which is what this character is here to detect.")

    return {
        "parcel_ulpin": parcel,
        "building": int(building) if building else None,
        "level_code": code,
        "level_kind": LEVEL_KINDS[code[0]],
        "level_number": int(code[1:]),
        "unit": int(unit),
        "check_character": check,
    }


def _checked_storey(storey_m: float) -> float:
    """
    A storey height that can actually produce a volume.

    `level_elevation` is the single chokepoint every elevation flows through,
    so validating here covers `stack()` and any direct caller at once. This is
    not a theoretical guard: storey_m arrives from an HTTP request body, and a
    negative value used to build a whole tower whose every top sat BELOW its
    base. Such volumes report no conflict with anything at all, because an
    inverted range shares no height with a real one - so bad input produced a
    building that was structurally incapable of colliding, which is the worst
    possible failure for a conflict detector.
    """
    try:
        storey_m = float(storey_m)
    except (TypeError, ValueError):
        raise VerticalError(f"Storey height {storey_m!r} is not a number.")
    if storey_m != storey_m or storey_m in (float("inf"), float("-inf")):
        raise VerticalError("Storey height must be a finite number of metres.")
    if storey_m <= 0:
        raise VerticalError(
            f"Storey height must be a positive number of metres, not "
            f"{storey_m}. A level with no height, or an inverted one, cannot "
            f"be tested for conflict against anything.")
    if storey_m > MAX_STOREY_M:
        raise VerticalError(
            f"Storey height {storey_m} m exceeds the {MAX_STOREY_M} m sanity "
            f"limit. Check the units: this is metres, not centimetres or feet.")
    return storey_m


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
    storey_m = _checked_storey(storey_m)
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
    building: Optional[int] = None
    # None means "the only building on this parcel", which is a statement and
    # not a missing value. Two volumes on one parcel both with building=None
    # are in the same structure; 1 and 2 are in different ones. It sits among
    # the defaulted fields because `footprint` above it has no default.
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

    @property
    def has_valid_range(self) -> bool:
        """
        Whether this volume occupies any height at all.

        Checked rather than enforced in __post_init__ on purpose: rows arrive
        from a database that an older version of this module wrote, and a
        validation system must REPORT bad stored data, not refuse to load it.
        Raising here would turn one corrupt row into a dead API endpoint.
        """
        return self.top_m > self.base_m

    def to_dict(self) -> dict:
        d = {
            "ulpin_3d": self.ulpin_3d,
            "parcel_ulpin": self.parcel_ulpin,
            "level_code": self.level_code,
            "level_kind": LEVEL_KINDS.get(self.level_code[0], "unknown"),
            "level_number": int(self.level_code[1:]),
            "unit": self.unit,
            "building": self.building,
            # The footprint travels with the volume because a consumer cannot
            # draw, measure or re-check one without it - the 3D view renders
            # each level as an extruded ring, and base_m/top_m alone describe
            # a height with no ground under it.
            "footprint": [list(point) for point in self.footprint],
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
          include_ground: bool = True,
          building: Optional[int] = None,
          footprint_is_parcel: bool = True) -> List[VerticalParcel]:
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
                ulpin_3d=make_3d_ulpin(parcel_ulpin, code, unit,
                                       building=building),
                parcel_ulpin=parcel_ulpin.strip().upper(),
                level_code=code, unit=unit, building=building,
                footprint=list(ring), base_m=base, top_m=top,
                surveyed=False, footprint_is_parcel=footprint_is_parcel))
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

    ASSUMES both ranges are valid (top > base). An inverted range returns 0.0
    here, which reads as "no overlap" and is not a safe answer - so callers
    check `has_valid_range` and report INVALID_Z_RANGE first, as
    `find_conflicts` does. The check is deliberately not duplicated inside this
    function: it stays total and cheap for the O(n^2) pair loop, and the one
    caller that matters validates up front.
    """
    low = max(a.base_m, b.base_m)
    high = min(a.top_m, b.top_m)
    return round(high - low, 3) if high > low else 0.0


def _share_an_undivided_level(a: VerticalParcel, b: VerticalParcel) -> bool:
    """
    Two units on one level of one parcel, neither with a real unit boundary.

    Building-aware, and the cross-building case is the interesting one. Two
    towers on one plot both inherit the WHOLE parcel as their footprint,
    because nothing in the record says where either stands. Their third floors
    therefore share ground and height range on paper while occupying different
    ground in reality - the same unknowability as two flats on a landing, one
    level up. Both are suppressed here and reported by the grouping below.
    """
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
    groups: Dict[Tuple[str, Optional[int], str], List[VerticalParcel]] = {}
    for p in parcels:
        if p.footprint_is_parcel:
            groups.setdefault(
                (p.parcel_ulpin, p.building, p.level_code), []).append(p)

    out: List[dict] = []
    for key in sorted(groups, key=lambda k: (k[0], -1 if k[1] is None else k[1], k[2])):
        parcel_ulpin, building, code = key
        members = groups[key]
        if len(members) < 2:
            continue                 # one unit holding the whole level is fine
        where = (f"Level {code} of {parcel_ulpin}" if building is None
                 else f"Level {code} of building {building:02d} on {parcel_ulpin}")
        out.append({
            "rule": "LEVEL_NOT_PARTITIONED", "severity": "info",
            "parcel_ulpin": parcel_ulpin, "building": building,
            "level_code": code,
            "ulpins": sorted(m.ulpin_3d for m in members),
            "message": (f"{where} holds {len(members)} units, each recorded "
                        f"with the whole parcel as its footprint because the "
                        f"document carries no floor plan. Whether they overlap "
                        f"cannot be determined from this record."),
            "suggestion": ("Attach a floor plan or unit measurements to make "
                           "overlap on this level checkable. Until then the "
                           "units are neither confirmed separate nor "
                           "conflicting."),
        })

    out += _unlocated_buildings(parcels)
    return out


def _unlocated_buildings(parcels: Sequence[VerticalParcel]) -> List[dict]:
    """
    One finding per parcel carrying several buildings none of which is placed.

    Separate from the level finding because it is a different gap with a
    different remedy. A floor plan divides a level; it does not say where a
    tower stands on the plot. When two structures both inherit the entire
    parcel footprint, their matching storeys look co-located and are not - so
    the thing to report is that the BUILDINGS are unlocated, and the fix is a
    site plan rather than a floor plan.
    """
    by_parcel: Dict[str, set] = {}
    for p in parcels:
        if p.footprint_is_parcel and p.building is not None:
            by_parcel.setdefault(p.parcel_ulpin, set()).add(p.building)

    out: List[dict] = []
    for parcel_ulpin, buildings in sorted(by_parcel.items()):
        if len(buildings) < 2:
            continue
        listed = ", ".join(f"{b:02d}" for b in sorted(buildings))
        out.append({
            "rule": "BUILDINGS_NOT_LOCATED", "severity": "info",
            "parcel_ulpin": parcel_ulpin,
            "buildings": sorted(buildings),
            "message": (f"{parcel_ulpin} carries {len(buildings)} separate "
                        f"buildings ({listed}), each recorded with the whole "
                        f"parcel as its footprint because the record does not "
                        f"say where any of them stands. Storeys at the same "
                        f"height in different buildings therefore cannot be "
                        f"compared for overlap."),
            "suggestion": ("Attach a site plan giving each building's own "
                           "footprint. A floor plan does not answer this - it "
                           "divides a storey, not the plot."),
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

    # Corrupt elevations first, because every later answer depends on them.
    # An inverted volume shares no height with anything, so without this it
    # would pass silently AND suppress the overlap it genuinely has - a
    # conflict hidden by bad data rather than reported.
    for p in parcels:
        if not p.has_valid_range:
            out.append({
                "rule": "INVALID_Z_RANGE", "severity": "error",
                "ulpins": [p.ulpin_3d],
                "message": (f"{p.ulpin_3d} has its top at {p.top_m} m and its "
                            f"base at {p.base_m} m, so it encloses no space. "
                            f"Overlap against it cannot be assessed, and a "
                            f"clean result for it would be meaningless."),
                "suggestion": ("Correct the floor level or the storey height "
                               "on this record and regenerate the stack."),
            })

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


def floors_from_height(height_m: float, storey_m: float = DEFAULT_STOREY_M) -> int:
    """
    How many storeys a measured envelope height could hold.

    Deliberately a floor division, so 8.9 m at 3 m a storey is two floors and
    not three. A measured envelope is an upper bound on the building, and
    rounding up would turn evidence that CONSTRAINS a claim into evidence that
    invents a storey.
    """
    storey_m = _checked_storey(storey_m)
    if height_m is None or height_m < 0:
        raise VerticalError("A measured height cannot be negative.")
    return int(height_m // storey_m)


def check_against_measured(parcels: Sequence[VerticalParcel],
                           measured_height_m: Optional[float],
                           storey_m: float = DEFAULT_STOREY_M) -> List[dict]:
    """
    Compare a DECLARED stack against a REMOTELY SENSED envelope height.

    This is the only check in the module backed by an outside measurement, and
    its whole value depends on not overstating what that measurement is. A
    satellite-derived envelope knows how tall the building is, to about 4 m
    resolution. It does not know storeys, it does not know unit boundaries,
    and it is not a survey - so it can contradict a claim of twelve floors on
    a 6 m building and it can never confirm that anyone owns flat 3B.

    Hence one finding, in one direction: the declared stack reaches
    MATERIALLY above what was measured. The reverse - a building taller than
    the floors declared on it - is not reported, because declaring two floors
    of a six-storey block is a perfectly ordinary record. Only one of those is
    evidence of an error.
    """
    if measured_height_m is None or not parcels:
        return []

    storey_m = _checked_storey(storey_m)
    above = [p for p in parcels if p.level_code[0] in ("G", "F")
             and p.has_valid_range]
    if not above:
        return []

    declared_top = max(p.top_m for p in above)
    if declared_top <= 0:
        return []

    # The tolerance is the dataset's own effective resolution plus one storey.
    # Smaller would flag honest records over a metre of satellite noise, which
    # is how a check earns a reputation for crying wolf.
    tolerance = MEASURED_TOLERANCE_M + storey_m
    if declared_top <= measured_height_m + tolerance:
        return []

    could_hold = floors_from_height(measured_height_m, storey_m)
    declared_floors = len({p.level_code for p in above})
    return [{
        "rule": "HEIGHT_CONTRADICTS_FLOORS", "severity": "warning",
        "parcel_ulpin": above[0].parcel_ulpin,
        "declared_top_m": round(declared_top, 2),
        "measured_height_m": round(measured_height_m, 2),
        "measured_could_hold_floors": could_hold,
        "declared_levels_above_ground": declared_floors,
        "message": (f"The declared stack reaches {declared_top:.1f} m above "
                    f"ground, but the building measured on this footprint is "
                    f"about {measured_height_m:.1f} m - roughly "
                    f"{could_hold} level(s) at {storey_m:.1f} m each, against "
                    f"{declared_floors} declared."),
        "suggestion": ("Check the floor count on the record. The measurement "
                       "is a satellite-derived envelope at 4 m resolution, "
                       "not a survey, so it bounds the building rather than "
                       "settling it - but a gap this large usually means the "
                       "declared count is wrong."),
    }]


def describe() -> dict:
    """Capability line for run.py --check."""
    return {
        "format": ("<14-char parcel ULPIN>[-<building>]-<level>-<unit>"
                   "[-<check>], 22 characters without the optional segments"),
        "example": "UP091223700412-F03-012",
        "example_multi_building": "UP091223700412-02-F03-012",
        "example_with_check": make_3d_ulpin("UP091223700412", "F03", 12,
                                           with_check=True),
        "check_scheme": ("ISO 7064 Mod 37,2, optional. Mod 37,2 and not the "
                         "commonly quoted Mod 11,2, because the parent ULPIN "
                         "contains letters and Mod 11,2 is defined over "
                         "digits - measured, 2 of 183 single-character errors "
                         "slipped past it before the switch; Mod 37,2 catches "
                         "all of them and all adjacent transpositions."),
        "levels": dict(LEVEL_KINDS),
        "default_storey_m": DEFAULT_STOREY_M,
        "standard": ("DILRMP 3.0 defines the 14-character parcel ULPIN. The "
                     "vertical suffix is this project's proposal and is not a "
                     "published standard."),
        "modelled_on": ("ISO 19152 (Land Administration Domain Model): the "
                        "level codes follow LADM's LA_Level concept and a "
                        "volume is its 3D spatial unit. Aligned with those "
                        "concepts; NOT tested for schema conformance."),
        "max_storey_m": MAX_STOREY_M,
    }
