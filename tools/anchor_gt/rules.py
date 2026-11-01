"""Rule library for the ANCHOR testbed (design_proposals.md §4.1, v0.3.2 + one AND-only template).

Predicates are 3-valued: True, False or None (None = inside the ignore band U).
Every template carries its program, its primitives (= truth-table cell order), the clauses under
NOT / EXCEPT, its Q-matrix components, and three paraphrases that keep the operational wording. `{S}` is replaced by the subject reference.
"""
import re

# ---------------------------------------------------------------- thresholds (§0.5 / §4.1)
SUBJ_MIN, SUBJ_MAX = 0.02, 0.60   # subject area as a fraction of the image
CAND_MIN = 0.002                  # candidates below this are not marked
MIN_CANDS, MAX_CANDS = 4, 20
SIZE_MARGIN = 1.5                 # larger/smaller: ratio > 1.5 / < 1/1.5, else U
TOUCH_PX = 1.5                    # touch: gap <= 1.5 px True, > FAR_FRAC * diag False, else U
FAR_FRAC = 0.02
POS_MARGIN = 0.02                 # left/above: +-2% of W/H around the subject's edge is U
BORDER_PX = 1                     # border: edge distance <= 1 px True, > BORDER_FRAC * diag False, else U
BORDER_FRAC = 0.01
U_DROP = 0.30                     # drop an instance if more than 30% of its candidates are U


# ---------------------------------------------------------------- 3-valued logic
def AND(*xs):
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


def OR(*xs):
    if any(x is True for x in xs):
        return True
    return None if any(x is None for x in xs) else False


def NOT(a):
    return None if a is None else (not a)


def EXCEPT(a, b):
    """The exception wins: a and not b."""
    return AND(a, NOT(b))


def band(x, lo, hi):
    """True above hi, False below lo, None in between."""
    return True if x > hi else (False if x < lo else None)


# ---------------------------------------------------------------- predicates
# Operational wording shown to the model, both in rule texts and in the probe.
PREDICATES = {
    "same_cat": "is the same kind of object as {S} (for example, both are cars)",
    "larger": "covers more image pixels than {S}",
    "smaller": "covers fewer image pixels than {S}",
    "touch": "has a visible outline that shares a border with the outline of {S} in the image (overlapping in the picture counts, even with no physical contact)",
    "left": "has its centre to the left of the leftmost visible point of {S}",
    "above": "has its centre above the topmost visible point of {S}",
    "border": "reaches the edge of the image",
    # diagnostic only (DIAG_TEMPLATES): everyday wording, GT from the caption
    "contact": "is physically touching {S}: resting on it, leaning on it, or held, worn, carried or ridden by it, or the reverse",
}


def predicates(o, s, W, H, diag):
    """Raw measures and 3-valued predicate values of candidate geometry `o` relative to subject `s`."""
    ratio = o["area"] / s["area"]
    dx = s["x0"] - o["cx"]     # > 0: centre is left of the subject's leftmost point
    dy = s["y0"] - o["cy"]     # > 0: centre is above the subject's topmost point
    gap = o["gap"]
    return dict(
        same_cat=o["cat"] == s["cat"],
        larger=band(ratio, 1 / SIZE_MARGIN, SIZE_MARGIN),
        smaller=band(1 / ratio, 1 / SIZE_MARGIN, SIZE_MARGIN),
        touch=True if gap <= TOUCH_PX else (False if gap > FAR_FRAC * diag else None),
        left=band(dx, -POS_MARGIN * W, POS_MARGIN * W),
        above=band(dy, -POS_MARGIN * H, POS_MARGIN * H),
        border=True if o["edge"] <= BORDER_PX else (False if o["edge"] > BORDER_FRAC * diag else None),
        ratio=round(ratio, 3), gap=round(gap, 1) if gap != float("inf") else None,
        dx=round(dx, 1), dy=round(dy, 1), edge=o["edge"],
    )


def touch_value(gap, diag):
    return True if gap <= TOUCH_PX else (False if gap > FAR_FRAC * diag else None)


def contact_value(gap, stated, diag):
    """Caption-stated contact: True if the caption states it and the outlines meet, False if the masks are
    far apart (they cannot touch), else None. Captions omit most contacts, so silence is never False."""
    if gap is None or gap > FAR_FRAC * diag:
        return False
    return True if stated and gap <= TOUCH_PX else None


# ---------------------------------------------------------------- templates
def _closure(p, cands, cc):
    """CLOSURE_touch(same_cat): same-kind objects, plus anything touching one of them (one hop).

    cc[(a, b)] is the 3-valued touch between candidates a and b.
    """
    seed = {c: p[c]["same_cat"] for c in cands}
    out = {}
    for c in cands:
        if seed[c] is not False:
            out[c] = seed[c]
            continue
        out[c] = OR(*[AND(seed[x], cc.get((min(c, x), max(c, x)), False)) for x in cands if x != c] or [False])
    return out

def _simple(fn):
    return lambda p, cands, cc: {c: fn(p[c]) for c in cands}


# Primitive -> family (§4.1 constraint 1). Components of the Q-matrix (§5.1e) are the
# primitives (larger/smaller share "size"), the operators AND and NOT, the EXCEPT wording,
# and CLOSURE (which also carries the candidate-candidate touch it needs).
FAMILY = {"touch": "topology", "larger": "size", "smaller": "size", "left": "spatial", "above": "spatial",
          "same_cat": "category", "border": "border"}
COMPONENTS = ["touch", "size", "left", "above", "same_cat", "border", "AND", "NOT", "EXCEPT", "CLOSURE"]

# Fields: prims = primitives in cell order; neg / exc = primitives under NOT / EXCEPT;
# fire = the cell where the NOT or EXCEPT clause decides (the "opportunity" cell, §3.2);
# wording = how the operator is phrased (twins differ only here).
TEMPLATES = [
    dict(id="T01", level="L1", name="touch", prims=["touch"], wording="none",
         fn=_simple(lambda q: q["touch"]),
         text=["Select every marked object whose visible outline shares a border with the outline of {S} in the image.",
               "Which marked objects border {S}? An object borders {S} when its visible outline shares a border with the outline of {S} in the image.",
               "List the marked objects that share a border with {S}, meaning their visible outline and the outline of {S} meet in the image."]),
    dict(id="T02", level="L1", name="larger", prims=["larger"], wording="none",
         fn=_simple(lambda q: q["larger"]),
         text=["Select every marked object that covers more image pixels than {S}.",
               "Which marked objects are larger than {S}? Here larger means the object covers more image pixels than {S}, not that it is bigger in real life.",
               "List the marked objects whose visible area in the image, counted in pixels, is greater than that of {S}."]),
    dict(id="T03", level="L1", name="left", prims=["left"], wording="none",
         fn=_simple(lambda q: q["left"]),
         text=["Select every marked object whose centre lies to the left of the leftmost visible point of {S}.",
               "Which marked objects are left of {S}? An object is left of {S} when its centre lies to the left of the leftmost visible point of {S} in the image.",
               "List the marked objects whose centre is further left in the image than the leftmost visible point of {S}."]),
    dict(id="T04", level="L1", name="above", prims=["above"], wording="none",
         fn=_simple(lambda q: q["above"]),
         text=["Select every marked object whose centre lies above the topmost visible point of {S}.",
               "Which marked objects are above {S}? An object is above {S} when its centre lies higher in the image than the topmost visible point of {S}.",
               "List the marked objects whose centre is higher in the image than the topmost visible point of {S}."]),
    dict(id="T05", level="L1", name="same_cat", prims=["same_cat"], wording="none", calibration=True,
         fn=_simple(lambda q: q["same_cat"]),
         text=["Select every marked object that is the same kind of object as {S} (for example, both are cars).",
               "Which marked objects belong to the same object category as {S}?",
               "List the marked objects of the same kind as {S}; for example, if {S} is a car, list the other cars."]),

    dict(id="T06", level="L2", name="smaller AND above", prims=["smaller", "above"], wording="and",
         fn=_simple(lambda q: AND(q["smaller"], q["above"])),
         text=["Select every marked object that covers fewer image pixels than {S} and whose centre lies above the topmost visible point of {S}.",
               "Which marked objects are smaller than {S} (they cover fewer image pixels) and above {S} (their centre lies higher than the topmost visible point of {S})?",
               "List the marked objects that satisfy both conditions: their visible area in pixels is less than that of {S}, and their centre is higher in the image than the topmost visible point of {S}."]),
    dict(id="T07", level="L2", name="left AND border", prims=["left", "border"], wording="and",
         fn=_simple(lambda q: AND(q["left"], q["border"])),
         text=["Select every marked object whose centre lies to the left of the leftmost visible point of {S} and that reaches the edge of the image.",
               "Which marked objects are left of {S} (their centre lies to the left of the leftmost visible point of {S}) and also reach the edge of the image?",
               "List the marked objects that satisfy both conditions: their centre is further left in the image than the leftmost visible point of {S}, and they reach the edge of the image."]),
    dict(id="T08", level="L2", name="left AND NOT same_cat", prims=["left", "same_cat"], neg=["same_cat"],
         wording="and-not", twin="T13", fire=lambda q: q["left"] is True and q["same_cat"] is True,
         fn=_simple(lambda q: AND(q["left"], NOT(q["same_cat"]))),
         text=["Select every marked object whose centre lies to the left of the leftmost visible point of {S} and that is not the same kind of object as {S}.",
               "Which marked objects are left of {S} (their centre lies to the left of the leftmost visible point of {S}) and are not the same kind of object as {S}?",
               "List the marked objects that satisfy both conditions: their centre is further left in the image than the leftmost visible point of {S}, and they are not the same kind of object as {S}."]),
    dict(id="T09", level="L2", name="NOT touch AND larger", prims=["touch", "larger"], neg=["touch"],
         wording="and-not", fire=lambda q: q["larger"] is True and q["touch"] is True,
         fn=_simple(lambda q: AND(NOT(q["touch"]), q["larger"])),
         text=["Select every marked object that covers more image pixels than {S} and whose visible outline does not share a border with the outline of {S}.",
               "Which marked objects are larger than {S} (they cover more image pixels) but do not border {S} (their outline shares no border with the outline of {S})?",
               "List the marked objects that satisfy both conditions: their visible area in pixels is greater than that of {S}, and their visible outline does not meet the outline of {S} anywhere in the image."]),
    dict(id="T10", level="L2", name="border AND NOT left", prims=["border", "left"], neg=["left"],
         wording="and-not", fire=lambda q: q["border"] is True and q["left"] is True,
         fn=_simple(lambda q: AND(q["border"], NOT(q["left"]))),
         text=["Select every marked object that reaches the edge of the image and whose centre does not lie to the left of the leftmost visible point of {S}.",
               "Which marked objects reach the image edge and are not left of {S} (their centre is not to the left of the leftmost visible point of {S})?",
               "List the marked objects that satisfy both conditions: they reach the edge of the image, and their centre is not further left than the leftmost visible point of {S}."]),
    dict(id="T11", level="L2", name="touch AND NOT larger", prims=["touch", "larger"], neg=["larger"],
         wording="and-not", twin="T12", fire=lambda q: q["touch"] is True and q["larger"] is True,
         fn=_simple(lambda q: AND(q["touch"], NOT(q["larger"]))),
         text=["Select every marked object whose visible outline shares a border with the outline of {S} and that does not cover more image pixels than {S}.",
               "Which marked objects border {S} (their visible outline shares a border with the outline of {S}) and do not cover more image pixels than {S}?",
               "List the marked objects that satisfy both conditions: their visible outline meets the outline of {S} in the image, and their visible area in pixels is not greater than that of {S}."]),
    dict(id="T12", level="L2", name="touch EXCEPT larger", prims=["touch", "larger"], exc=["larger"],
         wording="except", twin="T11", fire=lambda q: q["touch"] is True and q["larger"] is True,
         fn=_simple(lambda q: EXCEPT(q["touch"], q["larger"])),
         text=["Select every marked object whose visible outline shares a border with the outline of {S}, except objects that cover more image pixels than {S}. The exception overrides the first condition.",
               "Which marked objects border {S} (their visible outline shares a border with the outline of {S})? Leave out any of them that cover more image pixels than {S}; this exception always wins.",
               "List the marked objects whose visible outline meets the outline of {S} in the image. Exception: do not list an object if it covers more image pixels than {S}, even though it borders {S}."]),
    dict(id="T13", level="L2", name="left EXCEPT same_cat", prims=["left", "same_cat"], exc=["same_cat"],
         wording="except", twin="T08", fire=lambda q: q["left"] is True and q["same_cat"] is True,
         fn=_simple(lambda q: EXCEPT(q["left"], q["same_cat"])),
         text=["Select every marked object whose centre lies to the left of the leftmost visible point of {S}, except objects that are the same kind of object as {S}. The exception overrides the first condition.",
               "Which marked objects are left of {S} (their centre lies to the left of the leftmost visible point of {S})? Leave out any of them that are the same kind of object as {S}; this exception always wins.",
               "List the marked objects whose centre is further left in the image than the leftmost visible point of {S}. Exception: do not list an object if it is the same kind of object as {S}, even though it is left of {S}."]),
    dict(id="T14", level="L2", name="touch EXCEPT above", prims=["touch", "above"], exc=["above"],
         wording="except", fire=lambda q: q["touch"] is True and q["above"] is True,
         fn=_simple(lambda q: EXCEPT(q["touch"], q["above"])),
         text=["Select every marked object whose visible outline shares a border with the outline of {S}, except objects whose centre lies above the topmost visible point of {S}. The exception overrides the first condition.",
               "Which marked objects border {S} (their visible outline shares a border with the outline of {S})? Leave out any of them whose centre lies above the topmost visible point of {S}; this exception always wins.",
               "List the marked objects whose visible outline meets the outline of {S} in the image. Exception: do not list an object if its centre is higher in the image than the topmost visible point of {S}, even though it borders {S}."]),

    dict(id="T15", level="L3", name="(touch AND NOT same_cat) EXCEPT border", prims=["touch", "same_cat", "border"],
         neg=["same_cat"], exc=["border"], wording="except",
         fire=lambda q: q["touch"] is True and q["same_cat"] is False and q["border"] is True,
         fn=_simple(lambda q: EXCEPT(AND(q["touch"], NOT(q["same_cat"])), q["border"])),
         text=["Select every marked object whose visible outline shares a border with the outline of {S} and that is not the same kind of object as {S}, except objects that reach the edge of the image. The exception overrides the other conditions.",
               "Which marked objects border {S} (their visible outline shares a border with the outline of {S}) and are not the same kind of object as {S}? Leave out any of them that reach the edge of the image; this exception always wins.",
               "List the marked objects whose visible outline meets the outline of {S} and that are not the same kind of object as {S}. Exception: do not list an object if it reaches the edge of the image, even though it meets the other conditions."]),
    dict(id="T16", level="L3", name="CLOSURE_touch(same_cat)", prims=["same_cat", "touches"], closure=True,
         wording="closure", fn=_closure,
         text=["Select every marked object that is the same kind of object as {S}, and also every marked object whose visible outline shares a border with the outline of one of those same-kind marked objects. Sharing a border with {S} itself does not count.",
               "First find the marked objects of the same kind as {S}. Then add every marked object that borders one of those marked objects, meaning its visible outline shares a border with that object's outline; bordering {S} itself does not count. Which objects are selected?",
               "List the marked objects of the same kind as {S}, together with any marked object whose visible outline meets the outline of one of those same-kind marked objects in the image. Meeting the outline of {S} itself does not count."]),
]
for _t in TEMPLATES:
    _t.setdefault("neg", [])
    _t.setdefault("exc", [])
    # the rule-specific probe (Lite probe-equivalence check) asks exactly the rule's primitives
    _t["probe"] = list(_t["prims"])
    comps = {"size" if p in ("larger", "smaller") else p for p in _t["prims"] if p != "touches"}
    if len(_t["prims"]) > 1 and not _t.get("closure"):
        comps.add("AND")
    if _t["neg"] or _t["exc"]:
        comps.add("NOT")             # EXCEPT is AND-NOT with different wording
    if _t["exc"]:
        comps.add("EXCEPT")
    if _t.get("closure"):
        comps.add("CLOSURE")
    _t["components"] = [c for c in COMPONENTS if c in comps]

# Diagnostic templates: outside the 16-template design (no Q-matrix row, no quotas, not in the probe).
# C01 pairs with T01 on the same anchor: same candidates, everyday "touching" vs. operational "borders".
DIAG_TEMPLATES = [
    dict(id="C01", level="L1", name="contact (caption)", prims=["contact"], wording="none", pair="T01",
         fn=_simple(lambda q: q["contact"]),
         text=["Select every marked object that is physically touching {S}: resting on it, leaning on it, or held, worn, carried or ridden by it, or the reverse.",
               "Which marked objects are physically touching {S}? Count objects resting on or leaning on {S}, objects {S} holds, wears, carries or rides, and the reverse.",
               "List the marked objects in physical contact with {S}, such as objects resting on it, leaning on it, or held, worn, carried or ridden by it, and the reverse."]),
]
for _t in DIAG_TEMPLATES:
    _t.update(neg=[], exc=[], probe=[], components=[], diagnostic=True)
TEMPLATE_BY_ID = {t["id"]: t for t in TEMPLATES + DIAG_TEMPLATES}
TWINS = sorted({tuple(sorted((t["id"], t["twin"]))) for t in TEMPLATES if t.get("twin")})


def check_template_set():
    """§4.1 build-time constraints. Returns a report; raises if one fails."""
    import numpy as np
    neg_fam = sorted({FAMILY[p] for t in TEMPLATES for p in t["neg"]})
    exc_fam = sorted({FAMILY[p] for t in TEMPLATES for p in t["exc"]})
    Q = np.array([[int(c in t["components"]) for c in COMPONENTS] for t in TEMPLATES])
    rank = int(np.linalg.matrix_rank(Q))
    and_only = [t["id"] for t in TEMPLATES if "AND" in t["components"] and "NOT" not in t["components"]]
    rep = dict(not_families=neg_fam, except_families=exc_fam, q_rank=rank, q_cols=len(COMPONENTS),
               q_matrix={t["id"]: t["components"] for t in TEMPLATES}, twins=TWINS, and_without_not=and_only)
    fails = []
    if len(neg_fam) < 2 or len(exc_fam) < 2:
        fails.append("NOT/EXCEPT must each cover >= 2 primitive families")
    if rank < len(COMPONENTS):
        fails.append(f"Q-matrix rank {rank} < {len(COMPONENTS)} components")
    if len(TWINS) < 2:
        fails.append("need >= 2 twin pairs")
    for a, b in TWINS:
        if TEMPLATE_BY_ID[a]["prims"] != TEMPLATE_BY_ID[b]["prims"]:
            fails.append(f"twins {a}/{b} use different primitives")
    rep["fails"] = fails
    if fails:
        raise AssertionError("; ".join(fails))
    return rep


def cell(t, q):
    """Truth-table cell of one candidate under template t: one T/F/U letter per primitive."""
    return "".join("U" if q[p] is None else "T" if q[p] else "F" for p in t["prims"] if p != "touches")


# ---------------------------------------------------------------- question rendering
MODALITIES = {
    "text": "M-text: the subject is named by its caption phrase and carries no mark",
    "contour": "M-contour: the subject is outlined in magenta and carries no number",
    "mark": "M-mark: the subject carries a number like every other object",
}
PROBE_PRIMS = ["same_cat", "larger", "smaller", "touch", "left", "above", "border"]
# JSON keys the model sees in the probe; parse answers back with {v: k for k, v in PROBE_KEY.items()}.
# "touch" and "border" are internal names only: the predicate means outlines meeting in the picture.
PROBE_KEY = dict(same_cat="same_kind", larger="larger", smaller="smaller", touch="borders", left="left",
                 above="above", border="image_edge", touches="borders_marked")
# a self-consistent example answer for the probe format (never all-true: larger and smaller exclude each other)
PROBE_EXAMPLE = dict(same_cat="false", larger="false", smaller="true", touch="true", left="false", above="true",
                     border="false", touches="[5]")


# Caption phrases may carry their own determiner ("a police officer", "another woman"); "the" replaces it.
_LEAD_DET = re.compile(r"^(?:a|an|the|another|one|this|that|his|her|their|its)\s+", re.I)


def subject_ref(modality, phrase=None, sid=None):
    if modality == "text":
        return "the " + _LEAD_DET.sub("", phrase.strip())
    if modality == "contour":
        return "the object outlined in magenta"
    return f"object #{sid}"


def subject_intro(modality, phrase=None, sid=None):
    if modality == "text":
        return f"The subject is {subject_ref(modality, phrase)}. It is not marked."
    if modality == "contour":
        return "The subject is the object outlined in magenta. It has no number."
    return f"The subject is object #{sid}."


def objects_line(marks=None):
    """Smoke test 2026-10-29: without the list, models skip visible marks (Qwen3-VL-8B left out >= 1
    candidate in 21% of probes), and a skipped candidate silently counts as "not selected" when the
    probe is executed. Listing the numbers gives the direct and probe prompts the same information."""
    line = "The other objects are outlined and marked with numbers."
    return line + (" The marked objects are " + ", ".join(f"#{m}" for m in sorted(marks)) + "." if marks else "")


# Visual audit (design §3.3): a quarter of touch items meet the subject only by occlusion, which
# readers take as "not touching". The definition is 2D on purpose; the note makes that explicit.
TOUCH_NOTE = ("Note: outlines that meet in the picture count as sharing a border, including where one object "
              "is in front of or behind the other; physical contact is not required.")


# Negation-phrasing ablation (design §5.1f): paraphrase 0 of each NOT template with the "not" clause
# restated positively. Same truth set and same U band; only the phrasing of the clause changes.
POSITIVE_TEXT = {
    "T08": "Select every marked object whose centre lies to the left of the leftmost visible point of {S} and that is a different kind of object from {S}.",
    "T09": "Select every marked object that covers more image pixels than {S} and whose visible outline is separated from the outline of {S} by a gap in the image.",
    "T10": "Select every marked object that reaches the edge of the image and whose centre lies at or to the right of the leftmost visible point of {S}.",
    "T11": "Select every marked object whose visible outline shares a border with the outline of {S} and that covers at most as many image pixels as {S}.",
    "T15": "Select every marked object whose visible outline shares a border with the outline of {S} and that is a different kind of object from {S}, except objects that reach the edge of the image. The exception overrides the other conditions.",
}


def render_direct(tid, para, modality, phrase=None, sid=None, text=None, marks=None):
    """Direct prompt. `text` replaces the template paraphrase (used by the POSITIVE_TEXT ablation)."""
    t = TEMPLATE_BY_ID[tid]
    S = subject_ref(modality, phrase, sid)
    rule = (text or t["text"][para]).replace("{S}", S)
    if "touch" in t["prims"] or t.get("closure"):
        rule += " " + TOUCH_NOTE
    return (f"{subject_intro(modality, phrase, sid)} {objects_line(marks)}\n"
            f"Rule: {rule}\n"
            "Answer with a JSON list of the numbers of all marked objects that satisfy the rule, e.g. [3, 7]. "
            "The subject itself is never part of the answer. If no marked object satisfies the rule, answer [].")


def _probe(prims, modality, phrase, sid, marks=None):
    S = subject_ref(modality, phrase, sid)
    lines = []
    for p in prims:
        if p == "touches":
            lines.append(f'"{PROBE_KEY[p]}": the list of numbers of the other marked objects whose visible outline shares a border with this object\'s outline')
        else:
            lines.append(f'"{PROBE_KEY[p]}": true if the object {PREDICATES[p].replace("{S}", S)}, otherwise false')
    keys = [f'"{PROBE_KEY[p]}": {PROBE_EXAMPLE[p]}' for p in prims]
    return (f"{subject_intro(modality, phrase, sid)} {objects_line(marks)}\n"
            "For every marked object, answer:\n" + "\n".join("- " + l for l in lines) + "\n"
            'Answer with one JSON object keyed by mark number, e.g. {"#3": {' + ", ".join(keys) + "}, ...}.")


def render_probe(tid, modality, phrase=None, sid=None, marks=None):
    """Rule-specific probe: only the rule's primitives (Lite probe-equivalence check, §4.3)."""
    return _probe(TEMPLATE_BY_ID[tid]["probe"], modality, phrase, sid, marks)


def render_probe_all(modality, phrase=None, sid=None, touches=False, marks=None):
    """The shared probe of §4.3: every primitive, once per (image, anchor, modality).

    `touches` adds the candidate-candidate list that CLOSURE needs; ask it only on anchors with a CLOSURE item.
    """
    return _probe(PROBE_PRIMS + (["touches"] if touches else []), modality, phrase, sid, marks)


def render_replica(prim, cand, modality, phrase=None, sid=None, marks=None):
    """Pairwise-benchmark replica (§4.3): one yes/no question about one candidate, CV-Bench style."""
    S = subject_ref(modality, phrase, sid)
    q = f"Is it true that object #{cand} {PREDICATES[prim].replace('{S}', S)}?"
    if prim == "touch":
        q += " " + TOUCH_NOTE
    return (f"{subject_intro(modality, phrase, sid)} {objects_line(marks)}\n"
            f"Question: {q}\n(A) yes\n(B) no\nAnswer with the letter A or B only.")


def evaluate(tid, preds, cands, cc):
    """Ground truth of template `tid`: (S*, U, keep, values) where keep applies the U_DROP rule."""
    vals = TEMPLATE_BY_ID[tid]["fn"](preds, cands, cc)
    S = [c for c in cands if vals[c] is True]
    U = [c for c in cands if vals[c] is None]
    return S, U, len(U) <= U_DROP * len(cands), vals
