"""One tag taxonomy for the golden set.

Every tag is `dimension:value` — exactly one colon, lowercase, hyphens inside values. Before this
the namespace was flat: `squad` (a corpus), `freshness` (a risk axis), `attack:jailbreak` (an
injection sub-dimension) and `type:bridge` (a HotpotQA field) all sat at the same level, and 21
tags carried no dimension at all, so a report could not say what a tag described.

DIMENSIONS
  origin:      real | synthetic                       where the row came from
  source:      squad, pubmedqa, casehold, ...         which corpus
  subcorpus:   popqa, triviaqa, realtimeqa, freshqa   sub-dataset inside a source
  risk:        the axis the row exercises             single-hop, multi-hop, numerical,
                                                      freshness, ambiguity, conflict,
                                                      high-stakes-medical|legal|financial,
                                                      citation-need, retrieval-need,
                                                      real-contradiction, table-context
  transform:   synthetic family                       injection, context-removal, ...
  attack: carrier: placement: trust: evasiveness: payload:      injection sub-dimensions
  pressure: citation-style: ambiguity-style: tool-arm: transform-arm: donor:   family sub-dims
  meta:        corpus-specific descriptors            level-hard, hops-3, type-bridge, ...
  lang:        en | zh
  intensity:   low | medium | high
  need:        web | retrieval | calculator
  judge:       unanimous | boundary
"""

# bare tags -> dimensioned
BARE = {
    # corpora
    "squad": "source:squad", "hotpot_qa": "source:hotpot-qa", "musique": "source:musique",
    "pubmedqa": "source:pubmedqa", "drop": "source:drop", "finqa": "source:finqa",
    "casehold": "source:casehold", "streamingqa": "source:streamingqa", "asqa": "source:asqa",
    "retrievalqa": "source:retrievalqa", "crrag_conflict": "source:crrag-conflict",
    # risk axes
    "single-hop": "risk:single-hop", "multi-hop": "risk:multi-hop", "numerical": "risk:numerical",
    "freshness": "risk:freshness", "ambiguity": "risk:ambiguity", "conflict": "risk:conflict",
    "citation-bearing": "risk:citation-need", "retrieval-need": "risk:retrieval-need",
    "real-contradiction": "risk:real-contradiction", "table-context": "risk:table-context",
}

# prefix renames: old dimension -> new dimension
PREFIX = {
    "synthetic": "transform",
    "high-stakes": "risk",
    "judged": "judge",
    "origin": "subcorpus",          # origin:popqa was a sub-dataset, not real/synthetic
    "citation": "citation-style",
    "ambiguity_style": "ambiguity-style",
    "tool_arm": "tool-arm",
    "transform_arm": "transform-arm",
    "donor_source": "donor",
    "level": "meta", "hops": "meta", "type": "meta",
    "interpretations": "meta", "adapted": "meta",
}

# these old dimensions fold into meta: as `meta:<olddim>-<value>`
FOLD_INTO_META = {"level", "hops", "type", "interpretations", "adapted"}

# values that need reshaping under the new dimension
VALUE = {
    ("risk", "medical"): "high-stakes-medical",
    ("risk", "legal"): "high-stakes-legal",
    ("risk", "financial"): "high-stakes-financial",
}


def normalize(tag: str) -> str:
    """Map one legacy tag onto the schema. Idempotent."""
    if tag in BARE:
        return BARE[tag]
    if ":" not in tag:
        return f"meta:{tag}"
    head, _, rest = tag.partition(":")
    if head in FOLD_INTO_META:
        return f"meta:{head}-{rest}".replace("_", "-")
    dim = PREFIX.get(head, head)
    value = VALUE.get((dim, rest), rest)
    # payload:real:deepset had two colons
    value = value.replace(":", "-").replace("_", "-")
    return f"{dim.replace('_', '-')}:{value}"


ORIGIN_VALUES = {"real", "synthetic"}


def normalize_all(tags):
    """Normalize every tag, stamp origin:, and de-duplicate while preserving order."""
    out, seen = [], set()
    for tag in tags:
        # origin:real / origin:synthetic are ours; legacy origin:<dataset> is a sub-corpus
        n = tag if tag.partition(":")[2] in ORIGIN_VALUES and tag.startswith("origin:") else normalize(tag)
        if n not in seen:
            seen.add(n)
            out.append(n)
    has_transform = any(t.startswith("transform:") for t in out)
    origin = "origin:synthetic" if has_transform else "origin:real"
    if origin not in seen:
        out.append(origin)
    return out
