from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


from dotenv import load_dotenv  # type: ignore
load_dotenv(override=True)

from rdflib import Graph, URIRef, Literal

# =====================
# Config & CLI defaults
# =====================

ROOT_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

TTL_PATH = Path(os.getenv("TTL_PATH", DATA_DIR / "premon-2018a-fn15-inf.ttl"))
FRAME_CORPUS = Path(os.getenv("FRAME_CORPUS", DATA_DIR / "frame_corpus.txt"))

# Optional language filter prefix (e.g., "en"). If None, accept any.
LANG_FILTER = os.getenv("LANG_FILTER")  # e.g. "en"

# =====================
# Namespaces (as strings)
# =====================

CORE    = "http://premon.fbk.eu/ontology/core#" 
FN      = "http://premon.fbk.eu/ontology/fn#"
ONTOLEX = "http://www.w3.org/ns/lemon/ontolex#"
RDFS    = "http://www.w3.org/2000/01/rdf-schema#"
SKOS    = "http://www.w3.org/2004/02/skos/core#"
RDF     = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
DCT     = "http://purl.org/dc/terms/"
PMOFN   = "http://premon.fbk.eu/ontology/fn#"
PMO     = "http://premon.fbk.eu/ontology/core#"

# Optimize repeated look-ups
P_RDFS_LABEL        = URIRef(RDFS + "label") 
P_RDFS_COMMENT      = URIRef(RDFS + "comment")
P_SKOS_DEFINITION   = URIRef(SKOS + "definition")  # keep name readable
P_DCT_CREATED       = URIRef(DCT + "created")
P_DCT_IDENTIFIER    = URIRef(DCT + "identifier")
P_DCT_CREATOR       = URIRef(DCT + "creator")
P_RDF_TYPE          = URIRef(RDF + "type")
P_CORE_SEMROLE      = URIRef(CORE + "semRole")
P_PMO_SEMROLE       = URIRef(PMO + "semRole")
P_ONTOLEX_EVOKEDBY  = URIRef(ONTOLEX + "isEvokedBy")

# Outgoing relation predicates of interest
REL_PREDS = [
    PMOFN + "frameRelation",
    PMOFN + "inheritsFrom",
    PMOFN + "subframeOf",
    PMOFN + "semType",
    PMO   + "classRel",
    PMO   + "semRole",
    SKOS  + "broader",
    FN    + "inheritsFrom",
    FN    + "uses",
    FN    + "subFrameOf",
]
REL_PREDS_URIS = [URIRef(p) for p in REL_PREDS]

# Inverse relation predicates to count
INV_PREDS = [
    PMO   + "classRel",
    PMOFN + "subframeOf",
    SKOS  + "broader",
    PMO   + "item",
    PMOFN + "inheritsFrom",
    PMOFN + "frameRelation",
]
INV_PREDS_URIS = [URIRef(p) for p in INV_PREDS]

# ======================================
# RDF helpers (merged from fn_utils/text)
# ======================================

def load_framenet(path: str) -> Graph:
    g = Graph()
    g.parse(path, format="turtle")
    return g

def _lit_ok(lit: Literal) -> bool:
    if not isinstance(lit, Literal):
        return False
    lang = lit.language
    return (LANG_FILTER is None) or (lang is None) or (str(lang).lower().startswith(LANG_FILTER.lower()))

def _short(uri: str) -> str:
    """Compact a URI for readability."""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]

def list_frames_graph(g: Graph, limit: Optional[int] = None) -> List[Dict]: # small query to know what and how many frames are there 
    q = f"""
    PREFIX fn: <{FN}>
    PREFIX rdfs: <{RDFS}>
    SELECT DISTINCT ?f ?label
    WHERE {{
      ?f a fn:Frame .
      OPTIONAL {{ ?f rdfs:label ?label }}
    }}
    """
    if limit:
        q += f" LIMIT {limit}"

    results = []
    for row in g.query(q):
        uri = str(row.f)
        label = str(row.label) if getattr(row, "label", None) else _short(uri) #handle a few cases where label is missing 
        results.append({"uri": uri, "label": label})
    return results

def frame_info_graph(g: Graph, frame_uri: str) -> Dict:
    f = URIRef(frame_uri) # Turn the frame’s URI string into an rdflib URI node so it can be used in graph queries.
    info: Dict[str, object] = {"frame": _short(frame_uri), "uri": frame_uri} # start the output dictionary. 

    # Labels
    labels = [o for o in g.objects(f, P_RDFS_LABEL) if isinstance(o, Literal)] # Ask the graph for all objects of the triple (f, rdfs:label, ?o), and keep only those that are RDF.
    labels_filtered = [str(l) for l in labels if _lit_ok(l)]
    if labels_filtered:
        info["label"] = max(set(labels_filtered), key=len)
    elif labels:
        info["label"] = str(labels[0])

    defs = [str(o) for o in g.objects(f, P_SKOS_DEFINITION) if isinstance(o, Literal) and _lit_ok(o)]
    if defs:
        info["definition"] = max(set(defs), key=len)

    # Core literals
    created = next((o for o in g.objects(f, P_DCT_CREATED) if isinstance(o, Literal)), None)
    if created:
        info["created"] = str(created)
    identifier = next((o for o in g.objects(f, P_DCT_IDENTIFIER) if isinstance(o, Literal)), None)
    if identifier:
        info["identifier"] = str(identifier)
    creator = next((o for o in g.objects(f, P_DCT_CREATOR)), None)
    if creator:
        info["creator"] = _short(str(creator)) if not isinstance(creator, Literal) else str(creator)

    # Types
    types = {_short(str(o)) for o in g.objects(f, P_RDF_TYPE)}
    if types:
        info["types"] = sorted(types)

    # Frame Elements (labels)
    fe_labels = set()
    for fe in g.objects(f, P_CORE_SEMROLE):
        for lab in g.objects(fe, P_RDFS_LABEL):
            if isinstance(lab, Literal) and _lit_ok(lab):
                fe_labels.add(str(lab))
    for fe in g.objects(f, P_PMO_SEMROLE):
        for lab in g.objects(fe, P_RDFS_LABEL):
            if isinstance(lab, Literal) and _lit_ok(lab):
                fe_labels.add(str(lab))
    if fe_labels:
        info["elements"] = sorted(fe_labels)

    # Lexical Units
    lus = {_short(str(o)) for o in g.objects(f, P_ONTOLEX_EVOKEDBY)}
    if lus:
        info["lexical_units"] = sorted(lus)

    # Outgoing relations of interest
    relations: List[Tuple[str, str]] = []
    for p in REL_PREDS_URIS:
        for o in g.objects(f, p):
            if isinstance(o, Literal):
                continue
            relations.append((_short(str(p)), _short(str(o))))
    if relations:
        info["relations"] = sorted(set(relations))

    # Inverse relation counts
    inverse_counts: List[Tuple[str, int]] = []
    for p in INV_PREDS_URIS:
        cnt = sum(1 for _ in g.subjects(p, f))
        if cnt:
            inverse_counts.append((_short(str(p)), cnt))
    if inverse_counts:
        info["inverse_relations"] = sorted(inverse_counts, key=lambda x: x[0])

    return info

# =========================
# Textualization (simplify)
# =========================

def _prefixed(rel_local: str) -> str:
    if rel_local in {
        "frameRelation", "inheritsFrom", "subframeOf", "semType",
        "perspectiveOn", "uses", "reFrameMapping", "feCoreSet"
    }:
        return f"pmofn:{rel_local}"
    if rel_local in {"classRel", "semRole", "item"}:
        return f"pmo:{rel_local}"
    if rel_local == "broader":
        return "skos:broader"
    if rel_local in {"subFrameOf"}:
        return f"fn:{rel_local}"
    return rel_local

def _group_relations(relations: List[Tuple[str, str]], hide_semrole: bool = True) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for rel_local, tgt in relations:
        if hide_semrole and rel_local == "semRole":
            continue
        key = _prefixed(rel_local)
        grouped.setdefault(key, []).append(tgt)
    for k in list(grouped.keys()):
        grouped[k] = sorted(set(grouped[k]))
    return grouped

def _append_if(lines: List[str], key: str, value) -> None:
    if value is None:
        return
    if isinstance(value, list):
        if not value:
            return
        lines.append(f"{key}: " + ", ".join(value))
    else:
        lines.append(f"{key}: {value}")

def frame_to_text(info: Dict) -> str: # dict to txt 
    lines: List[str] = []

    title = info.get("label") or info.get("frame", "Unknown Frame")
    lines.append(title)

    if info.get("uri"):
        lines.append(f"uri: {info['uri']}")

    if info.get("types"):
        lines.append("Entity of type: " + ", ".join(info["types"]))

    if info.get("definition"):
        lines.append("skos:definition: " + info["definition"])

    if info.get("created"):
        lines.append(f"dcterms:created: {info['created']}")

    if info.get("label"):
        lines.append(f"rdfs:label: {info['label']}")

    if info.get("identifier"):
        lines.append(f"dcterms:identifier: {info['identifier']}")

    if info.get("lexical_units"):
        lines.append("ontolex:isEvokedBy: " + ", ".join(info["lexical_units"]))

    grouped_rels = _group_relations(info.get("relations", []), hide_semrole=True) # 

    for key in [
        "pmofn:frameRelation",
        "pmofn:inheritsFrom",
        "pmofn:semType",
        "pmofn:subframeOf",
        "pmofn:perspectiveOn",
        "pmofn:uses",
        "pmofn:reFrameMapping",
        "pmofn:feCoreSet",
    ]:
        if key in grouped_rels:
            _append_if(lines, key, grouped_rels[key])

    if info.get("creator"):
        lines.append(f"dcterms:creator: {info['creator']}")

    if "pmo:classRel" in grouped_rels:
        _append_if(lines, "pmo:classRel", grouped_rels["pmo:classRel"])

    if info.get("elements"):
        lines.append("pmo:semRole: " + ", ".join(info["elements"]))

    if "skos:broader" in grouped_rels:
        _append_if(lines, "skos:broader", grouped_rels["skos:broader"])

    if info.get("inverse_relations"):
        lines.append("")
        lines.append("INVERSE RELATIONS")
        for rel_local, n in sorted(info["inverse_relations"], key=lambda x: x[0]):
            pred = _prefixed(rel_local)
            res_word = "resources" if int(n) != 1 else "resource"
            lines.append(f"is {pred} of {int(n)} {res_word}")

    return "\n".join(lines)

def frames_to_texts(infos: List[Dict]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for info in infos:
        if "frame" in info:
            out.append((info["frame"], frame_to_text(info))) # [(fn15-give, ...), (fn15-tranfer,...)]
    return out

# =========================
# Build / write the corpus
# =========================

def build_frame_corpus(ttl_path: Path, out_path: Path, lang_filter: Optional[str] = LANG_FILTER, verbose: bool = False) -> int:

    if verbose: print("[corpus] Parsing TTL…", flush=True)
    g: Graph = load_framenet(str(ttl_path))
    if verbose: print("[corpus] Parsed. Collecting frames…", flush=True)

    frames = list_frames_graph(g, limit=None) # create a list of dictionaries -> [{uri:http..., label:transfer}, {...}]
    if verbose: print(f"[corpus] Found {len(frames)} frames. Extracting info…", flush=True)

    infos: List[Dict] = [] # gather details for each frame
    for i, f in enumerate(frames, 1):
        infos.append(frame_info_graph(g, f["uri"])) # Iterates over the frames list (from list_frames_graph), counting from 1 (not 0). Each f is a small dict like {"uri": "...", "label": "..."}.
        if verbose and i % 500 == 0: 
            print(f"[corpus] Processed {i}/{len(frames)} frames…", flush=True) # infos becomes a list of detailed per-frame records, ready to be turned into text with frames_to_texts(...) and written to the corpus file.

    if verbose: print("[corpus] Textualizing…", flush=True) # 
    docs = frames_to_texts(infos) # textualizing the list of dictionaries 

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose: print(f"[corpus] Writing {len(docs)} docs to {out_path}…", flush=True)
    with open(out_path, "w", encoding="utf-8") as out: # write content to output file 
        for frame_id, text in docs:
            out.write(f"### {frame_id}\n{text}\n\n") # -> output: corpus.txt

    if verbose: print("[corpus] Done.", flush=True)
    return len(docs)

# ========
# __main__
# ========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FrameNet frame corpus from Premon TTL.")
    parser.add_argument("--ttl", type=Path, default=TTL_PATH, help="Path to Premon TTL file.")
    parser.add_argument("--out", type=Path, default=FRAME_CORPUS, help="Path to output text corpus.")
    parser.add_argument("--lang", type=str, default=LANG_FILTER, help="Language prefix filter (e.g., 'en'). Omit to keep all.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs.")
    args = parser.parse_args()

    n = build_frame_corpus(ttl_path=args.ttl, out_path=args.out, lang_filter=args.lang, verbose=args.verbose)
    print(f"Saved {n} frames to {args.out}")
