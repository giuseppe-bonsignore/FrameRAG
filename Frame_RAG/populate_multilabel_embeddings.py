#!/usr/bin/env python
# populate_embeddings.py
import os, time
from typing import List, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

# --- config ---
load_dotenv(".env", override=True)
NEO4J_URI       = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME  = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE  = os.getenv("NEO4J_DATABASE", os.getenv("NEO4J_DB", "neo4j"))

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = os.getenv("EMBEDDER_MODEL", os.getenv("EMBED_MODEL", "text-embedding-3-small"))  # 1536-dim
READ_BATCH      = int(os.getenv("READ_BATCH", "500"))
EMBED_BATCH     = int(os.getenv("EMBED_BATCH", "256"))
INDEX_NAME      = "MultiLabel_embeddings"

EMBED_DIM = int(os.getenv("EMBED_DIMENSIONS", "1536"))

client  = OpenAI(api_key=OPENAI_API_KEY)
driver  = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def check_index(session):
    rec = session.run("""
      SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options
      WHERE name = $name AND type = 'VECTOR' AND entityType = 'NODE'
      RETURN labelsOrTypes AS labels, properties AS props,
             coalesce(options['indexConfig']['vector.dimensions'], options['vector.dimensions']) AS dim,
             coalesce(options['indexConfig']['vector.similarity_function'], options['vector.similarity_function']) AS sim
    """, name=INDEX_NAME).single()
    if not rec: raise RuntimeError(f"Vector index '{INDEX_NAME}' not found.")
    if "MultiLabel" not in rec["labels"]: raise RuntimeError("Index must target :MultiLabel")
    if "MultiLabel_embedding" not in rec["props"]: raise RuntimeError("Index must be on property 'MultiLabel_embedding'")
    if int(rec["dim"]) != EMBED_DIM: raise RuntimeError(f"Index dim {rec['dim']} != {EMBED_DIM}")
    print(f"OK index {INDEX_NAME}: dim={EMBED_DIM} sim={rec['sim']} labels={rec['labels']}")

def count_frames(session) -> int:
    return session.run("MATCH (n:MultiLabel) RETURN count(n) AS c").single()["c"]

# def fetch_frames(session, skip: int, limit: int):
#     q = """
#     MATCH (n:MultiLabel)
#     RETURN n.uri AS uri, coalesce(n.label,'') AS label, coalesce(n.definition,'') AS definition
#     ORDER BY n.uri SKIP $skip LIMIT $limit
#     """
#     return list(session.run(q, skip=skip, limit=limit))

def fetch_frames(session, skip: int, limit: int):
    q = """
    MATCH (n:MultiLabel)
    RETURN 
      n.uri AS uri, 
      labels(n) AS labels,
      coalesce(n.label,'') AS label, 
      coalesce(n.definition,'') AS definition
    ORDER BY n.uri SKIP $skip LIMIT $limit
    """
    return list(session.run(q, skip=skip, limit=limit))


def embed_batch(texts: List[str]) -> List[List[float]]:
    tries = 0
    while True:
        try:
            resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
            return [d.embedding for d in resp.data]
        except (RateLimitError, APITimeoutError, APIError):
            tries += 1
            if tries > 5: raise
            time.sleep(min(2**tries, 30))

def set_embeddings(session, rows: List[Tuple[str, List[float]]]):
    q = """
    UNWIND $rows AS row
    MATCH (n:MultiLabel {uri: row.uri})
    SET n.MultiLabel_embedding = row.MultiLabel_embedding
    """
    session.run(q, rows=[{"uri": u, "MultiLabel_embedding": vec} for u, vec in rows])

# def main():
#     with driver.session(database=NEO4J_DATABASE) as s:
#         check_index(s)
#         total = count_frames(s)
#         print(f"{total} nodes to (re)embed")

#         processed = 0
#         while processed < total:
#             rows = fetch_frames(s, processed, READ_BATCH)
#             if not rows: break

#             uris  = [r["uri"] for r in rows]
#             texts = [f"{r['label']} — {r['definition']}".strip() for r in rows]

#             vectors: List[List[float]] = []
#             for i in range(0, len(texts), EMBED_BATCH):
#                 vectors.extend(embed_batch(texts[i:i+EMBED_BATCH]))

#             assert len(vectors) == len(uris)
#             set_embeddings(s, list(zip(uris, vectors)))

#             processed += len(rows)
#             print(f"upserted {processed}/{total}")

def main():
    with driver.session(database=NEO4J_DATABASE) as s:
        check_index(s)
        total = count_frames(s)
        print(f"{total} nodes to (re)embed")

        processed = 0
        while processed < total:
            rows = fetch_frames(s, processed, READ_BATCH)
            if not rows:
                break

            uris = []
            texts = []

        for r in rows:
            uri  = r["uri"]
            labs = r["labels"] or []

            # 1) Primary "kind" (for quick type bias)
            if "Frame" in labs:
                kind = "Frame"
            elif "FrameElement" in labs:
                kind = "FrameElement"
            elif "LexicalEntry" in labs:
                kind = "LexicalEntry"
            else:
                kind = "Other"

            # 2) Preserve *all* labels in a separate field
            labels_str = ", ".join(labs)

            text = (
                f"TYPE: {kind}; "
                f"LABELS: {labels_str}; "
                f"URI: {uri}; "
                f"NAME: {r['label']}; "
                f"DEFINITION: {r['definition']}"
            ).strip()

            uris.append(uri)
            texts.append(text)

            vectors: List[List[float]] = []
            for i in range(0, len(texts), EMBED_BATCH):
                vectors.extend(embed_batch(texts[i:i + EMBED_BATCH]))

            assert len(vectors) == len(uris)
            set_embeddings(s, list(zip(uris, vectors)))

            processed += len(rows)
            print(f"upserted {processed}/{total}")

    driver.close()
    print("Done.")


if __name__ == "__main__":
    main()
