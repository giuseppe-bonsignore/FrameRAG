# Graph-based Retrieval-Augmented Generation over FrameNet / PreMOn

Code used in the dissertation:

> **Enhancing Large Language Models with Graph-based Retrieval-Augmented Generation on Linguistic Linked Open Data Knowledge Bases**  

This repository contains the research prototype I used to build and compare:

1. A **baseline, document-based RAG** system over a **text corpus of FrameNet frames**.
2. A **GraphRAG system** over a **Neo4j property graph** built from the PreMOn FrameNet RDF data, with vector search and Cypher-based neighborhood expansion.

This package is still under review. Currently, the goal is just to make the implementation and experimental scripts available to anyone interested in reproducing or inspecting the methodology.

---

## High-level Overview

The code implements the two pipelines described in Chapter 2 of the thesis:

- **Baseline RAG (Section 2.6)**  
  - Build a text corpus of FrameNet frames as distributed through PreMOn.
  - Embed each frame description using `text-embedding-3-small`.
  - Build a **cosine similarity index** over these embeddings.
  - Retrieve the top-k frames for a user query and pass their descriptions to an LLM.

- **GraphRAG over Neo4j (Section 2.7)**  
  - Import the **PreMOn FrameNet module** into Neo4j as a **labelled property graph**.
  - Assign a common label `:MultiLabel` to relevant nodes (`:Frame`, `:FrameElement`, `:LexicalEntry`).
  - Populate a Neo4j **vector index** on `n.MultiLabel_embedding` using OpenAI embeddings.
  - Use `neo4j-graphrag`’s `VectorCypherRetriever` to:
    - perform vector search over the graph
    - expand **one-hop** and **two-hop** neighbourhoods via Cypher queries
    - turn these neighbourhoods into textual context for an LLM (e.g. `gpt-5`).

The scripts in this repo correspond roughly to the steps described in Sections 2.5–2.7 of the thesis.

---

## Repository Structure

- **Corpus & Baseline RAG**

  - `build_corpus.py`  
    Extracts frame-level information from PreMOn/FrameNet RDF and builds a **plain-text corpus**. Each document corresponds to one frame.

  - `frame_index_builder.py`  
    Embeds each document in the corpus using `text-embedding-3-small`, normalises the vectors, and writes:
    - `vectors.npy` — matrix `M ∈ ℝ^{N×D}` (L2-normalised)
    - `frame_ids.json` — list of frame identifiers aligned with the rows of `M`.

- **Graph Construction & Embeddings in Neo4j**

  - `rdf_to_neo4j.py`  
    Imports the PreMOn FrameNet RDF Turtle data into Neo4j using `rdflib` / `rdflib-neo4j`. 

  - `populate_multilabel_embeddings.py`  
    Fetches nodes labelled `:MultiLabel`, builds short textual descriptions, sends them to the OpenAI embedding endpoint, and writes the resulting vectors back to Neo4j as `n.MultiLabel_embedding`.

  - `config.py`  
    Central configuration: paths, model names, Neo4j database name, batch sizes, etc.

- **GraphRAG Retrieval & Demos**

  - `neo4j_retriever.py`  
    Core retrieval logic using `neo4j-graphrag`:
    - sets up a `VectorCypherRetriever` over the Neo4j vector index;
    - defines **one-hop** and **two-hop** Cypher retrieval queries (as in Sections 2.7.2–2.7.3);
    - builds the textual context used as input to the LLM.

  - `demo_query.py`  
    Simple CLI demo to:
    - embed a natural language query
    - perform vector search over the graph
    - expand the neighbourhood with Cypher
    - send the resulting context + query to an LLM
    - print the answer (e.g. queries like “Which frames are evoked by lexical entry `n-retailer`?”)

---

## Data & Prerequisites

This repo **does not** include:

- the full PreMOn / FrameNet RDF dumps,
- any Neo4j databases,
- any API keys or credentials.

To run the pipeline end-to-end you will need:

1. **PreMOn FrameNet RDF data**  
   - e.g. the FrameNet module of PreMOn (Turtle/RDF files).

2. **Neo4j**  
   - A running Neo4j instance (local or remote).
   - A database with write access for import and embedding.

3. **OpenAI API access**  
   - Used for `text-embedding-3-small` and (optionally) `gpt-5` or another LLM.

---

## Installation

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
```

## Environment Configuration

Create a .env file (see ./.env.example). 

## Workflow

1. **Baseline RAG**

   1. **Build the corpus**
      ```bash
      python build_corpus.py
      ```

   2. **Build the embedding index for similarity search**
      ```bash
      python frame_index_builder.py
      ```

2. **GraphRAG**

   1. **Convert the triples to a property graph in Neo4j**
      ```bash
      python rdf_to_neo4j.py
      ```

   2. **Populate the vector index in Neo4j**
      ```bash
      python populate_multilabel_embeddings.py
      ```

   3. **Test GraphRAG retrievers (optional)**
      ```bash
      python neo4j_retrievers.py
      ```

   4. **Run GraphRAG demo**
      ```bash
      python demo_query.py
      ```

## Mapping to the Dissertation

If you are reading this alongside the thesis draft:

- **Section 2.5 – Graph-based knowledge representation in Neo4j**  
  - Implemented mainly by `rdf_to_neo4j.py` and the labelling/organisation logic.  
  - Details of node labels and relationship types match the descriptions in Subsection 2.5.3.

- **Section 2.6 – Baseline RAG methodology**  
  - `build_corpus.py` → Section 2.6.1 (corpus construction from the PreMOn dataset).  
  - `frame_index_builder.py` → Sections 2.6.2–2.6.3 (embedding, index construction and query-time retrieval).

- **Section 2.7 – GraphRAG methodology**  
  - `rdf_to_neo4j.py` + `populate_multilabel_embeddings.py` → Section 2.7.1 (vector index and embeddings in Neo4j).  
  - `neo4j_retriever.py` → Section 2.7.2 (setting up `VectorCypherRetriever`, one-hop retrieval queries, organiser).  
  - Multi-hop retrieval logic (two-hop paths) → Section 2.7.3 (complex graph traversal).

If you are trying to reproduce a specific example from the thesis, consider looking at
`demo_query.py` and `neo4j_retriever.py`, or contact me for the exact command/query used.

---

## Notes for Reviewers

This is **research code**, not a library:

- Error handling is minimal.
- Interfaces may be rough.
- Some scripts overlap in functionality or reflect intermediate experiments.

Some artefacts are **intentionally excluded**:

- `.env` and any credentials (see .env.example)
- large RDF dumps
- full Neo4j database backups

If something is unclear, or you want to know **which script generated which result** in the thesis, I’m happy to point to the relevant code snippet or pipeline.

## Citation

If you use this code or ideas in your own work, please cite the dissertation and, where appropriate, the underlying resources, such as:

- FrameNet and PreMOn resources  
- Neo4j and related tools (`rdflib-neo4j`, `neo4j-graphrag`)  
- OpenAI models (`text-embedding-3-small`, `gpt-4o-mini`, etc.)

Example citation:

> Bonsignore, Giuseppe. *Enhancing Large Language Models with Graph-based Retrieval-Augmented Generation on Linguistic Linked Open Data Knowledge Bases*. MSc Thesis, Università Cattolica del Sacro Cuore, 2025.


