# demo_query.py
import sys
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from frame_index_builder import load_frame_index, read_frames_corpus
import neo4j_retriever
from neo4j_retriever import close_driver

load_dotenv(find_dotenv(usecwd=True), override=True)

OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

TOP_K = int(os.getenv("TOP_K", "5"))

def vanilla_llm(query: str) -> str:
    prompt = (
        "Answer the following question about FrameNet (without any external context). "
        "If you're unsure, say so briefly.\n\n"
        f"{query}"
    )
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Answer concisely and factually. Do not speculate."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return r.choices[0].message.content

def baseline_rag(query: str, k: int = TOP_K) -> str:
    index, model, frame_ids = load_frame_index()
    ids, texts = read_frames_corpus()

    qv = model.encode([query])
    D, I = index.search(np.array(qv, dtype=np.float32), k)
    context = "\n\n".join([texts[i] for i in I[0]])

    prompt = (
        "You are answering questions about FrameNet. Use ONLY the provided context. "
        "If the answer is not present, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return r.choices[0].message.content

def neo4j_vector_cypher_rag(query:str, k:int=TOP_K) -> str: 
    vector_cypher_retriever = neo4j_retriever.vector_cypher_retriever
    context = vector_cypher_retriever.search(query_text=query)
    prompt = (
        "Answer the Question using the following Context. "
        "Only respond with information mentioned in the Context. "
        "Do not inject any speculative information not mentioned.\n\n"
        f"# Question:\n{query}\n\n# Context:\n{context}\n\n# Answer:\n"
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    close_driver()
    return response.choices[0].message.content

def neo4j_multilabel_vector_cypher_rag(query:str, k:int=TOP_K) -> str: 
    vector_cypher_retriever = neo4j_retriever.vector_cypher_retriever
    context = vector_cypher_retriever.search(query_text=query)
    prompt = (
        "Answer the Question using the following Context. "
        "Only respond with information mentioned in the Context. "
        "Do not inject any speculative information not mentioned.\n\n"
        f"# Question:\n{query}\n\n# Context:\n{context}\n\n# Answer:\n"
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    close_driver()
    return response.choices[0].message.content


def neo4j_graphRAG(query:str, k:int=TOP_K) -> str: 
    vector_cypher_retriever = neo4j_retriever.rag
    res = vector_cypher_retriever.search(query_text=query, retriever_config={"top_k":k})
    close_driver()
    return res.answer

    
if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Which frames relate to giving and receiving?"
    print(f"\n=== QUERY ===\n{q}\n")
    print("=== 1) Vanilla LLM ===\n")
    print(vanilla_llm(q))
    print("\n" + "="*60 + "\n")

    print("=== 2) Baseline RAG ===\n")
    print(baseline_rag(q))
    print("\n" + "="*60 + "\n")

    # print("=== 3) Neo4j Vector Cypher Retriever (ANN + graph expansion) ===\n")
    # print(neo4j_vector_cypher_rag(q))
    # print("\n" + "="*60 + "\n")

    # print("=== 3) Neo4j MultiLabel Vector Cypher Retriever (ANN + graph expansion) ===\n")
    # print(neo4j_multilabel_vector_cypher_rag(q))

    print("=== 3) GraphRAG (ANN + graph expansion) ===\n")
    print(neo4j_graphRAG(q, 3))

