import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, HybridCypherRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

load_dotenv()

URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASS = os.getenv("NEO4J_PASSWORD")
VECTOR_INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX")
MULTILABEL_INDEX_NAME = os.getenv("NEO4J_MULTILABEL_VECTOR_INDEX")
EMBED_MODEL = os.getenv("EMBED_MODEL") 

driver = GraphDatabase.driver(uri=URI, auth=("neo4j", PASS))
embedder = OpenAIEmbeddings(model=EMBED_MODEL, api_key=os.getenv("OPENAI_API_KEY"))

generic_one_hop_retrieval_query = """
// 'node' and 'score' come from the vector index search
WITH node, score, labels(node) AS labs

// one-hop across ALL relationship types
OPTIONAL MATCH (node)-[r]-(nbr)
WITH node, score, labs,
     collect(DISTINCT {
       rel_type: type(r),
       dir: CASE WHEN elementId(startNode(r)) = elementId(node) THEN 'out' ELSE 'in' END,
       nbr_labels: labels(nbr),
       nbr_label: coalesce(nbr.label, ''),
       nbr_uri:   coalesce(nbr.uri,   ''),
       nbr_def:   coalesce(nbr.definition, '')
     })[0..100] AS rels   // safety cap

// build a single textual context blob
WITH
  '=== seed ===\\n' +
  'labels: ' + reduce(s = '', l IN labs |
                       s + CASE WHEN s = '' THEN l ELSE ', ' + l END) + '\\n' +
  'label: ' + coalesce(node.label, '') + '\\n' +
  'uri: '   + coalesce(node.uri,   '') + '\\n' +
  CASE
    WHEN node.definition IS NOT NULL AND node.definition <> ''
    THEN 'definition: ' + node.definition + '\\n'
    ELSE ''
  END +
  '\\nsimilarity_score: ' + toString(score) + '\\n' +
  '\\n=== one-hop neighborhood ===\\n' +
  reduce(s = '', x IN rels |
    s +
    '- [' + x.dir + ' ' + x.rel_type + '] ' +
      coalesce(x.nbr_label, '') +
      CASE WHEN x.nbr_uri <> '' THEN ' (uri: ' + x.nbr_uri + ')' ELSE '' END +
      CASE
        WHEN x.nbr_def <> '' THEN '\\n    definition: ' + x.nbr_def
        ELSE ''
      END + '\\n'
  ) AS text

RETURN text
"""

multi_hop_query = """
// 1) Two-hop expansion from the seed node
WITH node, score
MATCH (node)-[r1]-(entity1)-[r2]-(entity2)

// 2) Collect both one-hop and two-hop entities with their relations
WITH node, score,
     collect(DISTINCT entity1) AS entities1,
     collect(DISTINCT entity2) AS entities2,
     collect(DISTINCT r1) + collect(DISTINCT r2) AS rels

// 3) Return the retrieved context as a single text block
RETURN apoc.text.join(
  [r IN rels |
    coalesce(startNode(r).label, '') +
    ' - ' + type(r) +
    ' -> ' + coalesce(endNode(r).label, '')
  ],
  '\n'
) AS info
"""


vector_cypher_retriever = VectorCypherRetriever(
    driver,
    index_name=MULTILABEL_INDEX_NAME,
    embedder=embedder,
    retrieval_query=generic_one_hop_retrieval_query
)

multi_hop_retriever = VectorCypherRetriever(
    driver,
    index_name=MULTILABEL_INDEX_NAME,
    embedder=embedder,
    retrieval_query=multi_hop_query
)

def close_driver(): 
    driver.close()

llm = OpenAILLM(model_name="gpt-5")
rag = GraphRAG(retriever=vector_cypher_retriever, llm=llm)
multi_hop_rag = GraphRAG(retriever=multi_hop_retriever, llm=llm)

if __name__ == "__main__":

    q = sys.argv[1] if len(sys.argv) > 1 else "Which frames relate to giving and receiving?"

    print("\n\n****** VECTOR CYPHER RETRIEVER RESPONSE: ******\n")
    response = rag.search(query_text=q, retriever_config={"top_k": 3})
    print(response.answer)
    close_driver()

    print("\n****** MULTI-HOP VECTOR CYPHER RETRIEVER RESPONSE: ******\n")
    response = multi_hop_rag.search(query_text=q, retriever_config={"top_k": 3})
    print(response.answer)
    close_driver()
