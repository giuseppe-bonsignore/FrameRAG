# pip install rdflib rdflib-neo4j

from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from rdflib import Graph
from config import TTL_PATH
import os
from dotenv import load_dotenv

load_dotenv()

AURA_DB_URI=os.getenv("NEO4J_URI")
AURA_DB_USERNAME="neo4j"
AURA_DB_PWD=os.getenv("NEO4J_PASSWORD")

auth_data = {
    "uri": AURA_DB_URI,
    "database": "neo4j",
    "user": AURA_DB_USERNAME,
    "pwd": AURA_DB_PWD,
}

# --- Namespaces (FrameNet / PMO + common RDF vocabularies) ---
prefixes = {
    "dc":      "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "decomp":  "http://www.w3.org/ns/lemon/decomp#",
    "lime":    "http://www.w3.org/ns/lemon/lime#",
    "nif":     "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
    "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
    "owl":     "http://www.w3.org/2002/07/owl#",
    "pmo":     "http://premon.fbk.eu/ontology/core#",
    "pmoall":  "http://premon.fbk.eu/ontology/all#",
    "pmofn":   "http://premon.fbk.eu/ontology/fn#",
    "pmonb":   "http://premon.fbk.eu/ontology/nb#",
    "pmopb":   "http://premon.fbk.eu/ontology/pb#",
    "pmovn":   "http://premon.fbk.eu/ontology/vn#",
    "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":    "http://www.w3.org/2000/01/rdf-schema#",
    "skos":    "http://www.w3.org/2004/02/skos/core#",
    "synsem":  "http://www.w3.org/ns/lemon/synsem#",
    "vann":    "http://purl.org/vocab/vann/",
    "vartrans":"http://www.w3.org/ns/lemon/vartrans#",
    "void":    "http://rdfs.org/ns/void#",
    "xsd":     "http://www.w3.org/2001/XMLSchema#",

}

# --- Store config ---
config = Neo4jStoreConfig(
    auth_data=auth_data,
    custom_prefixes=prefixes,
    handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
    batching=True, 
)

# --- Parse & ingest ---
file_path = TTL_PATH
g = Graph(store=Neo4jStore(config=config))
g.parse(file_path, format="ttl")  # format="xml" for .rdf/.owl, "n3" for N3, etc.
g.close(True)  # important: flush buffered triples when batching=True

    