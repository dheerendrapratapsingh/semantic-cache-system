from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import Embedder
from vector_db.faiss_store import VectorStore
from clustering.fuzzy import FuzzyCluster
from cache.semantic_cache import SemanticCache
from data.dataset_loader import load_dataset


app = FastAPI()


embedder = Embedder()
cache = SemanticCache()


documents = load_dataset()

doc_embeddings = embedder.encode(documents)


vector_store = VectorStore(len(doc_embeddings[0]))

vector_store.add(doc_embeddings, documents)


cluster_model = FuzzyCluster()

cluster_model.fit(doc_embeddings)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(req: QueryRequest):

    query = req.query

    query_embedding = embedder.encode_query(query)

    hit, entry, similarity = cache.lookup(query_embedding)

    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(similarity),
            "result": entry["result"],
            "dominant_cluster": entry["cluster"]
        }

    results = vector_store.search(query_embedding)

    dominant_cluster = cluster_model.dominant_cluster(query_embedding)

    cache.add(query, query_embedding, results, dominant_cluster)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0,
        "result": results,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "cache cleared"}