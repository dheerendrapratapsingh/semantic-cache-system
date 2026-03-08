from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    """ 
    Custom semantic cache that stores previous queries and their embeddings.

    Instead of exact string matching, this cache compares query embeddings
    using cosine similarity to detect semantically similar queries.

    If similarity exceeds a threshold, the cached result is reused.
    """

    def __init__(self, threshold=0.85):
        # Storage for cached entries
        self.cache = []
        # Tunable similarity threshold
        # Higher value = stricter matching
        # Lower value = more cache hits but less precision
        self.threshold = threshold
         # For Showing Statistics
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):
        """
        Check if a semantically similar query already exists in cache.
        """
        for entry in self.cache:

            similarity = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            if similarity >= self.threshold:

                self.hit_count += 1

                return True, entry, similarity

        self.miss_count += 1

        return False, None, None
    # For adding a new query
    def add(self, query, embedding, result, cluster):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total = len(self.cache)

        hit_rate = self.hit_count / (self.hit_count + self.miss_count + 1e-5)

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }
    # For Reseting the cache
    def clear(self):

        self.cache = []

        self.hit_count = 0
        self.miss_count = 0