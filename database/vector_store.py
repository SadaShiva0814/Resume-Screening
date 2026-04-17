import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """
    FAISS-powered Vector Database Wrapper.
    Replaces brute-force NumPy sorting with structural nearest-neighbor trees.
    """
    def __init__(self, vector_dim=384):
        self.vector_dim = vector_dim

    def build_index(self, embeddings_matrix, ids=None):
        """
        Builds a FAISS index from a batch of embeddings.
        
        Args:
            embeddings_matrix: numpy array of shape (N, 384)
            ids: Optional list of document IDs mapping to the matrix rows.
                 If None, the index implicitly matches array indices.
        
        Returns:
            index: A FAISS IndexFlatIP.
            id_mapping: (dict) Mapping of index position to your document ID.
        """
        if len(embeddings_matrix) == 0:
            return None, {}
        
        # We use IndexFlatIP (Inner Product) because our embeddings are L2 normalized,
        # which means inner product is mathematically identical to cosine similarity
        # but evaluates much faster natively in C++.
        index = faiss.IndexFlatIP(self.vector_dim)
        
        # Ensure embeddings are normalized and float32 (FAISS strict requirement)
        embeddings_matrix = np.array(embeddings_matrix, dtype=np.float32)
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        # Prevent division by zero
        norms[norms == 0] = 1e-10
        embeddings_matrix = embeddings_matrix / norms
        
        index.add(embeddings_matrix)
        
        id_mapping = {i: _id for i, _id in enumerate(ids)} if ids else {}
        
        logger.info(f"Built FAISS Index with {index.ntotal} vectors.")
        
        return index, id_mapping

    def search(self, index, query_vector, k=10):
        """
        Search the FAISS index for the top-k most similar vectors.
        
        Args:
            index: Constructed FAISS Index
            query_vector: A single embedding pattern (shape: (384,))
            k: Top-K results to return
            
        Returns:
            distances: numpy array of similarity scores (cosine similarity bounds 0-1)
            indices: numpy array of internal position matches
        """
        if index is None or index.ntotal == 0:
            return np.array([]), np.array([])
        
        k = min(k, index.ntotal)
        
        # Prepare query vector
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
            
        distances, indices = index.search(query_vector, k)
        
        # For a single query, return flattened 1D arrays
        return distances[0], indices[0]

    def build_global_index(self):
        """
        Pulls all cached candidate embeddings from the entire database history
        and statically compiles them into memory for instant semantic search.
        """
        from database.db import get_db
        db = get_db()
        
        # Pull all cached resumes from history
        cursor = db.resume_cache.find({}, {"_id": 1, "full_embedding": 1})
        docs = list(cursor)
        
        if not docs:
            logger.info("No cached resumes found to index.")
            return None, {}
            
        embeddings = []
        ids = []
        for doc in docs:
            if 'full_embedding' in doc and len(doc['full_embedding']) == self.vector_dim:
                embeddings.append(doc['full_embedding'])
                ids.append(str(doc['_id']))
                
        if not embeddings:
            return None, {}
            
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        
        index, id_mapping = self.build_index(embeddings_matrix, ids)
        logger.info(f"Global ATS Vector Search Index updated with {len(ids)} candidates.")
        return index, id_mapping

# Singleton instance
vector_store = VectorStore()
