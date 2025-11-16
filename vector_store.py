#vector_store.py 

# vector_store.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

class VectorStore:
    def __init__(self,
                 model_name="paraphrase-multilingual-MiniLM-L12-v2",
                 index_path="index.faiss",
                 meta_path="meta.json"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index_path = index_path
        self.meta_path = meta_path
        # use inner-product on normalized vectors -> cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.id_to_meta = []
        # if files exist, try to load
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load()

    def _normalize(self, emb: np.ndarray):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norms + 1e-10)

    def add(self, texts, metas):
        """
        texts: list[str]
        metas: list[dict] (meta should include a "text" field for the chunk content)
        """
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = self._normalize(emb)
        self.index.add(emb)
        for m in metas:
            self.id_to_meta.append(m)

    def search(self, query, k=5):
        # if index has no vectors, return empty
        try:
            ntotal = self.index.ntotal
        except Exception:
            # some faiss index types might not have ntotal property exposed until loaded
            ntotal = getattr(self.index, "ntotal", 0)

        if ntotal == 0:
            # no vectors indexed yet
            return []

        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb)
        D, I = self.index.search(q_emb, k)
        results=[]
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.id_to_meta[idx]
            results.append({"score": float(score), "meta": meta})
        return results


    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_meta, f, ensure_ascii=False, indent=2)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.id_to_meta = json.load(f)