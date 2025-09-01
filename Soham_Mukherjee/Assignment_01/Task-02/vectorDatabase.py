### requirements: faiss-cpu; numpy; sentence-transformers


import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorDatabase:
    def __init__(self, model_name='all-MiniLM-L6-v2') ->None:
        logging.info(f"let's load the model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []

    def build_index(self, documents: list) ->None:
        self.documents = documents
        texts = [doc['text'] for doc in self.documents]
        
        logging.info(f"making embeddings for {len(texts)} chunks, hang tight")
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        
        logging.info(f"building the faiss index, dim={self.dimension}")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        logging.info(f"all done, index has {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> list:
        if self.index is None:
            logging.error("hey, no index yet! run build_index() first")
            return []
        
        logging.info(f"searching for the top {k} hits for: {query}")
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = [self.documents[i] for i in indices[0]]
        logging.info(f"found {len(results)} good ones")
        return results
