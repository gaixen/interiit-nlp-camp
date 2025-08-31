import numpy as np
import re
import time
import logging
from typing import List

from data_models import ScrapedContent, DocumentChunk, RAGResult

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ImprovedRAGAgent:
    """RAG agent for chunking, indexing, and querying content."""
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, use_embeddings: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_chunks: List[DocumentChunk] = []
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE
        
        if self.use_embeddings:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = None
            self.embeddings = None
            logger.info("SentenceTransformer model loaded for embeddings.")
        else:
            self.model = None
            if use_embeddings and not EMBEDDINGS_AVAILABLE:
                logger.warning("Embeddings enabled, but sentence-transformers or faiss not installed. Disabling.")
        logger.info(f"RAG Agent initialized: chunk_size={chunk_size}, use_embeddings={self.use_embeddings}")
    
    def _preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_text(self, text: str, source_url: str) -> List[DocumentChunk]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(DocumentChunk(
                content=self._preprocess_text(chunk_text),
                source_url=source_url,
                chunk_index=len(chunks),
                metadata={"word_count": len(chunk_words)}))
        return chunks

    async def index_documents(self, documents: List[ScrapedContent]):
        logger.info(f"Indexing {len(documents)} new documents.")
        new_chunks = []
        for doc in documents:
            if doc.success and doc.content and len(doc.content.strip()) > 100:
                chunks = self._chunk_text(doc.content, doc.url)
                new_chunks.extend(chunks)
        
        self.document_chunks.extend(new_chunks)
        logger.info(f"Created {len(new_chunks)} new chunks. Total chunks in index: {len(self.document_chunks)}")

        if self.use_embeddings and self.model and new_chunks:
            logger.info(f"Generating embeddings for {len(new_chunks)} new chunks and rebuilding FAISS index...")
            texts = [chunk.content for chunk in new_chunks]
            new_embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
            if self.embeddings is None: self.embeddings = new_embeddings
            else: self.embeddings = np.vstack([self.embeddings, new_embeddings])

            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            normalized_embeddings = self.embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)
            self.index.add(normalized_embeddings.astype('float32'))
            logger.info("FAISS index rebuild complete.")

    def _calculate_keyword_similarity(self, query: str, chunk: DocumentChunk) -> float:
        query_words = set(self._preprocess_text(query).split())
        chunk_words = set(chunk.content.split())
        if not query_words: return 0.0
        intersection, union = len(query_words.intersection(chunk_words)), len(query_words.union(chunk_words))
        return intersection / union if union > 0 else 0.0

    async def query(self, query: str, top_k: int = 5) -> RAGResult:
        logger.info(f"Performing RAG query for: '{query}'")
        start_time = time.time()
        relevant_chunks = []
        
        if not self.document_chunks:
            logger.warning("Query attempted but no documents are indexed.")
            return RAGResult(query, [], "No indexed documents available.", 0.0, [], 0, 0)

        if self.use_embeddings and self.index:
            logger.info("Using embedding-based search with keyword re-ranking.")
            query_embedding = self.model.encode([self._preprocess_text(query)])
            faiss.normalize_L2(query_embedding)
            k_search = min(top_k * 3, len(self.document_chunks))
            scores, indices = self.index.search(query_embedding.astype('float32'), k_search)
            
            candidate_chunks = []
            for score, idx in zip(scores[0], indices[0]):
                chunk = self.document_chunks[idx]
                keyword_score = self._calculate_keyword_similarity(query, chunk)
                combined_score = 0.7 * float(score) + 0.3 * keyword_score
                chunk.metadata['similarity_score'] = combined_score
                candidate_chunks.append((combined_score, chunk))
            candidate_chunks.sort(reverse=True, key=lambda x: x[0])
            relevant_chunks = [chunk for _, chunk in candidate_chunks[:top_k]]
        else:
            logger.info("Using keyword-based search.")
            scored_chunks = []
            for chunk in self.document_chunks:
                score = self._calculate_keyword_similarity(query, chunk)
                if score > 0:
                    chunk.metadata['similarity_score'] = score
                    scored_chunks.append((score, chunk))
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            relevant_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks in {retrieval_time:.2f}s.")
        
        # Simple response generation
        generation_start = time.time()
        if relevant_chunks:
            context = "\n\n".join([f"Source: {chunk.source_url}\n{chunk.content}" for chunk in relevant_chunks[:3]])
            response = f"Based on the retrieved context for '{query}', here is a summary:\n\n{context}"
            top_scores = [chunk.metadata.get('similarity_score', 0) for chunk in relevant_chunks]
            confidence = float(np.mean(top_scores)) if top_scores else 0.0
        else:
            response = f"No relevant information found for query: {query}"
            confidence = 0.0
        
        return RAGResult(
            query=query, relevant_chunks=relevant_chunks, generated_response=response,
            confidence_score=confidence, sources=list(set(c.source_url for c in relevant_chunks)),
            retrieval_time=retrieval_time, generation_time=time.time() - generation_start)