from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class SearchResult:
    """Represents a single search result"""
    title: str
    url: str
    snippet: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResponse:
    """Represents a search response with multiple results"""
    success: bool
    results: List[SearchResult]
    source: str
    total_results: int
    response_time: float
    error_message: Optional[str] = None

@dataclass
class ScrapedContent:
    """Represents scraped content from a URL"""
    url: str
    title: str
    content: str
    text_length: int
    scrape_timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    content: str
    source_url: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResult:
    """Represents the result of a RAG query"""
    query: str
    relevant_chunks: List[DocumentChunk]
    generated_response: str
    confidence_score: float
    sources: List[str]
    retrieval_time: float
    generation_time: float