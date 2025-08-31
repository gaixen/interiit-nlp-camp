"""
This implements a web search, scraping, and RAG pipeline.
"""
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from WebSearchAgents import fallback_search

load_dotenv()

class ScrapedData(BaseModel):
    """Data model for storing scraped web page information."""
    url: str
    title: str
    content: str

class QueryResult(BaseModel):
    """Data model for storing the answer and its sources."""
    answer: str
    sources: List[Dict[str, str]]

class WebScraper:
    """Scrapes web pages and extracts clean text content."""
    def scrape_url(self, url: str) -> ScrapedData:
        """Scrape a single URL and return its cleaned content and title."""
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        title = soup.find('title').get_text().strip() if soup.find('title') else "No Title"
        content = ' '.join(soup.get_text(separator=' ', strip=True).split())
        
        return ScrapedData(url=url, title=title, content=content)
    
    def scrape_from_search(self, search_results: List[Dict[str, Any]]) -> List[ScrapedData]:
        """Scrape all URLs from search results and return a list of ScrapedData."""
        scraped_data = []
        for result in search_results:
            url = result.get('source', '')
            if url:
                scraped_data.append(self.scrape_url(url))
        return scraped_data

class RAGSystem:
    """Builds a vector store from scraped data and answers questions using a language model."""
    def __init__(self):
        """Initialize embeddings, splitter, vectorstore, and language model."""
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = None
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def build_vectorstore(self, scraped_data: List[ScrapedData]) -> None:
        """Build a FAISS vector store from the provided scraped data."""
        documents = [Document(page_content=data.content, metadata={"source": data.url, "title": data.title}) 
                    for data in scraped_data if data.content]
        
        chunks = self.splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
    
    def query(self, question: str) -> QueryResult:
        """Query the vector store and return an answer and its sources."""
        docs = self.vectorstore.similarity_search(question, k=3)
        
        context = "\n\n".join([f"Source: {doc.metadata['title']}\n{doc.page_content}" for doc in docs])
        
        prompt = f"""Answer based on this context: {context}
                     Question: {question}
                     Answer:"""
        
        response = self.model.generate_content(prompt)
        
        sources = [{"url": doc.metadata["source"], "title": doc.metadata["title"]} for doc in docs]
        return QueryResult(answer=response.text, sources=sources)

class WebSearchRAG:
    """Orchestrates web search, scraping, and RAG pipeline for user queries."""
    def __init__(self):
        """Initialize the web scraper and RAG system."""
        self.scraper = WebScraper()
        self.rag = RAGSystem()
    
    def search_and_answer(self, query: str) -> QueryResult:
        """Run web search, scrape content, build vector store, and answer the query."""
        search_result = fallback_search.invoke(query)
        search_results = search_result.get("final_result", [])
        
        scraped_data = self.scraper.scrape_from_search(search_results)
        self.rag.build_vectorstore(scraped_data)
        
        return self.rag.query(query)

if __name__ == "__main__":
    system = WebSearchRAG()
    query = input("Enter your query: ")
    result = system.search_and_answer(query)
    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")