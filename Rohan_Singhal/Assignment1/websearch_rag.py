from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import sys
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  
    ]
)
logger = logging.getLogger("WebsearchRAG")

try:
    load_dotenv()
    logger.info("Environment variables loaded")
except Exception as e:
    logger.error(f"Failed to load environment variables: {e}")
    sys.exit(1)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,  
    separators=["\n\n", "\n", " ", ""], 
    keep_separator=True  
)
logger.info("Text splitter configured with chunk_size=500, overlap=50")

try:
    logger.info("Initializing HuggingFace embeddings...")
    embedding_function = HuggingFaceEmbeddings()
    logger.info("Embeddings initialized successfully")
    
    logger.info("Creating in-memory vector store...")
    vector_Store = Chroma(embedding_function=embedding_function)
    logger.info("Vector store created successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    logger.debug(traceback.format_exc())
    sys.exit(1)

def web_scrape_store(query: str) :
   
    logger.info(f"Starting web scraping for query: '{query}'")
    start_time = time.time()
    
    try:
        try:
            logger.info("Initializing TavilySearch...")
            web_search = TavilySearch(k=3)
        except Exception as e:
            logger.error(f"Failed to initialize TavilySearch: {e}")
            logger.debug(traceback.format_exc())
            return False
            
        try:
            logger.info(f"Executing search with query: '{query}'")
            search_results = web_search.invoke(query)
            results_count = len(search_results.get("results", []))
            logger.info(f"Retrieved {results_count} search results")
            
            if results_count == 0:
                logger.warning("No search results found")
                return False
        except Exception as e:
            logger.error(f"Search failed: {e}")
            logger.debug(traceback.format_exc())
            return False
        
        stored_chunks = 0
        for i, message in enumerate(search_results.get("results", [])):
            try:
                content = message["content"]
                url = message.get("url", "unknown_url")
                title = message.get("title", "No title")
                logger.info(f"Processing result {i+1}/{results_count}: {title}")
                
                chunks = splitter.split_text(content)
                chunk_count = len(chunks)
                logger.info(f"Split into {chunk_count} chunks")
                
                metadata_list = [
                    {
                        "source": url,
                        "title": title,
                        "chunk_index": j,
                        "total_chunks": chunk_count,
                        "result_index": i
                    } for j in range(chunk_count)
                ]
                
                vector_Store.add_texts(chunks, metadatas=metadata_list)
                stored_chunks += chunk_count
                logger.info(f"Stored {chunk_count} chunks from result {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to process result {i+1}: {e}")
                logger.debug(traceback.format_exc())
                continue

        elapsed_time = time.time() - start_time
        if stored_chunks > 0:
            logger.info(f"Successfully stored {stored_chunks} chunks in {elapsed_time:.2f} seconds")
            return True
        else:
            logger.warning("No chunks were stored")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error in web_scrape_store: {e}")
        logger.debug(traceback.format_exc())
        return False
    
try:
    logger.info("Setting up retriever...")
    retriever = vector_Store.as_retriever()
    
    logger.info("Initializing Groq LLM...")
    llm = ChatGroq(model="llama-3.1-8b-instant")
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize retriever or LLM: {e}")
    logger.debug(traceback.format_exc())
    sys.exit(1)

def format_document(result):
   
    try:
        source = result.metadata.get('source', 'Unknown source')
        title = result.metadata.get('title', 'N/A')
        chunk_index = result.metadata.get('chunk_index', 'N/A')
        total_chunks = result.metadata.get('total_chunks', 'N/A')
        content = result.page_content
        
        return f"Source: {source}\nTitle: {title}\nChunk: {chunk_index}/{total_chunks}\nContent: {content}\n"
    except Exception as e:
        logger.error(f"Error formatting document: {e}")
        return f"[Error formatting document: {str(e)}]\nContent: {result.page_content}"

def main(query: str) :

    logger.info(f"Starting RAG pipeline for query: '{query}'")
    
    stored = web_scrape_store(query)

    if not stored:
        logger.error("Web scraping failed or no content was stored")
        print("Sorry, I couldn't find any information on that topi.")
        return
    
    try:
        logger.info("Building and executing RAG chain...")
        
        chain = (
            {"context": retriever | (lambda docs: "\n\n".join([format_document(doc) for doc in docs])),
             "query": (lambda x: x)}
            | ChatPromptTemplate.from_messages([
                ('system', '''You are a helpful assistant that can answer questions on any topic. 
                
IMPORTANT: Base your response ONLY on the actual content provided in the context below. 
DO NOT just list the source websites - extract and summarize the actual content relevant to the user's query.
Include specific details, examples, quotes, and relevant information from the retrieved documents.
Cite your sources inline when providing information.

Context: {context}'''),
                ("user", "{query}")
            ])
            | llm 
            | StrOutputParser()
        )
        
        
        response = chain.invoke(query)
        print("\n" + "="*80)
        print("RESPONSE:")
        print("="*80)
        print(response)
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error executing RAG chain: {e}")
        logger.debug(traceback.format_exc())
        print(f"An error occurred while processing your query: {str(e)}")


        


if __name__ == "__main__":
    
    query = "What are the latest advancements in artificial intelligence?"       
    main(query)
        
   