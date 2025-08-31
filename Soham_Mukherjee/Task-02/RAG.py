### requirements: google-generative-ai; web_search; scrape_and_chunk; VectorDatabase
### env variables: GEMINI_API_KEY

import logging
import os
import sys
from webSearch import web_search
from webScrapper import scrape_and_chunk
from vectorDatabase import VectorDatabase
import google.generativeai as genai
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger('RAG_APP')
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - RAG_APP - %(levelname)s - %(message)s'))
log.addHandler(handler)

load_dotenv()

class RAGAgent:
    def __init__(self, model_name='gemini-1.5-flash'):
        self.vdb = VectorDatabase()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please create a .env file and add it.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        log.info(f"RAG Agent initialized with model: {model_name}")

    def process_query(self, query):
        log.info("RAG pipeline for your query...")
        
        urls = web_search(query)
        if not urls:
            log.warning("search agent returned no URLs")
            return ""
        
        documents = scrape_and_chunk(urls)
        if not documents:
            log.warning("scrapper yielded no content.")
            return ""
            
        self.vdb.build_index(documents)
        
        log.info("Retrieving the most relevant context...")
        retrieved_docs = self.vdb.search(query, k=5)
        
        if not retrieved_docs:
            log.warning("No relevant documents found after indexing.")
            return ""

        context = "\n".join([doc['text'] for doc in retrieved_docs])
        sources = list(set([doc['source'] for doc in retrieved_docs]))
        
        prompt = f"""
        Based ONLY on the following context, provide a clear and concise answer to the question. Dont use any other information.
        Context:{context};Question: {query}
        """
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
            log.info("Successfully generated the answer.")
            return answer, sources
        except Exception as e:
            log.error(f"An error occurred while calling the Gemini API: {str(e)}")
            return "Sorry, there was an unexpected error while generating the final answer.", []

def main():
    try:
        rag_agent = RAGAgent()
        
        while True:
            user_query = input("\nUser Query (or type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                print("Exiting. Goodbye!")
                break
            if not user_query.strip():
                print("Query cannot be empty. Please try again.")
                continue

            answer, sources = rag_agent.process_query(user_query)
            print(f"Answer:\n\n{answer}")
            
            if sources:
                print("\nSources:")
                for source in sources:
                    print(f" {source}")

    except ValueError as e:
        log.error(e)
    except Exception as e:
        log.error(f"{str(e)}")

if __name__ == '__main__':
    main()
