from langchain.agents import initialize_agent, tool
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchResults
from langchain_community.utilities import SerpAPIWrapper
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

serp_search = SerpAPIWrapper(params = {"num" : 10, "gl" : "in"})
tavily_search = TavilySearchResults(max_results = 10)
duckduckgo_search = DuckDuckGoSearchResults(locale = "in-en", max_results = 10)


def safe_tavily_search(query: str):
    """
    Searches the web using Tavily and returns the results in a dictionary containing success as True, results and error as None.
    If the search fails, returns a dictionary with success as False, results as [] and an error message.
    """
    try:
        raw_result = tavily_search.invoke(query)
        result = []
        for r in raw_result:
            result.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content")
            })
        return {"success": True, "results": result, "error": None}
    except Exception as e:
        return {"success": False, "results": [], "error": str(e)}
    

def safe_serp_search(query: str):
    """
    Searches the web using SerpAPI and returns the results in a dictionary containing success as True, results and error as None.
    If the search fails, returns a dictionary with success as False, results as [] and an error message.
    """
    try:
        raw_result = serp_search.results(query)
        result = []
        for r in raw_result.get("organic_results", []):
            result.append({
                "title": r.get("title"),
                "url": r.get("link"),
                "content": r.get("snippet")
            })
        return {"success": True, "results": result, "error": None}
    except Exception as e:  
        return {"success": False, "results": [], "error": str(e)}
    

def safe_duckduckgo_search(query: str):
    """
    Searches the web using DuckDuckGo and returns the results in a dictionary containing success as True, results and error as None.
    If the search fails, returns a dictionary with success as False, results as [] and an error message.
    """
    try:
        raw_result = duckduckgo_search.invoke(query)
        result = []
        for r in raw_result:
            result.append({
                "title": r.get("title"),
                "url": r.get("link"),
                "content": r.get("body")
            })
        return {"success": True, "results": result, "error": None}
    except Exception as e:
        return {"success": False, "results": [], "error": str(e)}
    
@tool
def fallback_search(query: str):
    """
    Fallback search function that tries Tavily, then SerpAPI, and finally DuckDuckGo.
    Returns the first successful search result.
    """
    result = safe_tavily_search(query)
    if result["success"]:
        return {"agent" : "Tavily Search", "final_result" : result["results"]}
    
    result = safe_serp_search(query)
    if result["success"]:
        return {"agent" : "SERP Search", "final_result" : result["results"]}
    
    result = safe_duckduckgo_search(query)
    if result["success"]:
        return {"agent" : "DuckDuckGo Search", "final_result" : result["results"]}
    
    return {"error": "All searches failed", "final_result": []}



llm = HuggingFaceEndpoint(
    repo_id = "moonshotai/Kimi-K2-Instruct",
    task = "text-generation",
    temperature=0.1
)

model = ChatHuggingFace(llm = llm)

agent = initialize_agent(
    tools = [fallback_search], llm = model, agent = "zero-shot-react-description", verbose = False
)