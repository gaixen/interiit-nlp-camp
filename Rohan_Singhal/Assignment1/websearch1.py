
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

import pprint
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

def fallback_response(input_text):
    logging.warning("Using fallback response for input: %s", input_text)
    
    return [{"role": "assistant", "content": "Sorry I couldnt process your request."}]


tools = []
try:
    tools = [TavilySearch(k=3)]
    logging.info("TavilySearch initialized successfully.")
except Exception as e:
    logging.error("Error initializing TavilySearch: %s", e)


try:
    llm = ChatGroq(model="llama-3.1-8b-instant")
    agent = create_react_agent(llm, tools=tools)
    logging.info("LLM and agent initialized successfully.")
    ## input tect
    input_text = "give me latest AI NEWS"
    try:
        logging.info("Invoking agent with input: %s", input_text)
        response = agent.invoke({"messages": [{"role": "user", "content": input_text}]})
        messages = response.get("messages", [])
        if not messages:
            raise ValueError("No messages returned from agent.")
    except Exception as agent_error:
        logging.error("Agent error: %s", agent_error)
        messages = fallback_response(input_text)
except Exception as llm_error:
    logging.error("LLM or agent initialization error: %s", llm_error)
    input_text = "What is the capital of France?"
    messages = fallback_response(input_text)


for message in messages:
    print(50*"=")
    pprint.pprint(message)
