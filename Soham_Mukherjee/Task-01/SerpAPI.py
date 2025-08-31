### requirements: google-search-results; openai
### env variables: OPENAI_API_KEY; SERPAPI_API_KEY

from typing import List, Dict, Any, Optional, Union
import argparse, json, os, logging
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from serpapi import GoogleSearch
# from serpAPI import GoogleSearch

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class AGENT:
    """LLMpowered researcher that combines OpenAI models with SerpAPI."""

    def __init__(self,model: str = "o3",topn: int = 10,debug: bool = False,openai_key: Optional[str] = None,serpapi_key: Optional[str] = None,) -> None:
        """initialize the agent with model, topn, debug, and api keys"""
        self.model = model
        self.topn = topn
        self.debug = debug
        try: 
            self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
            self.serp_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
            logger.debug("API loading successful")
        except Exception as e:
            logger.error("error loading env. variables: %s", e)
        if not self.openai_key or not self.serp_key:
            raise RuntimeError("OPENAI_API_KEY and SERPAPI_API_KEY must be set.")

        self.client = OpenAI(api_key=self.openai_key)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search_agent",
                    "description": "Search Google and return the top result snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Google search string",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        self.sys_prompt = (
            "You are a careful and thorough research assistant.\n"
            "If you need information beyond your training, list all required `search_web` tool calls together in one assistant message before reviewing any results.\n\n"
            "Return the tool calls in the JSON format expected for `tool_calls`, with each call including its own `id`, `type`, and `function` fields.\n"
            "Skip any explanations—just provide the tool calls.\n\n"
            "Example:\n"
            "{\n"
            "  \"tool_calls\": [\n"
            "    {\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"search_web\", \"arguments\": \"{\\\"query\\\": \\\"first topic\\\"}\"}},\n"
            "    {\"id\": \"call_2\", \"type\": \"function\", \"function\": {\"name\": \"search_web\", \"arguments\": \"{\\\"query\\\": \\\"second topic\\\"}\"}}\n"
            "  ]\n"
            "}\n\n"
            "Batch between 2 and 50 calls in a single turn if you need outside data.\n"
            "After all tool outputs are returned, write your final answer with clear citations."
        )

    def _search_web(self, query: str) -> str:
        """search the web using the provided query string"""
        if self.debug:
            logger.debug(f"SerpAPI query: '{query}'")
        logger.info(f"Searching web for query: {query}")
        search = GoogleSearch({"q": query, "api_key": self.serp_key, "num": self.topn})
        org = search.get_dict().get("organic_results", [])[: self.topn]
        logger.info(f"Received {len(org)} results for query: {query}")
        return "".join(
            f"- {r.get('title','(untitled)')}: {r.get('snippet','(no snippet)')}" for r in org
        ) or "No results found."

    def run(self, question: str) -> dict[str, Any]:
        """run the agent to answer the given question"""
        logger.info(f"Received question: {question}")
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": question},
        ]
        steps: list[dict[str, Any]] = []

        while True:
            if self.debug:
                logger.debug("OpenAI chat.completions.create request …")
            logger.info("Sending request to OpenAI model.")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                logger.info(f"Model requested {len(msg.tool_calls)} tool calls.")
                messages.append(msg)

                def fetch(call):
                    args = json.loads(call.function.arguments)
                    q = args["query"]
                    logger.debug(f"Processing tool call for query: {q}")
                    steps.append({"type": "tool_call", "query": q})
                    return call.id, q, self._search_web(q)

                with ThreadPoolExecutor() as pool:
                    results = list(pool.map(fetch, msg.tool_calls))

                for call_id, q, result in results:
                    logger.info(f"Appending tool result for query: {q}")
                    steps.append({"type": "tool_result", "content": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result,
                        }
                    )
                continue

            answer = msg.content.strip()
            logger.info("Final answer generated.")
            steps.append({"type": "assistant_answer", "content": answer})
            result: dict[str, Any] = {"question": question, "answer": answer, "steps": steps}
            logger.info("Returning result.")
            return result


def func():
    """parse cli arguments and run the agent"""
    p = argparse.ArgumentParser(description="ResearchAgent CLI")
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-m", "--model", default="gpt-4o", choices=["o3", "o4-mini", "gpt-4o"])
    p.add_argument("-n", "--topn", type=int, default=10)
    p.add_argument("-o", "--outfile", type=Path)
    p.add_argument("-d", "--debug", action="store_true")
    cfg = p.parse_args()

    logger.info(f"start researchagent CLI")

    agent = AGENT(model=cfg.model, topn=cfg.topn, debug=cfg.debug)
    result = agent.run(cfg.query)

    print("Response:\n")
    print(result["answer"])
    

    if cfg.outfile:
        cfg.outfile.write_text(json.dumps(result, indent=2))
        logger.info(f"Saved full trace to {cfg.outfile}")


if __name__ == "__main__":
    func()