### requirements: requests
### running instructions: python duckduckgo.py


import os, sys, time, json, random, logging, requests, argparse
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ddg")

DDG_URL = os.getenv("DDG_URL")
GCS_URL = os.getenv("GCS_URL")

class netfail(Exception):
    pass

# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
# search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
# search.invoke("Obama")

def _http(url: str, params: Dict[str, Any], retries: int = 3, timeout: int = 8):
    """make an http get request with retries and timeout"""
    
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code >= 500:
                raise netfail(f"bad upstream {r.status_code}")
            r.raise_for_status()
            return r.json()
        
        except Exception as e:
            last = e
            wait = min(2 ** i, 8) + random.random()
            logger.warning(f"retry {i+1}/{retries} in {wait:.1f}s: {e}")
            time.sleep(wait)
    
    raise netfail(f"failed after retries: {last}")

def _ddg_params(q: str):
    """build duckduckgo query parameters"""
    
    return {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1, "t": "agent"}

def ddg(q: str):
    """perform duckduckgo search for the given query"""
    
    return _http(DDG_URL, _ddg_params(q))

def _flatten(rt: List[Dict[str, Any]]):
    """flatten related topics from duckduckgo results"""
    
    out = []
    
    for item in rt or []:
    
        if "Topics" in item:
    
            for t in item.get("Topics", []):
    
                if t.get("FirstURL") and t.get("Text"):
                    out.append({"title": t["Text"], "url": t["FirstURL"]})
    
        else:
    
            if item.get("FirstURL") and item.get("Text"):
                out.append({"title": item["Text"], "url": item["FirstURL"]})
    
    seen, dedup = set(), []
    
    for x in out:
    
        if x["url"] not in seen:
            seen.add(x["url"])
            dedup.append(x)
    
    return dedup

def gcs(q: str, num: int = 5):
    
    """perform google custom search for the given query"""
    
    key, cx = os.getenv("GOOGLE_CSE_KEY"), os.getenv("GOOGLE_CSE_CX")
    
    if not key or not cx:
        logger.info("gcs not configured")
        return []
    
    data = _http(GCS_URL, {"key": key, "cx": cx, "q": q, "num": num})
    
    items = data.get("items", [])
    
    return [{"title": it.get("title"), "url": it.get("link"), "snippet": it.get("snippet")} for it in items]

def _pick(d: Dict[str, Any]):
    """pick the best answer from duckduckgo result dict"""
    if d.get("Answer"): 
        return {"kind": "answer", "value": d["Answer"]}
    
    if d.get("Definition"): 
        return {"kind": "definition", "value": d["Definition"]}
    
    if d.get("AbstractText"): 
        return {"kind": "abstract", "value": d["AbstractText"]}
    
    if d.get("Heading") and d.get("Abstract"): 
        return {"kind": "abstract", "value": d["Abstract"]}
    
    else: 
        return None

def search(q: str, want: int = 6):
    """search using duckduckgo and google custom search, return structured result"""
    
    logger.info(f"searching: {q}")
    
    try:
        raw = ddg(q)
    
    except Exception as e:
        logger.error(f"ddg err: {str(e)}")
        raw = {}
    ans = _pick(raw) if raw else None
    related = _flatten(raw.get("RelatedTopics", [])) if raw else []
    
    if ans:
        return {"query": q, "mode": "instant", "answer": ans, "related": related[:want]}
    
    cse = gcs(q, num=want)
    
    if related and not cse:
        return {"query": q, "mode": "related_only", "related": related[:want], "hint": "try broader search"}
    
    if cse:
        return {"query": q, "mode": "fallback_cse", "related": related[:want], "cse_results": cse}
    
    return {"query": q, "mode": "dry", "related": related[:want], "hint": "nothing solid"}

def as_text(r: Dict[str, Any]):
    """convert search result dict to readable text"""
    
    lines = []
    
    if r.get("answer"): lines.append(f"{r['answer']['kind']}: {r['answer']['value']}")
    
    if r.get("related"):
        lines.append("related:")
        for x in r["related"]:
            lines.append(f"- {x['title']} -> {x['url']}")
    
    if r.get("cse_results"):
        lines.append("web:")
    
        for x in r["cse_results"]:
            lines.append(f"- {x['title']} -> {x['url']}")
    
    if r.get("hint"): lines.append(r["hint"])
    
    return "\n".join(lines)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
    p.add_argument("q", nargs="+")
    p.add_argument("--json", dest="as_json", action="store_true")
    p.add_argument("--n", type=int, default=6)
    a = p.parse_args()
    q = " ".join(a.q)
    
    try:
        r = search(q, want=a.n)
    
    except Exception as e:
        logger.error(str(e))
        sys.exit(2)
    
    if a.as_json:
        print(json.dumps(r, ensure_ascii=False))
    
    else:
        print(as_text(r))
