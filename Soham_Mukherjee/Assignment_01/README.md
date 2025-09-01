## Assignment-01

- Individual Running Instructions are given at the top of the files

- Environment Variables:
  - `GEMINI_API_KEY`: Your Google API key for accessing Google services.
  - `SERPAPI_API_KEY`: Your SerpAPI key for accessing SerpAPI services.
  - `OPENAI_API_KEY`: Your OpenAI API key for accessing OpenAI services.
- Requirements:

```
google-generative-ai
langchain
beautifulsoup4
requests
faiss-cpu
numpy
sentence-transformers
duckduckgo-search
selenium
db-sqlite3
```

- Virtual Environment :
  `conda create -n myenv python=3.13 -y`
- Activate the environment:
  `conda activate myenv`
- Install dependencies:
  `Install using pip install -r requirements.txt`

### Task-01: Build web-search Agents

- Implement a web search agent using
  - DuckDuckGo API.
  - WebScrapper(agmarknet.gov.in) using `bs4` and `selenium`.
  - serpAPI

The requirements of the project include:

- Python 3.13

#### Running Instructions are given at the top of the files

### Task-02: Build a RAG system

- Running Instruction:

```Bash
python RAG.py
```

- then give the query in the terminal as asked.

#### Description:

- Implement a Retrieval-Augmented Generation (RAG) system using this automated pipeline:
  - `Web_Search` agent.
  - `WebScrapper(agmarknet.gov.in)` agent using `bs4`.
  - `vector_database` for storing chunks of data.
  - `RAG` for generating responses for custom user queries.
