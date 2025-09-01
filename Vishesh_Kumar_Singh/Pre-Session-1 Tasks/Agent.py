import dotenv
dotenv.load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from Web_Search import WebSearch, scrape_page

# Function to process web search results and scrape content
def web_results(results):
    if results == "ERROR":
        return "ERROR"
    
    scraped_results = []
    for result in results:
        if result == "ERROR":
            continue
        url=result['url']
        if url:
            scraped_text = scrape_page(url)
            if scraped_text != "ERROR":
                doc=Document(page_content=scraped_text, metadata={"source": url})
                scraped_results.append(doc)
    return scraped_results if scraped_results else "ERROR"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def RAG(query):
    results = WebSearch(query)
    scraped_docs = web_results(results)
    if scraped_docs == "ERROR":
        return "ERROR: Failed to scrape web results"

    print("Encoding text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(scraped_docs)
    texts = [doc.page_content for doc in all_splits]
    vectorstore = FAISS.from_texts(texts, embed_model)

    print("Retrieving Relevant Information...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    return context



prompt_template_main = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a Philosophy Proffesor who tries to explain evrything intutively to your students rather than just throwing techinal terms. Answer all the questions to the best of your ability, but also make sure to not give any information you are not sure about.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

prompt_template_refine = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a part of ai agent workflow. Here we are doing webscrapping for the query but the query can also contain references from past conversation. Your task is to convert the queries passed into an ideal query to be for web scrapping thorugh websearch agents.
            You also have option to return 'NO' if web search is not required at all.
            Be very precise to either return the refined query or 'NO' if web search is not required, you're not supposed to give the answer itself ot the user query.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)



memory = MemorySaver()
model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
workflow1 = StateGraph(state_schema=MessagesState)
workflow2 = StateGraph(state_schema=MessagesState)

def call_model_main(state: MessagesState):
    prompt=prompt_template_main.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

def call_model_refine(state: MessagesState):
    prompt=prompt_template_refine.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


workflow1.add_edge(START, "model")
workflow1.add_node("model", call_model_main)
workflow2.add_edge(START, "model")
workflow2.add_node("model", call_model_refine)

app1 = workflow1.compile(checkpointer=memory)
app2 = workflow2.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "000"}}




def Answer(query):
    prompt0 = f"""Crude Question: {query}
            Query for Web Search:"""
    Web_input=[HumanMessage(content=prompt0)]
    web_prompt = app2.invoke({"messages": Web_input},config)
    web_query=web_prompt["messages"][-1].content

    
    
    if web_query == "NO":
        input_messages = [HumanMessage(content=query)]
        output = app1.invoke({"messages": input_messages}, config)
    else:
        print(f"Searching the web for: {web_query}")
        context = RAG(web_query)
        print("Finishing Up...\n\n")
        prompt = f""" Context from Web: {context}

            Question: {query}
            Answer:"""
        input_messages = [HumanMessage(content=prompt)]
        output = app1.invoke({"messages": input_messages}, config)
    

    return output["messages"][-1]