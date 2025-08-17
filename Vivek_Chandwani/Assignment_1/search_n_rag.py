from search_agent import fallback_search
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()

def get_content(query: str):
    content = []
    try:
        results_list = (fallback_search.invoke(query)).get("final_result")
        for res in results_list:
            content.append(res.get("content", ""))
        return content
    except Exception as e:
        print(f"Error fetching content: {e}")

def split(docs):
    documents = [Document(page_content = doc) for doc in docs]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

def store(chunks):
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="SEARCH_RESULTS"
    )

template = PromptTemplate(
    template = "Answer the question based on the provided context: \n Context: {context} \n Question: {question}\n Also provide the exact reference from the document",
    input_variables=["context", "question"]
)

llm = HuggingFaceEndpoint(
    repo_id = "moonshotai/Kimi-K2-Instruct",
    task = "text-generation",
    temperature=0.1
)

model = ChatHuggingFace(llm = llm)

def retrieve_and_answer(query: str):
    content = get_content(query)
    if not content:
        return "No content found for the query."

    chunks = split(content)
    vector_store = store(chunks)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )

    parser = StrOutputParser()

    chain = template | model | parser
    
    final = chain.invoke({
        "context": "\n\n".join(doc.page_content for doc in retriever.invoke(query)),
        "question": query
    })
    
    if final:
        return final
    else:
        return "No relevant information found."
    
print(retrieve_and_answer("Who was the chief guest at the 2025 Independence Day celebration in Delhi?"))