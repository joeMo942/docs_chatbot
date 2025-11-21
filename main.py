import os
import uvicorn
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

# --- IMPORTANT DOCKER CHANGE ---
# We are now talking to the 'ollama' container, not 'localhost'
OLLAMA_BASE_URL = "http://ollama:11434"
# --- END OF CHANGE ---

# --- Prompts ---
# 1. Contextualize Question Prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 2. QA Prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- Global App State ---
app_state = {}
store = {} 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("--- Server is starting up... ---")
    print("Loading models and database... This may take a moment.")
    
    # Initialize models, pointing them to the Ollama container
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL, 
        base_url=OLLAMA_BASE_URL
    )
    llm = ChatOllama(
        model=LLM_MODEL, 
        base_url=OLLAMA_BASE_URL
    )

    # Load the existing vector database
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        print("Please run 'docker compose exec app python3 ingest.py' first.")
        app_state["rag_chain"] = None
        return

    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    
    # Create retriever with MMR
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    # --- Create the Conversational RAG Chain ---
    
    # 1. History Aware Retriever
    history_aware_retriever = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }
        | contextualize_q_prompt
        | llm
        | StrOutputParser()
        | retriever
    )

    # Define the final QA chain
    rag_chain = (
        {
            "context": history_aware_retriever,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"]
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with Message History
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # Store the chain in the global state
    app_state["rag_chain"] = conversational_rag_chain
    print("--- Database and models loaded successfully. Server is ready. ---")


# --- API Endpoints ---

class AskRequest(BaseModel):
    question: str
    session_id: str = "default_session"

async def stream_answer(question: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    A generator function that streams the RAG chain's response.
    """
    rag_chain = app_state.get("rag_chain")
    if rag_chain is None:
        yield "Error: Database not found. Please run the ingestion script."
        return

    try:
        async for chunk in rag_chain.astream(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        ):
            if isinstance(chunk, str):
                yield chunk
            else:
                yield str(chunk)
    except Exception as e:
        print(f"Error during RAG streaming: {e}")
        yield f"An error occurred: {e}"

@app.post("/ask")
async def ask(request: AskRequest):
    return StreamingResponse(
        stream_answer(request.question, request.session_id), 
        media_type="text/plain"
    )

# Serve the main HTML file
@app.get("/")
async def get_root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return Response(content="Error: index.html not found.", status_code=500)
    return FileResponse(index_path)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    print("--- Starting DocuBot Server ---")
    print(f"Access the UI at: http://127.0.0.1:8000")
    # Host="0.0.0.0" is crucial for Docker
    uvicorn.run(app, host="0.0.0.0", port=8000)