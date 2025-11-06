import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse

import traceback
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

# --- This is the key to your "docs-only" requirement ---
# This prompt template FORCES the AI to only use the context.
# If the answer isn't in the context, it MUST say so.
TEMPLATE_STRING = """
make the answer using ONLY the provided context and make it short and concise.
Context:
{context}

Question:
{question}

Answer:
"""

# --- Global App State ---
# We will load the models and DB at startup
app_state = {}

# --- FastAPI App Setup ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Load all the heavy models and database on server startup.
    This makes the first request fast.
    """
    print("Loading models and database... This may take a moment.")
    
    # Initialize Models
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL)
    
    # Load existing vector database
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        print("Please run 'python3 ingest.py' first.")
        # In a real app, you might want to exit, but here we'll let it run
        # so the user can see the error on the frontend.
        app_state["rag_chain"] = None
        return

    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    
    # Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 5})   
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(TEMPLATE_STRING)

    # Create the RAG chain
    rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

    
    # Store the chain in the global state
    app_state["rag_chain"] = rag_chain
    print("Database and models loaded successfully.")


# --- API Endpoints ---

class AskRequest(BaseModel):
    question: str

async def stream_answer(question: str) -> AsyncGenerator[str, None]:
    """
    A generator function that streams the RAG chain's response.
    """
    rag_chain = app_state.get("rag_chain")
    if rag_chain is None:
        yield "Error: Database not found. Please run the ingestion script."
        return

    try:
        async for chunk in rag_chain.astream(question):
            if isinstance(chunk, str):
                yield chunk
            else:
                yield str(chunk)
    except Exception as e:
        print(f"Error during RAG streaming: {e}")
        traceback.print_exc()
        yield "An internal error occurred."

@app.post("/ask")
async def ask(request: AskRequest):
    """
    The main API endpoint for asking questions.
    It streams back the response.
    """
    return StreamingResponse(
        stream_answer(request.question), 
        media_type="text/plain"
    )

# --- Static File Serving ---
# Mount the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    """
    Serves the main index.html file.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("Error: index.html not found.", status_code=500)
    return FileResponse(index_path)


# --- Main entry point ---
if __name__ == "__main__":
    print("--- Starting DocuBot Server ---")
    print(f"Access the UI at: http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)