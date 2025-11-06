import os
import uvicorn
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
    
        # NEW IMPORTS: Use the langchain-ollama package
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

TEMPLATE_STRING = """
make the answer using ONLY the provided context and make it short and concise.
Context:
{context}

Question:
{question}

Answer:
"""

app = FastAPI()

# Global variables to hold the models and DB
db_retriever = None
rag_chain = None

@app.on_event("startup")
def startup_event():
    global db_retriever, rag_chain
    
    print("--- Server is starting up... ---")
    
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        print("Please run 'docker compose exec app python3 ingest.py' first.")
        return
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
    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    db_retriever = db.as_retriever(search_kwargs={"k": 5})

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(TEMPLATE_STRING)

    # Create the RAG chain
    rag_chain = (
        {"context": db_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("--- Database and models loaded successfully. Server is ready. ---")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    if not rag_chain:
        return Response(content="Error: RAG chain is not initialized. Check server logs.", status_code=500)
    
    try:
        # Return a streaming response
        return StreamingResponse(
            rag_chain.stream(query.question), 
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Error during streaming: {e}")
        return Response(content=f"Error: {e}", status_code=500)

# Serve the main HTML file
@app.get("/")
async def get_root():
    return FileResponse('index.html')

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    print("--- Starting DocuBot Server ---")
    print(f"Access the UI at: http://127.0.0.1:8000")
    # Host="0.0.0.0" is crucial for Docker
    uvicorn.run(app, host="0.0.0.0", port=8000)