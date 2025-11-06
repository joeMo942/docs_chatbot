import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles  # <-- Import StaticFiles
from pydantic import BaseModel
import sys
import os  # <-- Import os

# --- All the LangChain imports from your ask.py ---
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

# --- Load Models & Database ONCE on startup ---
print("Loading models and database... This may take a moment.")
try:
    llm = ChatOllama(model=LLM_MODEL)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Load the existing database
    db = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 5})

    print("Database and models loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR: Failed to load models or database.")
    print(f"Error: {e}")
    print("Have you run 'python3 ingest.py' successfully?")
    sys.exit(1)


# --- Define the RAG Prompt Template ---
TEMPLATE_STRING = """
CRITICAL: You are an expert programming assistant. You must answer the user's question 
based  on the documentation context provided below.

If the answer is not found in the provided context, you MUST say:
'I do not have that information in my documentation.'

Do not use any of your own internal knowledge.
Make the answer short and concise.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(TEMPLATE_STRING)

# --- Define the RAG Chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

app = FastAPI()

class Query(BaseModel):
    question: str


static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    print(f"WARNING: Created 'static' directory.")
    print(f"Please make sure 'style.css' and 'script.js' are inside it.")

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serves the main index.html file."""
    html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_file_path):
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Please create an index.html file in the same directory as main.py.</p>", status_code=404)
    return FileResponse(html_file_path)

async def get_response_stream(question: str):
    try:
        for chunk in rag_chain.stream(question):
            yield chunk
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield "An error occurred while processing your request."

@app.post("/ask")
async def ask_question(query: Query):
    return StreamingResponse(
        get_response_stream(query.question), 
        media_type="text/plain"
    )

if __name__ == "__main__":
    print("--- Starting DocuBot Server ---")
    print("Access the UI at: http://127.0.0.1:8000")
    
    # uvicorn.run() is the command to start the server.
    # host="0.0.0.0" makes it accessible from other computers on your network.
    uvicorn.run(app, host="127.0.0.1", port=8000)