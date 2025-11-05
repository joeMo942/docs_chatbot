import os
# Loaders are still in langchain_community
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# UPDATED IMPORT: Switched from langchain_community to langchain-ollama
from langchain_ollama import OllamaEmbeddings

# --- Configuration ---
DOCS_PATH = "./docs"
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

print("Starting ingestion process...")

# 1. Load all .txt documents from the 'docs' folder
loader = DirectoryLoader(
    DOCS_PATH, 
    glob="**/*.txt", 
    loader_cls=TextLoader, 
    show_progress=True,
    use_multithreading=True
)
documents = loader.load()

if not documents:
    print(f"No documents found in {DOCS_PATH}. Did 'convert.py' run correctly?")
    exit()

print(f"Loaded {len(documents)} documents.")

# 2. Split the documents into small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Split documents into {len(chunks)} chunks.")

# 3. Initialize the embedding model (via Ollama)
# Use the new class from langchain-ollama
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# 4. Create a new Chroma vector database and ingest the chunks
print("Creating vector database... This may take a moment.")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("---")
print(f"Successfully created vector database at {DB_PATH}")
print("Ingestion complete. You can now run 'main.py'.")