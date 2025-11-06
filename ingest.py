# this files runnes on GPU google colab 
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
DOCS_PATH = "./docs_files"
DB_PATH = "./chroma_db"

# GPU-accelerated embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("Starting ingestion process on GPU...")

# 1. Load all .txt documents
loader = DirectoryLoader(
    DOCS_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True
)

documents = loader.load()

if not documents:
    print(f"No documents found in {DOCS_PATH}. Please check your input folder.")
    exit()

print(f"Loaded {len(documents)} documents.")

# 2. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Split documents into {len(chunks)} chunks.")

# 3. Initialize GPU embeddings (replaces OllamaEmbeddings)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda"}  # ensures GPU usage
)

# 4. Create and persist Chroma database
print("Creating vector database on GPU... This may take a few minutes.")
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("\n---")
print(f"Successfully created vector database at {DB_PATH}")
print("Ingestion complete. You can now query it with main.py.")
