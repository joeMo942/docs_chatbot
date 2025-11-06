import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Configuration ---
DOCS_PATH = "./docs"
DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"

# --- LOCAL CHANGE ---
OLLAMA_BASE_URL = "http://localhost:11434"
# --- END OF CHANGE ---

# Batch sizes for reliable processing
FILE_BATCH_SIZE = 500  # Process 500 files from disk at a time
CHUNK_EMBED_BATCH_SIZE = 200 # Embed 200 chunks at a time

print("Starting batched ingestion process...")

# 1. Initialize models and database
try:
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    print("Ollama embeddings initialized.")
except Exception as e:
    print(f"Error initializing Ollama embeddings: {e}")
    print("Is your local 'ollama serve' running?")
    exit()
    
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
print("ChromaDB and text splitter initialized.")

# 2. Walk the directory and collect all file paths
all_filepaths = []
for root, dirs, files in os.walk(DOCS_PATH):
    for file in files:
        if file.endswith(".txt"):
            all_filepaths.append(os.path.join(root, file))

if not all_filepaths:
    print(f"No .txt documents found in {DOCS_PATH}. Did 'convert.py' run correctly?")
    exit()

total_files = len(all_filepaths)
print(f"Found {total_files} total .txt files to process.")

# 3. Process files in "File Batches"
for i in range(0, total_files, FILE_BATCH_SIZE):
    batch_filepaths = all_filepaths[i : i + FILE_BATCH_SIZE]
    
    print(f"\n--- Processing File Batch {i//FILE_BATCH_SIZE + 1}/{(total_files + FILE_BATCH_SIZE - 1)//FILE_BATCH_SIZE} ---")
    print(f"Loading {len(batch_filepaths)} files (files {i+1} to {min(i + FILE_BATCH_SIZE, total_files)})...")

    # Load only the files for this batch
    batch_docs = []
    for filepath in batch_filepaths:
        try:
            loader = TextLoader(filepath, encoding="utf-8")
            batch_docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Skipping file {filepath} due to error: {e}")

    if not batch_docs:
        print("No documents loaded in this batch, skipping.")
        continue

    # Split the batch documents
    chunks = text_splitter.split_documents(batch_docs)
    total_chunks = len(chunks)
    print(f"Split {len(batch_docs)} documents into {total_chunks} chunks.")
    
    if not chunks:
        print("No chunks created in this batch, skipping.")
        continue

    # 4. Process chunks in "Embedding Batches"
    for j in range(0, total_chunks, CHUNK_EMBED_BATCH_SIZE):
        chunk_batch = chunks[j : j + CHUNK_EMBED_BATCH_SIZE]
        
        print(f"  -> Embedding chunk batch {(j//CHUNK_EMBED_BATCH_SIZE) + 1}/{(total_chunks + CHUNK_EMBED_BATCH_SIZE - 1)//CHUNK_EMBED_BATCH_SIZE} ({len(chunk_batch)} chunks)...")
        try:
            db.add_documents(chunk_batch)
            print(f"  -> Successfully added {len(chunk_batch)} chunks to DB.")
        except Exception as e:
            print(f"Error adding chunk batch to database: {e}")
            
    print(f"--- Finished File Batch {i//FILE_BATCH_SIZE + 1} ---")

# 5. Finalize
print("\n--- All batches complete. ---")
print("Finalizing database (persisting changes)...")
print("Ingestion complete. Your database is ready!")