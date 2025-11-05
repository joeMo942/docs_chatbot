import sys
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"

TEMPLATE_STRING = """
CRITICAL: You are an expert programming assistant. You must answer the user's question 
based *only* on the documentation context provided below.

If the answer is not found in the provided context, you MUST say:
'I do not have that information in my documentation.'

Do not use any of your own internal knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ask.py \"Your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}\n")

    llm = ChatOllama(model=LLM_MODEL)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    try:
        db = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"Error loading database: {e}")
        print(f"Did you run 'python3 ingest.py' first?")
        sys.exit(1)

    retriever = db.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(TEMPLATE_STRING)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Answer:")
    try:
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\nAn error occurred: {e}")

    print()

if __name__ == "__main__":
    main()