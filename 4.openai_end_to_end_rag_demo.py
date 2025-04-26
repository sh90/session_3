#pip install tiktoken openai
# replace your api key on line number 18
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
import openai
# --------- Config ---------
import data_info

TEXT_FILE_PATH = "data/onboarding.txt"
CHROMA_PERSIST_DIR = "chroma_store_openai"

LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"
OPENAI_API_KEY = data_info.open_ai_key # Replace this with your open ai key "SK-"
# --------- Load and Split Text ---------
print("[INFO] Loading and splitting text...")
with open(TEXT_FILE_PATH, "r", encoding="utf-8") as f:
    text = f.read()

raw_docs = [Document(page_content=text)]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

# --------- Initialize Embeddings ---------
EMBED_MODEL = "text-embedding-3-small"
print(f"[INFO] Using OpenAI Embeddings: {EMBED_MODEL}")
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL,openai_api_key=OPENAI_API_KEY)

# --------- Create or Load ChromaDB ---------
if os.path.exists(CHROMA_PERSIST_DIR):
    print("[INFO] Loading existing Chroma vector store...")
    vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)
else:
    print("[INFO] Creating Chroma vector store and embedding documents...")
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PERSIST_DIR)
    vectordb.persist()

# --------- Initialize LLM ---------
print(f"[INFO] Using OpenAI LLM: {LLM_MODEL}")
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0,openai_api_key=OPENAI_API_KEY)

# --------- Create RetrievalQA Chain ---------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# --------- Q&A Loop ---------
print("\n[READY] Ask me anything about the content in your file. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa_chain(query)
    print("\nAI:", result["result"], "\n")
