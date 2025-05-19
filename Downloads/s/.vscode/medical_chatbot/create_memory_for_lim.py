import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(documents, embedding_model, save_path):
    text_chunks = create_chunks(documents)
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(save_path)
    print(f"✅ FAISS vectorstore saved at: {save_path}")

def main():
    if not os.path.exists(DB_FAISS_PATH):
        print("🔵 Building FAISS vectorstore...")
        documents = load_pdf_files(DATA_PATH)
        embedding_model = get_embedding_model()
        build_vectorstore(documents, embedding_model, DB_FAISS_PATH)
    else:
        print("✅ FAISS already exists. No need to rebuild.")

if __name__ == "__main__":
    main()
