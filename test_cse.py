from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

#lädt PDF
loader = PyPDFLoader("PDFs/fi-magazin-2-2023-neu-64a433fd8b27d709701238.pdf")
pages = loader.load_and_split()

#Erstellt embeddings
faiss_index = FAISS.from_documents(
    pages,
    embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
)

#speichert embeddings Lokal
faiss_index.save_local("faiss_index")

