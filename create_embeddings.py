from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader("PDFs/fi-magazin-2-2023-neu-64a433fd8b27d709701238.pdf")
pages = loader.load_and_split()

faiss_index = FAISS.from_documents(
    pages,
    embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
)

question = input("Hallo, stelle mir eine Frage: ")

docs = faiss_index.similarity_search(question, k=2)

for doc in docs:
    print(str(doc.metadata["page"])+":", doc.page_content[:200])
