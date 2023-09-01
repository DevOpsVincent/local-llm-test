from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader("PDFs/fi-magazin-2-2023-neu-64a433fd8b27d709701238.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function = len,
    is_separator_regex = False
)

texts = text_splitter.split_documents(pages)

embeddings = FAISS.from_documents(
    texts,
    embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
)

query = "Warum ist Nachhaltigkeit im Banking so wichtig?"
docs = embeddings.similarity_search(query)

print(docs[0])