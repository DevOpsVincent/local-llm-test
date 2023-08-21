import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from typing import List

# lädt PDF
file_path = "PDFs/fi-magazin-2-2023-neu-64a433fd8b27d709701238.pdf"
pdf_text = ""
with open(file_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

# Text cleanen
clean_pdf = pdf_text.replace("\n", "")

# Größere Textabschnitte in Chunks speichern, um sie effizienter weiterverarbeiten zu können
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

chunks = text_splitter.split_text(clean_pdf)

#Text in Embeddings umwandeln
class Embed:
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.transformer = SentenceTransformer(model_name, device="cpu")

    def __call__(self, text_batch: List[str]):
        embeddings = self.transformer.encode(
            text_batch,
            batch_size=100,
            device="cpu",
        ).tolist()

        return list(zip(text_batch, embeddings))

embedder = Embed()
embeddings_data = embedder(chunks)

# Erstellt embeddings
vector_store = FAISS.from_embeddings(
    embeddings_data,
    embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
)

# speichert embeddings Lokal
vector_store.save_local("faiss_index_chunks")
