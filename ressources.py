from typing import List
import PyPDF2
import itertools
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

file_path = "PDFs/fi-magazin-2-2023-neu-64a433fd8b27d709701238.pdf"

def split_text(page_text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    split_text = text_splitter.split_text(page_text)
    return [text.replace("\n", " ") for text in split_text]

class Embed:
    #Text in Vektoren umwandeln
    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.transformer = SentenceTransformer(model_name, device="cpu")

    def __call__(self, text_batch: List[str]):
        embeddings = self.transformer.encode(
            text_batch,
            batch_size=100,
            device="cpu",
        ).tolist()

        return list(zip(text_batch, embeddings))


# PDF-Datei öffnen und Text extrahieren
pdf_text = ""
with open(file_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

# Text aufteilen in Chunks, um sie besser weiterverarbeiten zu
split_data = list(itertools.chain.from_iterable(map(split_text, [pdf_text])))

embedder = Embed()
embeddings_data = embedder(split_data)

"""
#print der Chunks und deren Vektoren
for text, embedding in embeddings_data:
    print(f"Text: {text}")
    print(f"Embedding: {embedding}")
"""
#Clustert ähnliche Vektoren in die Nähe durch faiss-index, um sie weiterzuverarbeiten
vector_store = FAISS.from_embeddings(
    embeddings_data,
    embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
)
#Speichern des faiss_index
vector_store.save_local("faiss_index")

