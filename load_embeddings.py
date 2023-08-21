from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")

# lädt gespeicherte Embeddings
new_db = FAISS.load_local("faiss_index", embedding)


# Test Ähnlichkeitssuche in den Embeddings
question = input("Hallo, stelle mir eine Frage: ")

#print(new_db.similarity_search(question, k=2, ))

# Mit Seitenzahl
docs = new_db.similarity_search(question, k=2)

# Anschaulichere Darstellung mit Seitenzahl: & Limit an Wörtern
for doc in docs:
    print(str(doc.metadata["page"])+":", doc.page_content[:500])
