from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os

embedding = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")

# lädt gespeicherte Embeddings
db = FAISS.load_local("faiss_index", embedding)

os.environ["API_KEY"] = "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"

model_id = "tiiuae/falcon-7b"

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ["API_KEY"],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.5,"max_new_tokens":300})

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Context: You are an assistant for one of the biggest german banking IT-service providers and asked to answer the questions based on the data input the company gave you. Do not repeat yourself in your answers and give concise answers.

Question: {question}
Answer in German:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=falcon_llm,
                                 chain_type="stuff",
                                 retriever=db.as_retriever(),
                                chain_type_kwargs=chain_type_kwargs
                                 )

query = "Welche neue Funktionen bietet das neue OsPlus für die Sparkassen?"
print(qa.run(query))