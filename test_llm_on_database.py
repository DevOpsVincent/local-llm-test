from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import os

os.environ["API_KEY"] = "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"

model_id = "tiiuae/falcon-7b-instruct"

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ["API_KEY"],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.2,"max_new_tokens":500})

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer kind and in a logical way.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#lädt PDF
loader = PyPDFLoader("PDFs/fi-magazin-2-2023-neu-64a433fd8b27d709701238.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    length_function = len,
    is_separator_regex = False
)
texts = text_splitter.split_documents(pages)


chain_type_kwargs = {"prompt": PROMPT}

embeddings = FAISS.from_documents(
    texts,
    embedding=HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
)

qa = RetrievalQA.from_chain_type(llm=falcon_llm,
                                 chain_type="stuff",
                                 retriever=embeddings.as_retriever(),
                                 chain_type_kwargs=chain_type_kwargs
                                 )

query = "Warum ist Nachhaltigkeit wichtig für die Zukunft der Finanzinformatik?"
print(qa.run(query))