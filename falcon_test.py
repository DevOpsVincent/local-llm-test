#import test_easy_embeddings
import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

os.environ["API_KEY"] = "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"

model_id = "tiiuae/falcon-7b-instruct"

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=os.environ["API_KEY"],
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})

prompt_template = """

You are an AI assistant that provides helpful, kind and logical answers to user queries.

{question}

"""

prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

falcon_chain = LLMChain(llm=falcon_llm,
                        prompt=prompt,
                        verbose=True)

print(falcon_chain.run("Wie viele Banken gibt es auf der Welt?"))
