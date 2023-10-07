import streamlit as st
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle



def load_llm():
    llm = CTransformers(model="models\llama-2-7b-chat.ggmlv3.q2_K.bin",
                    model_type='llama',
                    config={'max_new_tokens':128,
                            'temperature':0})
    return llm


def load_vector_store():
    with open('vector_store.pkl', 'rb') as pickle_file:
        loaded_vector_store = pickle.load(pickle_file)
    return loaded_vector_store

def load_qa_prompt():
    with open('qa_prompt.pkl', 'rb') as pickle_file:
        loaded_qa_prompt = pickle.load(pickle_file)
    return loaded_qa_prompt

llm = load_llm()

print('Model Loaded')
vector_store = load_vector_store()
print('Vector store Loaded')

qa_prompt = load_qa_prompt()
print('Prompt loaded')

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vector_store.as_retriever(search_kwargs={'k':2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt':qa_prompt})


# input query here
query = "What datasets are used in the study? How large are these datasets?"


# response elements
response = chain({'query': query})
result = response['result']
source_documents = response['source_documents']

print(response)