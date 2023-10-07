import streamlit as st
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from glob import glob
import pickle

@st.cache_resource()
def load_llm():
    llm = CTransformers(model="models\llama-2-7b-chat.ggmlv3.q2_K.bin",
                    model_type='llama',
                    config={'temperature':0})
    return llm

@st.cache_resource()
def create_vector_store(folder_path:str):
    # load the pdf files from the data folder

    pdf_files = glob(f'{folder_path}/*.pdf')
    text = ""

    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                chunk_overlap=100)

    text_chunks = text_splitter.split_text(text)

    print(len(text_chunks))

    # create the embedding model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device':'cpu'})


    # convert the chunks into embeddings and store them in a vector store
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    return vector_store

@st.cache_data()
def create_qa_prompt():
    
    template = """Use the provided information to answer the user's query. 
                    The reesponse should be have at most 2 sentences. 

    Context: {context}
    Question: {question}
    """

    qa_prompt = PromptTemplate(template=template,
                            input_variables=['context', 'question'])
    return qa_prompt

def generate_response(llm, vector_store, qa_prompt, query):
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=vector_store.as_retriever(search_kwargs={'k':3}),
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt':qa_prompt})

    # response elements
    response = chain({'query': query})
    result = response['result']
    source_documents = response['source_documents']
    return response

def main():

    st.title('Say Hello To Our Llama2-Powered Chatbot! Trained with Social Media Research!')


    vector_store = create_vector_store(folder_path='data')
    qa_prompt = create_qa_prompt()


    query = st.text_input('User Query', 'Enter Your Query Here')
    if st.button('Submit question'):
        response = generate_response(llm=llm,
                                     vector_store=vector_store,
                                     qa_prompt=qa_prompt,
                                     query=query)
        
        st.write(f'Response: {response["result"]}')
        #st.write(f'Source: {response["source_documents"]}')

if __name__ == "__main__":

    llm=load_llm()
    vector_store = create_vector_store(folder_path='data')
    qa_prompt = create_qa_prompt()

    query = "How does social media affect fear speech?"
    response = generate_response(llm, vector_store, qa_prompt, query)

    print(response)
    
    main()