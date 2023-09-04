from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import sys

# load the pdf files from the data folder
loader = DirectoryLoader('data/',
                         glob='*.pdf',
                         loader_cls=PyPDFLoader)

documents = loader.load()

# split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)

text_chunks = text_splitter.split_documents(documents)

print(len(text_chunks))

# create the embedding model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device':'cpu'})


# convert the chunks into embeddings and store them in a vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

# create an llm object
llm = CTransformers(model="models\llama-2-7b-chat.ggmlv3.q2_K.bin",
                    model_type='llama',
                    config={'max_new_tokens':128,
                            'temperature':0})

# crete the query
query = "Describe this document"

# perform similarity search
docs = vector_store.similarity_search(query)

template = """Use the provided information to answer the user's query. The reesponse should be have 5 or less sentences. 


Context: {context}
Question: {question}
"""

qa_prompt = PromptTemplate(template=template,
                           input_variables=['context', 'question'])


chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vector_store.as_retriever(search_kwargs={'k':2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt':qa_prompt})


# input query here
query = "What are the components of HOUYI?"


# response elements
response = chain({'query': query})
result = response['result']
source_documents = response['source_documents']


print(f"""
query: {query}\n
result: {result}\n
source documents: {source_documents}
""")