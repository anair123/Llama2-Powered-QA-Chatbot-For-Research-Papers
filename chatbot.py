from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

# prepare the template we will use when prompting the AI
template = """Use the provided context to answer the user's question.
If you don't know the answer, respond with "I do not know".

Context: {context}
Question: {question}
Answer:
"""

# load the language model
llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q4_1.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0})

# load the interpreted information from the local database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
db = FAISS.load_local("faiss", embeddings)

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

# ask the AI chat about information in our local files
prompt = "Give me examples of sports that incorporate computer vision"

# What is the benefit of computer vision in sports analysis?
output = qa_llm({'query': prompt})
print(output["result"])