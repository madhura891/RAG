import streamlit as st
from langchain.schema import Document
import PyPDF2
groqapi = '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'

from langchain.text_splitter import RecursiveCharacterTextSplitter

uploaded_file = r"C:\Users\Admin\Documents\EDA Python.pdf"

text=""
pdf_reader = PyPDF2.PdfReader(uploaded_file)
for page in pdf_reader.pages:
    text += page.extract_text() + "\n"

from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
text_chunks=splitter.split_text(text)
docs = [Document(page_content=chunk)for chunk in text_chunks]
st.subheader('Document Splitted Succesfully')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")     
vectordb = FAISS.from_documents(docs,embeddings)     
st.success("FAISS Vectorstore created succesfully")
retriver = vectordb.as_retriever()
      
from langchain.chat_models import init_chat_model   
model = init_chat_model(model='gemma2-9b-it',model_provider='groq',api_key=groqapi) 
from langchain.prompts import PromptTemplate
template="""
you are a helpful assistance. Answer the question using only the context below.
If the answer is not present, just say no. Do not try to make up an answer.

Contex:
{context}

Question:
{question}

Helpful Answer:
"""    

rag_prompt = PromptTemplate(input_variables=["context", "question"],template=template)

user_query = st.text_input("ask a question about the PDF")

if user_query:
    relevant_docs = retriver.invoke(user_query) 
    
    final_prompt = rag_prompt.format(context=relevant_docs, question=user_query)
    
    with st.spinner("Generating answer..."):
        response = model.invoke(final_prompt)
        
    st.write('### Answer')
    st.write(response.content)

                                 