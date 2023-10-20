import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
# For reading PDF files
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
# To split
from langchain.text_splitter import RecursiveCharacterTextSplitter
# To generate embedding
from langchain.embeddings.openai import OpenAIEmbeddings
# VectorStores
from langchain.vectorstores import FAISS
import os
# LLM of OpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_lottie import st_lottie
import os
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

chat_icon = load_lottieurl(
    "https://lottie.host/b79f5040-0c8c-4c49-aed9-8a960bd22491/MeadGLI6fO.json")

with st.sidebar:
    st.title('💬LLM (Large Language Models) Chat App')
    st.markdown('''
    ## These are some Links
    - [GitHub Repository](https://github.com/teshchaudhary/Chat-PDF-App/tree/main)     
    - [GitHub](https://github.com/teshchaudhary)
    - [LinkedIn](https://www.linkedin.com/in/tesh-chaudhary/)
    ''')
    add_vertical_space(5)
    # Works as a print function of Python
    st.write('Made by Tesh Chaudhary')

load_dotenv()
def main():
    with st.container():
        left_column, centre_column, right_column = st.columns(3)
        with left_column:
            st_lottie(chat_icon, height = 100)

        with centre_column:
            st.header("ChatPDF App")
            st.write(" ")
            st.write(" ")

        with right_column:
            st.write(' ')
    
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')
    # Will give error to seek for a pdf when not uploaded 
    # pdf_reader = PdfReader(pdf)

    if pdf:
        # Reading PDF
        pdf_reader = PdfReader(pdf)
        whole_text = ""

        # Data Extraction
        for page in pdf_reader.pages:
            whole_text += page.extract_text()
        
        st.write(whole_text)

        # Now we need to split our data in smaller chunks because a limited amount of tokenization is available for the LLMs

        text_splitter = RecursiveCharacterTextSplitter(
            # Splitting into chunks of 1000 tokens at a time
            chunk_size = 1000,

            # This is to keep the context
            # Overlap of 200 tokens
            chunk_overlap = 300,
            length_function = len   
            )

        chunks = text_splitter.split_text(text = whole_text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        # Using Pickle to load if a vector store is already present
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            # Now we need to convert these texts to embeddings to make it machine understandable
            # embeddings
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
            # Using Pickle to save a vectorstore
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Check My Knowledge:")
        # st.write(query)
 
        if query:
            # To check the similarity between the vector stored embeddined texts and the query we have asked
            # k is representing how many references are there for the same query in out document.
            docs = VectorStore.similarity_search(query = query, k = 3)
            llm = OpenAI(model_name = 'gpt-3.5-turbo')
            chain = load_qa_chain(llm = llm, chain_type = "stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)

            st.write(response)
if __name__ == '__main__':
    main()
