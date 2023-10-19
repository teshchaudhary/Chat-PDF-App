import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
# For reading PDF files
from PyPDF2 import PdfReader
# To split
from langchain.text_splitter import RecursiveCharacterTextSplitter
# To generate embedding
from langchain.embeddings.openai import OpenAIEmbeddings
# VectorStores
from langchain.vectorstores import FAISS

with st.sidebar:
    st.title('💬LLM(Large Language Models) Chat App')
    st.markdown('''
    ## These are some Links
    - [GitHub](https://github.com/teshchaudhary)
    - [LinkedIn](https://www.linkedin.com/in/tesh-chaudhary/)
    ''')
    add_vertical_space(5)
    # Works as a print function of Python
    st.write('Made by Tesh Chaudhary')

def main():
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

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

        # Now we need to convert these texts to embeddings to make it machine understandable
        embeddings = OpenAIEmbeddings() # An object


if __name__ == '__main__':
    main()