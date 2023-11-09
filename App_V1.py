import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

import os
import pickle
import requests
from dotenv import load_dotenv

def open_link(url):
    import webbrowser
    webbrowser.open_new_tab(url)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return "Connection Failed"
    return r.json()

chat_icon = load_lottieurl("https://lottie.host/b79f5040-0c8c-4c49-aed9-8a960bd22491/MeadGLI6fO.json")

pdf_icon = load_lottieurl('https://lottie.host/4e044377-1b9b-42e0-8241-f6c3976558a4/CaycxG2x1G.json')

# Set page configuration
st.set_page_config(
    page_title = "ChatPDF App",
    page_icon = "💬",
    layout = "wide",
)

# Set sidebar
with st.sidebar:
    # Create a horizontal layout for "Chat-PDF" text and PDF icon
    col1, col2 = st.columns([4,1])  # Adjust the width ratios as needed

    with col1:
        st.markdown("<h1 style='font-size:50px;'>Chat - PDF</h1>", unsafe_allow_html=True)
        
    with col2:
        st_lottie(pdf_icon, height=99)

    st.markdown('''
        ## Links
        ''')

    # Add buttons for the links
    linkedin_button = st.button("LinkedIn", key = "linkedin_button", on_click = lambda: open_link("https://www.linkedin.com/in/tesh-chaudhary/"))
    github_button = st.button("GitHub Profile", key = "github_button", on_click = lambda: open_link("https://github.com/teshchaudhary"))
    github_repo_button = st.button("GitHub Repository", key = "github_repo_button", on_click = lambda: open_link("https://github.com/teshchaudhary/Chat-PDF-App"))

    st.write('Made by Tesh Chaudhary')

# Load environment variables
load_dotenv()

def main():
    with st.container():
        left_column, centre_column, right_column = st.columns(3)
        with left_column:
            st_lottie(chat_icon, height = 100)

        with centre_column:
            st.title("Talk to your PDF")
            st.write(" ")
            st.write(" ")

        with right_column:
            st.write(" ")

    uploaded_file = st.file_uploader("Upload your PDF", type = 'pdf')

    if uploaded_file:
        with st.spinner("Processing..."):
            # Reading PDF
            pdf_reader = PdfReader(uploaded_file)
            raw_text = ""

            # Data Extraction
            for page in pdf_reader.pages:
                raw_text += page.extract_text()

            # Now we need to split our data into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 300,
                length_function = len
            )

            chunks = text_splitter.split_text(text = raw_text)

            store_name = uploaded_file.name[:-4]

            # Using Pickle to load if a vector store is already present
            # if os.path.exists(f"{store_name}.pkl"):
            #     with open(f"{store_name}.pkl", "rb") as f:
            #         VectorStore = pickle.load(f)
            # else:
            # Now we need to convert these texts to embeddings
            # embeddings = OpenAIEmbeddings()
            embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Display success message after processing is complete
        st.success("Processing completed!")

        # Accept user questions/query when form is submitted
        with st.form(key = 'query_form'):
            query = st.text_input("Enter your query:")
            submit_button = st.form_submit_button("Submit Query")

            if submit_button and query:
                with st.spinner("Searching for answers..."):
                    # To check the similarity between the vector stored embeddings and the query
                    docs = VectorStore.similarity_search(query = query, k=3)
                    # llm = OpenAI(model_name = 'gpt-3.5-turbo')
                    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
                    chain = load_qa_chain(llm = llm, chain_type = "stuff")
                    response = chain.run(input_documents = docs, question = query)
                st.write(response)
            
if __name__ == '__main__':
    main()
