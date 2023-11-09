import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAIChat, HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

import os
import pickle
import requests
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def open_link(url):
    import webbrowser
    webbrowser.open_new_tab(url)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return "Connection Failed"
    return r.json()

# Chat elements
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

chat_icon = load_lottieurl("https://lottie.host/b79f5040-0c8c-4c49-aed9-8a960bd22491/MeadGLI6fO.json")

pdf_icon = load_lottieurl('https://lottie.host/4e044377-1b9b-42e0-8241-f6c3976558a4/CaycxG2x1G.json')

# Set page configuration
# st.set_page_config(
#     page_title = "ChatPDF App",
#     page_icon =  "💬",
#     layout = "wide",
# )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set sidebar
with st.sidebar:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("<h1 style='font-size:45px;'>Chat - PDF</h1>", unsafe_allow_html=True)
        pdf = st.file_uploader("Upload your PDF", type='pdf')
    with col2:
        st_lottie(pdf_icon, height=99)

    if pdf:
        with st.spinner("Processing..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            store_name = pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

        st.success("Processing completed!")

    st.markdown('''
    ## Links
    ''')
    linkedin_button = st.button("LinkedIn", key="linkedin_button", on_click=lambda: open_link("https://www.linkedin.com/in/tesh-chaudhary/"))
    github_button = st.button("GitHub Profile", key="github_button", on_click=lambda: open_link("https://github.com/teshchaudhary"))
    github_repo_button = st.button("GitHub Repository", key="github_repo_button", on_click=lambda: open_link("https://github.com/teshchaudhary/Chat-PDF-App"))
    st.write('Made by Tesh Chaudhary')

# Check if a PDF has been uploaded before displaying chat elements
if pdf:
    # chat_history = None
    # st.session_state.chat_history = chat_history
    query = st.text_input("Ask questions about your PDF file:")

    submit_button = st.button("Submit Query")

    if submit_button and query:
        with st.spinner("Searching for answers..."):
            chat_history = st.session_state.chat_history
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            custom_template = """
            Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. At the end of the standalone question, add this 'Answer the question in English language.' If you do not know the answer, reply with 'I am sorry I don't know.'
            Chat History:
            {chat_history}
            Follow-Up Input: {question}
            Standalone question:
            Remember to greet the user with 'Hi, welcome to the PDF chatbot. How can I help you?' if the user asks 'hi' or 'hello.'
            """

            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

            llm = OpenAIChat(model_name='gpt-3.5-turbo')

            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )
            response = conversation_chain({"question": query, "chat_history": chat_history})

            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            chat_history.append((query, response))
            st.session_state.chat_history = chat_history

else:
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

    st.error("Upload your PDF to start chatting.")
