import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.add_vertical_space import add_vertical_space

# For reading PDF files
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
# To split
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
# To generate embedding
from langchain.embeddings import OpenAIEmbeddings

# VectorStores
from langchain.vectorstores import FAISS

# LLMs use
from langchain.llms import OpenAIChat
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
import requests

#Chat elements
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with  st.chat_message("assistant"):
            st.markdown(message["content"])
# Sidebar contents
with st.sidebar:
    st.title('💬LLM (Large Language Models) Chat App')
    st.markdown('''
    ## These are some Links
    - [GitHub Repository](https://github.com/teshchaudhary/Chat-PDF-App)     
    - [GitHub](https://github.com/teshchaudhary)
    - [LinkedIn](https://www.linkedin.com/in/tesh-chaudhary/)
    ''')
    add_vertical_space(5)
    # Works as a print function of Python
    st.write('Made by Tesh Chaudhary')

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

chat_icon = load_lottieurl(
    "https://lottie.host/b79f5040-0c8c-4c49-aed9-8a960bd22491/MeadGLI6fO.json")

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

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type = 'pdf')
    st.button("Submit")

    if pdf is not None:
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

        # reading the file name
        store_name = pdf.name[:-4]
       
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)


        query = st.text_input("What do you want to know?:")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if query:
            chat_history = []
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in English language.' If you do not know the answer reply with 'I am sorry I don't know.'
                                   Chat History:
                                   {chat_history}
                                   Follow Up Input: {question}
                                   Standalone question:
                                   Remember to greet the user with hi welcome to pdf chatbot how can i help you? if user asks hi or hello """

            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

            llm = OpenAIChat()

            conversation_chain =ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                memory=memory
            )
            response = conversation_chain({"question": query, "chat_history": chat_history})

            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            chat_history.append((query, response))

if __name__ == '__main__':
    main()
