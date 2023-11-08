import streamlit as st
import os

path = 'D:\Internship_PearlThoughts\Chats'

# Initialize chat_number using st.session_state
if 'chat_number' not in st.session_state:
    st.session_state.chat_number = 1

chat_file = os.path.join(path, f'chat_number_1.txt')
with open(chat_file, 'w') as fp:
    pass
    
# Function to create a new chat
def create_pdf():
    st.session_state.chat_number += 1
    chat_file = os.path.join(path, f'chat_number_{st.session_state.chat_number}.txt')
    with open(chat_file, 'w') as fp:
        st.write(f'New Conversation Started')

# Function to clear all chat files
def clear_chats():
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            st.write(f'Error deleting {file_path}: {e}')

if st.button("New Chat"):
    create_pdf()

if st.button("Clear Chats"):
    clear_chats()
    st.write("All chat files have been cleared.")
