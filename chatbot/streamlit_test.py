import os
import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain.globals import set_verbose
from langchain.memory import ChatMessageHistory

from constants import URL_DICT_LIST
from modules.LLM_chain.create_LLM_chain import ask_question, create_full_chain
from modules.RAG.create_RAG import create_RAG_retriever

warnings.filterwarnings("ignore")

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# initialize assistant chat history
if "responses" not in st.session_state:
    st.session_state.responses = []

# initialize user chat history
if "questions" not in st.session_state:
    st.session_state.questions = []

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMessageHistory()

if "retriever" not in st.session_state:
    st.session_state.retriever = create_RAG_retriever(COHERE_API_KEY, URL_DICT_LIST)


chain = create_full_chain(
    st.session_state.retriever,
    st.session_state.chat_memory,
)

# set title
st.title("Forex Forest Assistant")

prompt_to_user = "Hi, human! Is there anything I can help you with?"
with st.chat_message("assistant"):
    st.markdown(prompt_to_user)
# React to user input and save to history
if query := st.chat_input("Say something"):

    for question, response in zip(
        st.session_state["questions"], st.session_state["responses"]
    ):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(response)

    with st.chat_message("user"):
        st.markdown(query)
    with st.spinner("wait a moment"):
        response = ask_question(chain, query)
    st.session_state.questions.append(query)

    st.session_state.responses.append(response)
    with st.chat_message("assistant"):
        st.markdown(response)


def clear_cache():
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]


st.sidebar.button("Clear Cache", on_click=clear_cache)
