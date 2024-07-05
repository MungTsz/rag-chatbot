import os
import streamlit as st
from dotenv import load_dotenv
from modules.get_chat_model import get_cohere_model
from modules.full_chain import create_full_chain, ask_question
from modules.prompt import qa_system_prompt, contextualize_q_system_prompt
from modules.vectorstore import (
    get_cohere_embedding_model,
    create_chroma_vector_db,
)
from modules.splitter import split_documents
from modules.remote_loader import load_web_page
from langchain.globals import set_verbose
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)


st_callback = StreamlitCallbackHandler(st.container())

# Set global verbosity to verbose
set_verbose(True)

# initialize assistant chat history
if "responses" not in st.session_state:
    st.session_state.responses = []

# initialize user chat history
if "questions" not in st.session_state:
    st.session_state.questions = []

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_model = get_cohere_model()
web_loader = load_web_page("https://www.mql5.com/en/blogs/post/752096", "content")
docs = web_loader.load()
chunks = split_documents(docs, 1000, 300)
cohere_embeddings = get_cohere_embedding_model(cohere_api_key, chunks)
database = create_chroma_vector_db(chunks, cohere_embeddings)
retriever = database.as_retriever()
chain = create_full_chain(
    cohere_model,
    retriever,
    qa_system_prompt,
    contextualize_q_system_prompt,
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
