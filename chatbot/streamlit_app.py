import os
from dotenv import load_dotenv
import streamlit as st
from modules.get_chat_model import get_cohere_model
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from modules.splitter import split_documents
from modules.vectorstore import (
    get_cohere_embedding_model,
    create_chroma_vector_db,
)
from modules.prompt import qa_system_prompt, contextualize_q_system_prompt
from modules.full_chain import create_full_chain, ask_question
from modules.remote_loader import load_web_page


st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")


def show_ui(qa, prompt_to_user="How may I help you?"):
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, prompt)
                st.markdown(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


# @st.cache_resource
def get_retriever(model_key, chunks):
    cohere_embeddings = get_cohere_embedding_model(model_key, chunks)
    database = create_chroma_vector_db(chunks, cohere_embeddings)
    retriever = database.as_retriever()
    return retriever


def get_chain(retriever, llm, qa_system_prompt, contextualize_q_system_prompt):
    chain = create_full_chain(
        llm,
        retriever,
        qa_system_prompt,
        contextualize_q_system_prompt,
        chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    )
    return chain


def run():
    ready = True
    # Load the environment variables from the .env file
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    web_loader = load_web_page("https://www.mql5.com/en/blogs/post/752096", "content")
    docs = web_loader.load()
    chunks = split_documents(docs, 1000, 300)
    retriever = get_retriever(cohere_api_key, chunks)
    cohere_model = get_cohere_model()
    if ready:
        chain = get_chain(
            retriever, cohere_model, qa_system_prompt, contextualize_q_system_prompt
        )
        st.subheader("Ask me questions about MQL5")
        show_ui(chain, "What would you like to know?")
    else:
        st.stop()


run()
