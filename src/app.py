import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_response(user_input):
    return "I dont know"

def get_vectorStrore_from_url(url):
    # load the html text from the document and split it into chunks
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    return document_chunks


# streamlit app config
#
st.set_page_config(page_title="Website Chat")
st.title("Webiste Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# sidebar setup
with st.sidebar:
    st.header("Setting")
    website_url = st.text_input("website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    # check documents in the sidbar
    document_chunks = get_vectorStrore_from_url(website_url)
    with st.sidebar:
        st.write(document_chunks)

    # user input
    user_query = st.chat_input("Type here...")
    if user_query is not None and user_query != "":

        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)