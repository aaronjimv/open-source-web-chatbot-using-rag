import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_community.llms import Ollama


def get_response(user_input):
    return "I dont know"

def get_vectorStrore_from_url(url):
    # load the html text from the document and split it into chunks
    #
    # store the chunk in a vectore store
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    vectore_store = Chroma.from_documents(document_chunks, HuggingFaceHubEmbeddings())

    return vectore_store

def get_context_retriever_chain(vector_store):
    # set up the llm, retriver and prompt to the retiver the chain
    #
    llm = Ollama(model="phi")

    retriver = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriver_chain = create_history_aware_retriever(
        llm, 
        retriver, 
        prompt
    )

    return retriver_chain

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
    vector_store = get_vectorStrore_from_url(website_url)

    retriver_chain = get_context_retriever_chain(vector_store)

    # user input
    user_query = st.chat_input("Type here...")
    if user_query is not None and user_query != "":

        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        retriver_documents = retriver_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
        st.write(retriver_documents)

    # conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
