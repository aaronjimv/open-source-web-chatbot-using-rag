import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings


def get_vectorStrore_from_url(url):
    # load the html text from the document and split it into chunks
    #
    # store the chunk in a vectore store
    #
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) # To do: test performance
    document_chunks = text_splitter.split_documents(document)

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectore_store = Chroma.from_documents(document_chunks, embeddings)

    return vectore_store

def get_context_retriever_chain(vector_store):
    # set up the llm, retriver and prompt to the retriver_chain
    #
    # retriver_chain -> retrieve relevant information from the database
    #
    llm = Ollama(model='phi3') # "or any other model that you have"

    retriver = vector_store.as_retriever(k=2) # To do: test `k`

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get the information relevant to the conversation")
        ]
    )

    retriver_chain = create_history_aware_retriever(
        llm, 
        retriver, 
        prompt
    )

    return retriver_chain

def get_conversation_rag_chain(retriever_chain):
    # summarize the contents of the context obtained from the webpage
    #
    # based on context generate the answer of the question
    #
    llm = Ollama(model='phi3') # "or any other model that you have"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_document_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain, stuff_document_chain)

def get_response(user_input):
    #  invokes the chains created to generate a response to a given user query
    #
    retriver_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retriver_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']


# streamlit app config
#
st.set_page_config(page_title="Lets chat with a Website", page_icon="💻")
st.title("Lets chat with a Website")

# sidebar setup
with st.sidebar:
    st.header("Setting")
    website_url = st.text_input("Type the URL here")

if website_url is None or website_url == "":
    st.info("Please enter a website URL...")

else:
    # Session State
    #
    # Check the chat history for follow the conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    # Check if there are already info stored in the vectorDB
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorStrore_from_url(website_url)
    
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
