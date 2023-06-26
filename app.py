"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Pinecone

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent

import pinecone

# Cohere embeddings
# small - 1024 dim
embeddings = CohereEmbeddings(model='small',
                              cohere_api_key=st.secrets['COHERE_API_KEY'])

# Pinecone Vectorstore
pinecone.init(
    api_key=st.secrets['PINECONE_API_KEY'],
    environment=st.secrets['PINECONE_ENVIRONMENT']
)

index_name = st.secrets['PINECONE_INDEX']

vectorstore = Pinecone.from_existing_index(index_name, embeddings)


def load_agent():
    llm = ChatOpenAI(
        openai_api_key=st.secrets['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.0,
    )

    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr")
    )

    tools = [
        Tool(
            name='Bangkok Airbnb',
            func=qa.run,
            description=(
                'use this tool when answering Airbnb or place to stay in Bangkok queries to get '
                'more information about the topic'
            )
        )
    ]

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=False,  # Only output
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

    sys_msg = """You are a helpful chatbot that try to find the Airbnb room using our tool 
    and try to answers the user's questions.
    """

    prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = prompt

    return agent


agent = load_agent()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="BKK Airbnb Demo", page_icon=":robot:")
st.header("BKK Airbnb Demo")
st.text("I'm here to help you find an Airbnb room in Bangkok! What is your preferences?")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input(
        "You: ", key="input")
    print(input_text)
    return input_text


user_input = get_text()

if user_input:
    output = agent.run(user_input)
    print(output)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i],
                is_user=True, key=str(i) + "_user")
