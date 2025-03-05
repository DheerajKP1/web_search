import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import chromadb
import uuid
load_dotenv()
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


# Set up LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="deepseek-r1-distill-llama-70b"
)

# Initialize embeddings model
# embedding_model = OllamaEmbeddings(model="llama3.2:1b")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Load Chroma vector store
# vectorstore = Chroma(persist_directory="chroma-BAAI", embedding_function=embedding_model)
# Streamlit UI
st.title("AI-Powered Question Answering Chatbot")
st.write("Ask a question based on the uploaded document.")
web_link = st.text_input("Enter a website URL:")
st.write(web_link)

if web_link:
    try:
        # Load and split webpage text
        loader = WebBaseLoader(web_link)
        page_data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        documents = text_splitter.split_documents(page_data)

        if not documents:
            st.error("⚠️ No content extracted from the webpage. Please try another URL.")
        else:
            # ✅ Store vectorstore in session state
            st.session_state.vectorstore = FAISS.from_documents(documents, embedding_model)
            # st.session_state.vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=None)
            st.success("✅ Webpage processed successfully! You can now ask questions.")
    
    except Exception as e:
        st.error(f"❌ Error loading webpage: {e}")
############# Part without memory storage (start)############# 
# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True
# )
# Retrieval-based QA
# # retriever = vectorstore.as_retriever()
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
#     memory=memory,
# )
###############         ends here         #################################

# memory = MemorySaver()
tools = []
tool_node = ToolNode(tools)
model = llm
bound_model = model.bind_tools(tools)

@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return "It is chatbot that helps user ask from the given web link."
def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise if there is, we continue
    return "action"
# Define the function that calls the model
def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": response}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Next, we pass in the path map - all the possible nodes this edge could go to
    ["action", END],
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable



if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
    vectorstore = st.session_state.vectorstore
else:
    vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()  # Store memory in session state
memory = st.session_state.memory
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "2"}}

query = st.text_input("Enter your question:")
results = vectorstore.similarity_search(query, k=2)
context = " ".join([doc.page_content for doc in results])
# prompt = query
prompt = f'''follow these instruction while genearating text there are three intruction go one by one-
            1. Based on the following information, answer the question:\n\n{context}\n\nQuestion: {query} .
            2. If you think query is not related to doc given to ,then please respond as per your knowledge but keep it consize.
            3. I have also Implemented memory using graph so can check it while answering
            '''

if query and vectorstore:
    with st.spinner("Thinking..."):
        # response = llm.invoke(prompt)
        # cleaned_response = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
        # disclaimer = "\n\n**Disclaimer:** Please verify this information, as it is generated by an AI and may contain errors."
        # full_response = cleaned_response.strip() + disclaimer
        # st.write("### Answer:")
        # st.write(full_response)
        # response = qa_chain.invoke({"question": query, "chat_history": st.session_state.chat_history})
        # cleaned_response = re.sub(r"<think>.*?</think>", "", response["answer"], flags=re.DOTALL)
        input_message = HumanMessage(content=f"{prompt}")
        for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
            cleaned_response = re.sub(r"<think>.*?</think>", "", event["messages"][-1].content, flags=re.DOTALL)
        # Extract the result
        

        follow_up_prompt = f'''just Generate 2-3 follow-up questions related to query nothing more: {query} please put it like would you like know more about this then follow up question but do not add any extra information.'''
        follow_up_response = llm.invoke(follow_up_prompt)
        follow_up_questions = follow_up_response
        follow_up_questions = re.sub(r"<think>.*?</think>", "", follow_up_questions.content, flags=re.DOTALL).split("\n")

        # Update chat history
        st.session_state.chat_history.append(HumanMessage(query))
        st.session_state.chat_history.append(AIMessage(cleaned_response))
        st.write("### Answer:")
        st.write(cleaned_response)
        if follow_up_questions:
            st.write("\n💡 **Follow-up Questions:**")
            for q in follow_up_questions:
                if q.strip():
                    st.write("-", q.strip())

# Display past conversation
if st.session_state.chat_history:
    st.write("\n\n### Chat History:")
    for message in st.session_state.chat_history:
        role = "👤 You:" if isinstance(message, HumanMessage) else "🤖 Bot:"
        st.write(f"{role} {message.content}")
