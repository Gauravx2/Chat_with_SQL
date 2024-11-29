import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from sqlalchemy import create_engine
import sqlite3

# Streamlit setup
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# Database selection
radio_opt = ["Use SQLite 3 Database - Student.db", "Connect to MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB to chat with", options=radio_opt)

# Get database details
if selected_opt == "Connect to MySQL Database":
    mysql_host = st.sidebar.text_input("MySQL Host", placeholder="e.g., localhost")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_filepath = (Path(__file__).parent / "student.db").absolute()

# Get Groq API Key
api_key = st.sidebar.text_input("Groq API Key", type="password")
if not api_key:
    st.info("Please provide the Groq API Key.")
    st.stop()

# Configure LLM
from langchain_groq import ChatGroq
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")

# Database connection
def configure_db(selected_opt):
    if selected_opt == "Connect to MySQL Database":
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please fill all MySQL connection details.")
            st.stop()
        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            )
        )
    else:
        if not db_filepath.exists():
            st.error(f"SQLite database file not found: {db_filepath}")
            st.stop()
        return SQLDatabase(
            create_engine(f"sqlite:///{db_filepath}")
        )

db = configure_db(selected_opt)

# Initialize the toolkit
toolkit = SQLDatabaseToolkit(llm=llm, db=db)

# Create the SQL agent
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Chat Interface
if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask a question about the database...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        try:
            response = agent.run(user_query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            error_message = f"Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.write(error_message)
