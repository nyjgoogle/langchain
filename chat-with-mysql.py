from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
import streamlit as st


# streamlit run app.py --browser.gatherUsageStats false

# def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
#   db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
#   return SQLDatabase.from_uri(db_uri)

def init_database(user:str,password:str,host:str,port:str,database:str)-> SQLDatabase:
    db_url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_url)

def get_sql_chain(db):
    template ="""
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatOpenAI(model="gpt-3.5-turbo")
    # llm = ChatGroq(model="llama3",temperature=0) 
    llm = ChatOllama(model="llama3", temperature=0)


    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query:str,db:SQLDatabase,chat_history:list):
    sql_chain = get_sql_chain(db)

    template = """"
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        SQL Query:<SQL>{query}</SQL>
        user question: {question}
        SQL Response: {response}"""
    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatOpenAI(model="gpt-3.5-turbo")
    # llm = ChatGroq(model="Mixtral-8x7b-32768",temperature=0)
    llm = ChatOllama(model="llama3", temperature=0)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars['query']),
            # response=lambda vars: print("variable: ",vars)
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
       "question": user_query,
       "chat_history": chat_history, 
    })


    

if "chat_history" not in st.session_state:
    st.session_state.chat_history =[
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database.")
    ]                                                              

if "db" not in st.session_state:
    st.session_state.db = None

load_dotenv()

st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title("Chat with MySQL")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    host = st.text_input("Host", value="localhost",key="Host")
    port = st.text_input("Port", value="3306",key="Port")
    user = st.text_input("User", value="root",key="User")
    password = st.text_input("Password", type="password", value="root",key="Password")
    database = st.text_input("Database", value="Chinook",key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(user,password,host,port,database)
            st.session_state.db = db
            st.success("Connected to database!")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # sql_chain = get_sql_chain(st.session_state.db)

        # response = sql_chain.invoke({
        #     "chat_history":st.session_state.chat_history,
        #     "question":user_query
        # })
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
    
    st.session_state.chat_history.append(AIMessage(content=response))
