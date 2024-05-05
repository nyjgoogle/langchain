from dotenv import load_dotenv
import os

load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")






from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3")

# import ollama
# ollama.embeddings(model='llama3'),    

import ollama

ollama.embeddings(
    model='llama3',
    prompt ="Llama are member of the camelid family"
    )