from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



async def run_async_joke():
    llm = ChatOllama(model="llama3")
    prompt = ChatPromptTemplate.from_template(
        "Tell me a shore jock about {topic}"
    )

    chain = prompt | llm | StrOutputParser()

    # print(chain.invoke({"topic":"Space travel"}))
    topic = {"topic":"Space travel"}

    async for chunks in chain.astream(topic):
        print(chunks)

import asyncio
asyncio.run(run_async_joke())