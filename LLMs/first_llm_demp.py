from langchain_openai import OpenAI
from dotenv import load_dotenv

#loading the environment variables from .env file
load_dotenv()

#Creating an instance of OpenAI LLM
llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

#With the help of Invoke function we can interact with the LLM
result = llm.invoke("What is LangChain?")
#The result of LLM is of type LLMResult and it does not contain additional metadata
print(result)