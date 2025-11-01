#ChatHugging face model free access
from langchain_huggingface import ChatHuggingFace
#As we are using HuggingFace inference endpoint service so we have to import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

#Define the HuggingFace endpoint with the desired model, that can be accessed for free and used for chat when we pass task as text-generation
llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2",
    task="text-generation",
)

#Creting the ChatHuggingFace model, passing the llm defined above because HuggingFaceEndpoint supports text generation task and we have to tell the ChatHuggingFace model to use that llm for chat
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result)
print(result.content)
