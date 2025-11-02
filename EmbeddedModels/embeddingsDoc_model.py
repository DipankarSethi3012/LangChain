from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536, chunk_size=100)


documents = [
    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity.",
    "It introduced the concepts of special relativity and general relativity, fundamentally changing physics.",
    "Special relativity deals with objects moving at constant speeds, while general relativity addresses gravity's effect on the fabric of spacetime."
    "here are two main parts to the theory: special relativity and general relativity.",
    "Special relativity focuses on the relationship between space and time for objects moving at constant speeds, particularly those approaching the speed of light.",
    "General relativity, on the other hand, explains how gravity affects the fabric of spacetime, leading to phenomena such as black holes and the bending of light around massive objects."
]
result = embeddings.embed_documents(documents)
print(result)