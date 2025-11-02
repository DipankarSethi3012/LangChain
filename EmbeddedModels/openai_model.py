from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


#Dimesnions and chunk_size are optional parameters
#Dimesnions is used to specify the size of the embedding vector
#chunk_size is used to specify the number of tokens to be processed at a time
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions= 45, chunk_size=1)

result = embeddings.embed_query("What is LangChain?")
print(result)
print(str(result))


#queries = ["What is LangChain?", "Who made it?", "How does it work?"]
# chunk_size=1 → har query alag call me jayegi (3 API calls)
# chunk_size=3 → ek hi API call me sab 3 queries jayengi
#Chunks is useed to split the queries into smaller parts to avoid exceeding the token limit of the model