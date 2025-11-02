from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "LangChain is a framework for developing applications powered by language models."
result = embeddings.embed_query(text)
print(str(result))