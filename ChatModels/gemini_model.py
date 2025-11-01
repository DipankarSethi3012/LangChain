from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.3, max_tokens=100)

result = model.invoke("Explain the langchain?")
print(result)
