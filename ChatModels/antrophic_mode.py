from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model_name ='claude-3.5-sonnet-20241002', temperature=0.3, max_tokens=10)

result = model.invoke("Explain the langchain?")
print(result)
print(result.content)  # Accessing the content of the response