from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


#default value of temperature is 0.7

#Creating an instance of ChatOpenAI LLM
model = ChatOpenAI(model_name='gpt-4o', temperature=0.2, max_completion_tokens=100)
#max_completion_tokens -> limits the response length from the model (When we use a paid model we are charged based on the number of tokens used in the prompt + response)

#Tokens are chunks of words. 1 token is roughly equal to 4 characters in English. For example, the word "fantastic" would be split into tokens like "fan", "tas", and "tic".

#With the help of Invoke function we can interact with the LLM
result = model.invoke("Explain the langchain?")

#The result of chat model is of type ChatResult and it contains additional metadata
print(result)

#the actual content can be accessed using .content attribute
print(result.content)  # Accessing the content of the response


#when to use what temperature? value?
#0 - 0.3 -> factual answers (facts, maths, code etc)
#0.4 - 0.7 -> General Tasks (Conversation, General Q&A etc)
#0.8 - 1.0 -> Creative Tasks (Storytelling, Poetry, etc)
#Higher the temperature more creative/wild the responses would be brain storming, new ideas etc