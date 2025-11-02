from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
#HuggingFacePipeline is used when we are using local models that we download from huggging face and run locally
import os
#It will download model from Hugging Face and run locally using transformers library
#Use only when we have gpus available otherwise it will be very slow on cpus

#By deafult c drive-> you can set it to any directory where you have enough space to store models   
# os.environ['HF_HOME'] = "D:/huggingface"  # Setting huggingface cache directory
llm = HuggingFacePipeline(
    model_id="MiniMaxAI/MiniMax-M2",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=100,
    )

)
model = ChatHuggingFace(
    llm = llm
)

result = model.invoke("What is the capital of India?")
print(result.content)