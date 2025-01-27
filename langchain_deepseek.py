import os
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class Multiply(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

llm = ChatOllama(
    model = "llama3.2",
    temperature = 0.0,
    num_predict = 256,
    # format="json",
    # other params ...
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)


if __name__ == "__main__":
    chain = prompt | llm | StrOutputParser()
    for chunk in chain.stream(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    ):
        print(chunk, end="|", flush=True)



