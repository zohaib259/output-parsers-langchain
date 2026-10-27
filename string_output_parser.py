import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


template1 = PromptTemplate(
    template="write a detailed report on {topic}", input_variables=["topic"]
)

template2 = PromptTemplate(
    template="write a 5 lines summary on {text}", input_variables=["topic"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"black hole"})
print(result)