from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 1. Define the desired output structure using Pydantic
class BlackHoleFacts(BaseModel):
    fact_1: str = Field(description="fact 1 about the topic")
    fact_2: str = Field(description="fact 2 about the topic")
    fact_3: str = Field(description="fact 3 about the topic")

parser = PydanticOutputParser(pydantic_object=BlackHoleFacts)

template = PromptTemplate(
    template="Give 3 facts about the {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }, 
)

chain = template | model | parser

result = chain.invoke({"topic": "black hole"})
print(result)