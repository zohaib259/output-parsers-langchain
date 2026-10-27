from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# 1. Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

# 2. Define schema (fixed typo: description)
class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(description="age of the person")
    nationality: str = Field(description="nationality of the person")

parser = PydanticOutputParser(pydantic_object=Person)

# 3. Define prompt (removed 'f' from template string)
template = PromptTemplate(
    template="Generate the name, age and nationality of {person}.\n\n{format_instruction}",
    input_variables=["person"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# 4. Invoke prompt or chain
# prompt = template.invoke({"person": "muhammad ali"})
# print(prompt)

# Un-comment to run full LCEL chain:
chain = template | model | parser
result = chain.invoke({"person": "muhammad ali"})
print(result)