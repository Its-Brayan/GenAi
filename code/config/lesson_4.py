import os
from paths import OUTPUTS_DIR
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq
from utils import load_publication, save_text_to_file
from langchain_core.output_parsers.pydantic import PydanticOutputParser

load_dotenv()

class Entity(BaseModel):
    type : str = Field(description="The type of the entity, either 'model' or 'task'")
    name : str = Field(description="the name of the entity")

class Entities(BaseModel):
    entities : list[Entity] = Field(
        description="The entities mentioned in the publication"
    )

def no_structured_output(model : str ='Groq'):
    """
    this function demonstrates how to use a llm without a structured output
    
    """
    publicaton_content = load_publication()

    prompt = """
             provide a list of entities mentioned in the publication, an entitiy is either a model or a task
             <publicaton>
             {publication_content}
             </publication>
             """.format(publication_content=publicaton_content)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        model = "groq/compound"
    )
    response_content = response.choices[0].message.content
    saved_text = f""" #prompt = {prompt}

    # Response:
    {response_content}

"""    
    save_text_to_file(
     saved_text,
     os.path.join(OUTPUTS_DIR,f"no structured_output.md"),
     header=f"LLM response without structured output", 
)
def with_prompting_structured_output(model : str = 'Groq'):
    """
    this function demonstrates how to use the llm with prompting for structured output
    """
    publication_content = load_publication()
    prompt = """ 
          provde a list of entitites mentioned in the publication, ane entity is either a model or a task
          
          <publication">
          {publication_content}
          </publication>

          Return a JSON object with a single field 'entities' which is a list of dictionaries. Each dictionary should have two fields : "type" and "name"

          Example:
          {{
            "entities":[
            {{
                "type":"model",
                "name" :"groq/compound"
            }},
            {{
                "type":"task",
                "name":"text generation"
            }}
            ]
          }}
          """.format(
              publication_content=publication_content
          )
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        model = "groq/compound"
    )
    response_content  =response.choices[0].message.content
    saved_text = f""" #prompt = {prompt}
    #Response:
    {response_content}
    """
    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR,f"with_prompting_structured_output.md"),
        header=f"LLM response with prompting for structured output",
    )
def with_output_parsers(model : str = 'Groq'):
    """
    this function demonstrates how to use the llm with output parsers for structured output
    """
    publication_content = load_publication()
    prompt ="""
            provide a list of entities mentioned in the publication, an entity is either a model or a task
            <publication>
            {publication_content}
            </publication>
            {format_instructions}
"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    output_parsers = PydanticOutputParser(pydantic_object=Entities)
    format_instructions = output_parsers.get_format_instructions()
    prompt = prompt.format(
        publication_content = publication_content,
        format_instructions = format_instructions
    )
    response = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": prompt
            },
        ],
        model = "groq/compound"
    )
    response_content = response.choices[0].message.content
    parsed_response = output_parsers.parse(response_content)
    saved_text = f""" #prompt = {prompt}
    #Response:
    #Before Parsing:
    {response_content}
    
    #After Parsing:
    {parsed_response}
    """
    save_text_to_file(
        saved_text,
        os.path.join(OUTPUTS_DIR,f"with_output_parsers.md"),
        header=f"LLM response with output parsers for structured output",
    )
if __name__ == "__main__":
    # no_structured_output()
    # with_prompting_structured_output(),
    with_output_parsers()