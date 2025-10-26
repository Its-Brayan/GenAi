from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()
customer_support_template = PromptTemplate(
    input_variables=["customer_name", "product_name", "issue_description", "previous_interactions", "tone"],
    template = """
You are a customer support specialist for {product_name}

customer = {customer_name}
issue = {issue_description}
previous interactions = {previous_interactions}

Respond to the customer in a {tone} tone, if you don't have enough information to resolve thier issue,
ask clariffying questions. Always prioritize customer satisfaction and accurace information.

"""
)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
  messages =[
      {
          "role" : "user",
          "content":customer_support_template.format(
                customer_name="Alice Johnson",
                product_name="SmartHome Thermostat",
                issue_description="The thermostat is not connecting to Wi-Fi.",
                previous_interactions="Customer has tried restarting the device and resetting the router.",
                tone="empathetic but technical"
          )
      }
  ],
    model = "groq/compound"
)
print(f"Response: {response.choices[0].message.content}\n")
