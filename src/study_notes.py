from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

text = input("Paste Content:\n")

prompt = f"""
Generate:

1. Important Topics
2. Short Notes
3. 5 Mark Questions
4. 10 Mark Questions

Content:

{text}
"""

response = llm.invoke(prompt)

print(response.content)