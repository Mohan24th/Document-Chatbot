from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

with open(
    "data/summary_context.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

prompt = f"""
Summarize the following document.

{text}
"""

response = llm.invoke(prompt)

print(response.content)