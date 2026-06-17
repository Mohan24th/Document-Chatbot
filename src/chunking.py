from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data\\Java Notes .pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Pages Loaded: {len(documents)}")
print(f"Chunks Created: {len(chunks)}")

print("\nChunk Metadata:")
print(chunks[0].metadata)

print("\nChunk Length:")
print(len(chunks[0].page_content))

print("\nChunk Content:")
print(chunks[0].page_content)