import os
import numpy as np
from google.colab import userdata, files

try:
    os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
except Exception:
    print("Critical: Please ensure your key is saved in Colab Secrets as 'OPENAI_API_KEY'")

#uploading the pdf 
print("Step 1/5: upload your PDF document...")
uploaded_files = files.upload()

if not uploaded_files:
    raise FileNotFoundError(" select a file")

pdf_filename = list(uploaded_files.keys())[0]
print(f" Successfully uploaded: {pdf_filename}")

#extracting the text from the pdf you've uploaded
from langchain_community.document_loaders import PyPDFLoader
try:
    loader = PyPDFLoader(pdf_filename)
    docs = loader.load()
    print(" Text successfully extracted from your PDF document.")
except Exception as e:
    raise RuntimeError(f" Failed to parse the PDF file. Details: {e}")

#chunking
print("\n Step 2/5: Splitting text into chunks...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f" Successfully created {len(chunks)} contextual chunks from your PDF.")

#embedding and retriever
#chroma-->vector database
print("\n Step 3/5: Loading local embedding model...")
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
print("Vector database is indexed locally and live.")

#LLM configuration
print("\n Step 4/5: Connecting to Engine...")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="gpt-5.2-chat",
    temperature=0.0,
    openai_api_base="https://api.bluesminds.com/v1"
)

system_prompt = (
    f"You are an expert document assistant. The user has uploaded a file named: '{pdf_filename}'.\n"
    "Use the provided pieces of context to answer the user's question directly. "
    "If the user asks about the name of the file or document, use the filename provided above. "
    "If you cannot find structural information (like total chapters) explicitly stated in the context, "
    "state what you can find rather than guessing.\n\n"
    "Context:\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

#RAG pipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print(" Complete RAG pipeline assembled successfully.")

#chat loop
print("\n Step 5/5: Enter your questions below.")
print("Type 'end' at any point to stop the program.\n")

while True:
    print("=" * 60)
    user_input = input(" Enter your question: ")

    if user_input.strip().lower() == "end":
        print("\n Chat session closed. Goodbye!")
        print("=" * 60)
        break

    if not user_input.strip():
        continue

    try:
        response = rag_chain.invoke({"input": user_input})
        print(f"\n Bot Answer:\n{response['answer']}")

    except Exception as e:
        print(f" Error communicating with API: {e}")
