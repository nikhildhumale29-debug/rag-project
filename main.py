import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
from langchain_community.document_loaders import PyMuPDFLoader

docs = PyPDFLoader("sample_rag.pdf").load()
print("Pages loaded:", len(docs))

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150
).split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=api_key
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", temperature=0, google_api_key=api_key
)

prompt = ChatPromptTemplate.from_template(
    "Answer using only the context below. If the answer isn't there, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("What is rag?"))