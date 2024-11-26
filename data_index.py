import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

load_dotenv()

AZURE_OPENAI_EMBEDDING_MODEL = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL')


def get_retriever():
    file_path = "docs"
    file_names = [
        "leave_policy.pdf",
        "certification_policy.pdf",
        "reimbursement_policy.pdf",
        "rewards_and_recognition_guide.pdf",
    ]
    docs = [
        PyPDFLoader(os.path.join(file_path, file_name)).load() for file_name in file_names
    ]
    doc_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(doc_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=AzureOpenAIEmbeddings(model=AZURE_OPENAI_EMBEDDING_MODEL),
        persist_directory="./db",
    )
    vectorstore.add_documents(documents=doc_splits)
    retriever = vectorstore.as_retriever(k=5)
    return retriever

# retriever = vectorstore.as_retriever()
#
#   vectorstore = Chroma(
#         collection_name="rag-chroma",
#         embedding_function=AzureOpenAIEmbeddings(model=AZURE_OPENAI_EMBEDDING_MODEL),
#         persist_directory="./db",
#     )
