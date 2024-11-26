__author__ = "akhtar"

import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv('AZURE_OPENAI_EMBEDDING_MODEL')

# LLM with function call
llm = AzureChatOpenAI(
    model=AZURE_OPENAI_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0
)


# Post-processing
def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


prompt = hub.pull("rlm/rag-prompt")

# Chain
rag_chain = prompt | llm | StrOutputParser()
