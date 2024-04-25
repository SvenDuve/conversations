
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
    CSVLoader
)
import os

# Vector Stores
#from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter


def langchain_document_loader(path):
    documents = []

    txt_loader = DirectoryLoader(
        path, glob="*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    md_loader = DirectoryLoader(
        path, glob="*.md", loader_cls=UnstructuredMarkdownLoader, show_progress=True
    )
    documents.extend(md_loader.load())

    pdf_loader = DirectoryLoader(
        path, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())
    
    return documents



def select_embeddings_model(LLM_service="OpenAI", api_key=None):
    """Connect to embeddings model endpoint based on the name of the service"""
    if LLM_service == "OpenAI":
        embeddings = OpenAIEmbeddings(
            model = 'text-embedding-ada-002',
            api_key=api_key
        )
    if LLM_service == "HuggingFace":
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name="thenlper/gte-large"
        )
    return embeddings




def create_vectorstore(embeddings, documents, vectorstore_name):
    """Create Chroma Vectorstore"""
    persist_directory = os.path.join("./", vectorstore_name)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


def instantiate_LLM(LLM_provider="OpenAI", api_key=None, temperature=0.5, top_p=0.95, model_name=None):
    """Instantiate Language Model"""
    if LLM_provider == "OpenAI":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            model_kwargs={
                "top_p":top_p
                }
            )
        
    if LLM_provider == "HuggingFace":
        llm = HuggingFaceHub(
            repo_id = model_name,
            huggingfacehub_api_token = api_key,
            model_kwargs={
                "temperature":temperature,
                "top_p":top_p,
                "do_sample":True,
                "max_new_tokens":1024
                }
            )
        
    return llm