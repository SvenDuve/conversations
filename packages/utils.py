
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
import json

# Vector Stores
#from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart



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



def load_conversation_dataset():
    with open('./context/few_shot_context/conversation_dataset.json', 'r') as f:
        conversation_dataset = json.load(f)
    return conversation_dataset


def sendProductRequest(message: str, summary: str):

    # Environment variables for security
    sender_email = os.getenv('EMAIL')
    receiver_email = "svenduve@gmail.com"
    # receiver_email = "bjoern.etzel@express-kabel.de"
    # password = 'Pilsener/123'  # Secure way to handle credentials
    password = os.getenv('EMAIL_PASSWORD')  # Secure way to handle credentials
    # Create the multipart container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Chat Summary"
    # Email body
    text = "Hello, this is a summary of the chat.\n\n" + summary + "\n\n and this is the requested product: \n\n" + message
    msg.attach(MIMEText(text, 'plain'))
    # Send the email
    try:
        # Create server object with SSL option
        server = smtplib.SMTP_SSL('send.one.com', 465)
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
    except Exception as e:
        print("Error: unable to send email", e)