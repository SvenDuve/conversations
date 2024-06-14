
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

# Database Manipulation
import pandas as pd
import sqlite3
from langchain_community.utilities import SQLDatabase

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



def load_conversation_dataset():
    with open('./context/few_shot_context/conversation_dataset.json', 'r') as f:
        conversation_dataset = json.load(f)
    return conversation_dataset


def get_dbase(create=False, dbase_name='cables.db', csv_name='cables.csv'):
    
    if create:
        data = pd.read_csv(os.path.join('dbase', csv_name), delimiter=';')

        data.columns = ['Artikel', 'Beschreibung', 'Einsatzgebiet', 'Nennquerschnitt', 
                        'Außendurchmesser', 'Zulässige_Abweichung', 'Gewicht', 
                        'Min_Temperaturbeständigkeit', 'Max_Temperaturbeständigkeit', 
                        'Produktblatt']

        conn = sqlite3.connect(os.path.join('dbase', dbase_name))
        print(True) if conn else print(False)
        cursor = conn.cursor()
        # Create the table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cables (
            Artikel TEXT,
            Beschreibung TEXT,
            Einsatzgebiet TEXT,
            Nennquerschnitt TEXT,
            Außendurchmesser TEXT,
            Zulässige_Abweichung TEXT,
            Gewicht TEXT,
            Min_Temperaturbeständigkeit INTEGER,
            Max_Temperaturbeständigkeit INTEGER,
            Produktblatt TEXT
        )
        ''')

        data.to_sql('cables', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
    # conn = sqlite3.connect(dbase_name)
    # test_db(dbase_name)

    return SQLDatabase.from_uri(f"sqlite:///{os.path.join('dbase', dbase_name)}")
    # return conn


def test_db(dbase_name, table_name='cables'):
    conn = sqlite3.connect(os.path.join('dbase', dbase_name))
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {table_name}')
    rows = cursor.fetchall()
    for row in rows[:5]:
        print(row[0])
    conn.close()
    return None