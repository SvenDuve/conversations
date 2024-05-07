#!/usr/bin/env python
import time
start_time = time.time()
from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",  # Allow localhost for development
    "https://xp-cable-chat-bot-git-main-martin-renners-projects.vercel.app",  # Allow localhost for development
    "https://xp-cable-chat-bot-martin-renners-projects.vercel.app",  # Allow your production URL
    "https://xp-cable-chat-bot.vercel.app",  # Allow your production URL
]

from packages.nest_retrievers import HelloWorld, vectorstore_backed_retriever, create_compression_retriever, CohereRerank_retriever, retrieval_blocks
from packages.utils import langchain_document_loader, select_embeddings_model, create_vectorstore, instantiate_LLM, load_conversation_dataset

## New Imports
import numpy as np
from itertools import combinations
from operator import itemgetter
from typing import List, Tuple

# OpenAI
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI

# Hugging Face
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# prompts memory chains
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSequence
from langchain.schema import Document, format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

# Load docs
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader,
    CSVLoader
)

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

# Output Parsers
from langchain_core.output_parsers import StrOutputParser


# Vector Stores
#from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma


# Contextual Compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Cohere
# from langchain.retrievers.document_compressors import CohereRerank
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere



# unclear
from langchain_core.messages.utils import get_buffer_string
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field


# Import the load_dotenv function
from dotenv import load_dotenv

# Call the load_dotenv function
load_dotenv()

# Now you can get the value of the environment variable
import os
openaiapikey = os.getenv('OPENAI_API_KEY')
hfapikey = os.getenv('HF_API_KEY')
cohereapikey = os.getenv('COHERE_API_KEY')


##


from langchain_core.prompts import format_document

print("Package Loading Time: ", time.time() - start_time)


# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=openaiapikey)
model = instantiate_LLM("OpenAI", api_key=openaiapikey, temperature=0.1, model_name="gpt-3.5-turbo")
# model = instantiate_LLM("HuggingFace", api_key=hfapikey, temperature=0.1, model_name="mistralai/Mistral-7B-Instruct-v0.2")


# Some Data
# Example of structuring a conversation dataset
conversation_dataset = load_conversation_dataset()


# Zero shot classifier

ZERO_SHOT_PROMPT_TEMPLATE = """From the following request:

Request: {question}

classify the request into one of the following categories:

- general
- product
- service
- conditions

"""

ZERO_SHOT_PROMPT = PromptTemplate.from_template(ZERO_SHOT_PROMPT_TEMPLATE)


# Condense Question
_TEMPLATE = """Given the following conversation a follow up question, rephrase the 
follow up question to be a standalone question, in english language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""



CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)


# General Answer Template
ANSWER_TEMPLATE = """You are a very friendly and helpful assistant from the company ExpressKabel GmbH. You speak all possible languages and you pay attention to the grammar rules of each language. You are having a conversation with a potential client.

Please take into account the following example conversations:

{few_shot_conv}

Please extract relevant information from the context:

{context}

Now consider the following question and translate it to english when necceassary:

{question}

Provide the answer only if you are sure with two to three sentences, taking into account all input given in a very friendly, formal and assisting way in {language}

Answer:
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


# Simple Language Prompt
LANGUAGE_PROMPT = PromptTemplate.from_template("Please state the language of the following text: {question}?")


# Placeholder Prmpt for the RAG retriever
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

# This pulls together the documents from the RAG retriever into a single string
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


# Not used anymore
# Adapting the chat prompt to include few-shot learning examples
# def generate_prompt_with_examples(conversation_dataset):
#     example_conversations = "\n".join([
#         f"Human: {ex['user']}\nAssistant: {ex['bot']}"
#         for conversation in conversation_dataset
#         for ex in conversation['conversation']
#     ])
#     return f"\n{example_conversations}"



# Adapting the chat prompt to include few-shot learning examples
def _get_conversations(classification: str, conversation_dataset=conversation_dataset):
    example_conversations = "\n".join([
        f"Human: {conversation['user']}\nAssistant: {conversation['bot']}"
        for conversation in conversation_dataset[classification.lower()]
    ])
    return f"\n{example_conversations}"


# This formats the chat history into a string
def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    #buffer = generate_prompt_with_examples(conversation_dataset=conversation_dataset) + buffer
    return buffer


# Get context from the website:
# loader = WebBaseLoader(["https://www.express-kabel.de/ueber-uns/"])
# print(os.listdir("./"))
# loader = UnstructuredMarkdownLoader("./context/ek_context_en.md", mode="elements")
# # loader = TextLoader("./context/ek_context.md")
# data = loader.load()



# # Split
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256)
# all_splits = text_splitter.split_documents(data)


# # Add to vectorDB
# vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


start_time = time.time()
print("Loading Vectorstore")


retriever = retrieval_blocks(
    build_vectorstore=True,
    LLM_service = "OpenAI",
    vectorstore_name="vs",
    chunk_size=1024,
    chunk_overlap=256,
    # retriever_type="vectorstore_backed_retriever",
    retriever_type="Cohere_reranker",
    # retriever_type="Contextual_compression",
    base_retriever_search_type = "similarity",
    base_retriever_k = 10,
    base_retriever_score_threshold = None,
    compression_retriever_k = 16,
    cohere_api_key = cohereapikey,
    cohere_model = "rerank-multilingual-v2.0",
    cohere_top_n = 2
)

print("Retriever Loading Time: ", time.time() - start_time)
# retriever = vectorstore.as_retriever()


entry = RunnableParallel(chat_history = RunnableLambda(lambda x: _format_chat_history(x["chat_history"])),
            question = RunnableLambda(lambda x : x["question"]))#.invoke(hist.dict())


zero_shot_classifier = (
    ZERO_SHOT_PROMPT
    | model
    | StrOutputParser()
)


standalone = (
    CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser()
)


checklanguage = (
    LANGUAGE_PROMPT
    | model
    | StrOutputParser()
)

# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str



conversational_qa_chain = (
    entry
    | RunnableParallel(
        few_shot_conv = standalone | zero_shot_classifier | _get_conversations,
        #chat_history = itemgetter("chat_history"),
        question = standalone,
        context = standalone | retriever | _combine_documents,
        language = checklanguage 
    )
    | ANSWER_PROMPT # Requires question, history and context
    | model
    | StrOutputParser()
)


chain = conversational_qa_chain.with_types(input_type=ChatHistory)





app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Edit this to add the chain you want to add
add_routes(app, chain, enable_feedback_endpoint=True)

# add a simple hello world route
@app.get("/")
def read_root():
    return {"Hello": "World"}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)

