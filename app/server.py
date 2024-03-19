#!/usr/bin/env python

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



from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import WebBaseLoader

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

# Import the load_dotenv function
from dotenv import load_dotenv

# Call the load_dotenv function
load_dotenv()

# Now you can get the value of the environment variable
import os
openaiapikey = os.getenv('OPENAI_API_KEY')


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openaiapikey)

_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)


ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
# ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


# This pulls together the documents from the RAG retriever into a single string
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    # print("Doc Strings:")
    # print(doc_strings)
    return document_separator.join(doc_strings)



# This formats the chat history into a string
def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer



# Get context from the website:
# loader = WebBaseLoader(["https://www.express-kabel.de/ueber-uns/"])
loader = WebBaseLoader("https://www.express-kabel.de/ueber-uns/")

data = loader.load()


# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


# Add to vectorDB
vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()




# vectorstore = FAISS.from_texts(
#     ["Martin liebt Bier", 
#      "Johannes trinkt Whiskey", 
#      "Sven trinkt nie Alkohol, er mag Tee.", 
#      "Martin und Johannes trinken gemeinsam Bier", 
#      "Nadja, Martin und Sven trinken gemeinsam Schnaps."], embedding=OpenAIEmbeddings()
# )
# retriever = vectorstore.as_retriever()

# This is the input chain
_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
)


# This is the context chain
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}



# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str



conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | model | StrOutputParser()
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)

