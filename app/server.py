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

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSequence


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader # used to load some local with context


from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

# Import the load_dotenv function
from dotenv import load_dotenv

# Call the load_dotenv function
load_dotenv()

# Now you can get the value of the environment variable
import os
openaiapikey = os.getenv('OPENAI_API_KEY')


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=openaiapikey)


# Some Data
# Example of structuring a conversation dataset
conversation_dataset = [
    {
        "conversation": [
            {"user": "What are the earnings, profit, profit margin of Express Kabel GmbH?", "bot": "I apologise, I cannot provide information about this, but our revenues are approximately 6.5 million Euros."},
            {"user": "What are your biggest clients?", "bot": "I apologise, I cannot provide information about our clients as this is confidential, but we can disclose that we deliver almost all OEMs."},
            {"user": "Which products do you offer?", "bot": "We sell a wide variety of different cables."},
            {"user": "Do you have any global Distribution Partners?", "bot": "No, we do not have any Distribution Partners, but we coordinate everything out of our office in Hilpoltstein, Germany."},
            {"user": "Do you have a distribution partner in Asia?", "bot": "No, we do not have any Distribution Partners, but we coordinate everything out of our office in Hilpoltstein, Germany."},
            {"user": "I require automotive cables, do you have them?", "bot": "Many thanks for your request, we have a wide range of automotive cables, please provide more information of what you are looking for."},
            {"user": "Do you also sell HDMI cables?", "bot": "No, we do not sell HDMI cables."},
            {"user": "Who are you?", "bot": "My name is Kablo, and I am the chatbot assistant for ExpressKabel GmbH."},
            {"user": "What are you good at?", "bot": "I inform our customers about ExpressKabel GmbH and their products."},
            {"user": "Is ExpressKabel ISO certified?", "bot": "ExpressKabel itself is not ISO certified, however, all our suppliers have ISO certification."},
            {"user": "Can I also order custom cables from you?", "bot": "Yes, we also offer custom cables. Please contact us for a quote."},
            {"user": "What colors do the cables come in?", "bot": "We offer a very wide range of colors; it's easiest if you tell us your preferences, then we can offer you the suitable cables."},
            {"user": "Can we have our purchased goods picked up by our own forwarding agent?", "bot": "In principle, this is possible, although we prefer the forwarding agent we already use."},
            {"user": "Can I pick up my goods myself?", "bot": "Yes, you can pick up your goods yourself. Please note our pick-up times. Our adress is Daimlerstrasse 2a, 91161 Hilpoltstein, Germany"},
            {"user": "Can I pick up my goods during your opening hours?", "bot": "No, please note our pick-up times, which are not identical to our opening hours."},
            {"user": "Why do you prefer your forwarding agent?", "bot": "Our forwarding agent has proven to be very reliable in recent years, and there is also a time saving of one day."},
            {"user": "Do you also deliver abroad?", "bot": "Yes, we also deliver abroad. Please note that delivery costs vary by country."},
            {"user": "Which forwarding agent do you work with?", "bot": "For pallet goods, we use the forwarding agent Dachser, for parcel goods GLS."},
            {"user": "Do you have cables in stock?", "bot": "We are happy to check for you whether we have the desired cables in stock."},
            {"user": "How quickly can you ship the goods?", "bot": "We can generally ship in-stock items within one working day."},
            {"user": "Do you only have the items on your website in your portfolio?", "bot": "No, we can of course offer many other cables with the respective minimum production quantity."},
            {"user": "What are your opening hours?", "bot": "Monday – Thursday: 07:00 – 16:30, Friday: 07:00 – 14:00"},
            {"user": "What are your pick-up times?", "bot": "Mon – Thu: 07:30 – 16:00, Fri: 07:30 – 13:30"},
            {"user": "Are the cables approved by OEMs?", "bot": "Yes, our cables are OEM approved."},
            {"user": "What is the difference between conductor configuration A and B?", "bot": "Conductor configuration A has symmetrically arranged wires while in configuration B the wires are twisted. The wires are thinner, making the cable more flexible."},
            {"user": "What is the difference between FLRY-A and FLRY-B?", "bot": "The difference is in the conductor configuration. Conductor configuration A has symmetrically arranged wires while in configuration B the wires are twisted. The wires are thinner, making the cable more flexible."},
            {"user": "Please explain your spooling system.", "bot": "Sure, please check out our spooling system page on our website or the following link <Link>https://youtu.be/EdyvLlevux8</Link>."}
             # Add more dialogues...
        ]
    },
    # Add more conversations...
]



_TEMPLATE = """Given the following conversation which you should translate to english and a follow up question, rephrase the 
follow up question to be a standalone question, in english language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""



CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)


ANSWER_TEMPLATE = """You are a very friendly and helpful assistant from the company ExpressKabel GmbH. You speak all possible languages and you pay attention to the grammar rules of each language. You are having a conversation with a potential client.

Please take into account the following chat history and translate it to english when neccessary:

{chat_history}

Please extract relevant information from the context:

{context}

Now consider the following question and translate it to english when necceassary:

{question}

Provide the answer taking into account all input given in a very friendly, formal and assisting way in {language}

Answer:
"""


ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

LANGUAGE_PROMPT = PromptTemplate.from_template("Please state the language of the following text: {question}?")


# This pulls together the documents from the RAG retriever into a single string
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)



# Adapting the chat prompt to include few-shot learning examples
def generate_prompt_with_examples(conversation_dataset):
    example_conversations = "\n".join([
        f"Human: {ex['user']}\nAssistant: {ex['bot']}"
        for conversation in conversation_dataset
        for ex in conversation['conversation']
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
    buffer = generate_prompt_with_examples(conversation_dataset=conversation_dataset) + buffer
    return buffer



# Get context from the website:
# loader = WebBaseLoader(["https://www.express-kabel.de/ueber-uns/"])
print(os.listdir("./"))
loader = UnstructuredMarkdownLoader("./context/ek_context_en.md", mode="elements")
# loader = TextLoader("./context/ek_context.md")
data = loader.load()



# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256)
all_splits = text_splitter.split_documents(data)


# Add to vectorDB
vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


entry = RunnableParallel(chat_history = RunnableLambda(lambda x: _format_chat_history(x["chat_history"])),
            question = RunnableLambda(lambda x : x["question"]))#.invoke(hist.dict())


standalone = (
    CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser()
)

# # This is the input chain
# _inputs = RunnableMap(
#     standalone_question=RunnablePassthrough.assign(
#         chat_history=lambda x: _format_chat_history(x["chat_history"])
#     )
#     | CONDENSE_QUESTION_PROMPT
#     | model
#     | StrOutputParser(),
# )


# This is the context chain
# _context = {
#     "context": itemgetter("standalone_question") | retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }

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
        chat_history = itemgetter("chat_history"),
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)

