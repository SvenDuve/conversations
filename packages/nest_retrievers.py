
from .utils import *
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
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

# Contextual Compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Cohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere



# unclear
from langchain_core.messages.utils import get_buffer_string
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field


def HelloWorld():
    print("Hello World!")


def vectorstore_backed_retriever(vectorstore, search_type="similarity", k=4, score_threshold=None):
    """Create a retriever backed by a vectorstore with Parameters:
    search_type: str = "similarity", "mmr" or "similarity_score_threshold"
    k: int = number of documents to retrieve
    score_threshold: float = minimum similarity score for retrieval
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever



def create_compression_retriever(embeddings, base_retriever, chunk_size=512, k=16, similarity_threshold=None):
    """Create a compression retriever with Parameters:
    embeddings: embeddings model
    base_retriever: a vectorstore backed retriever
    chunk_size: int = size of the chunk
    k: int = number of documents to retrieve
    similarity_threshold: float = minimum similarity score for retrieval
    """
    
    # 1. splittin documents into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")

    # 2. filtering out redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query

    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold)

    # 4. Reordering the documents based on relevance
    # Less relevant documents are move do the middle of the document, more relevant to the top

    reordering = LongContextReorder()

    # 5. Create compressor pipeline and retriever

    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )

    return compression_retriever


def CohereRerank_retriever(
        base_retriever,
        cohere_api_key,
        cohere_model="rerank-multilingual-v2.0",
        top_n=8
    ):
    """Create a Cohere rerank retriever with Parameters:
    base_retriever: a vectorstore backed retriever
    cohere_api_key: str = Cohere API key
    cohere_model: str = Cohere model name
    top_n: int = number of documents to retrieve
    """

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        model=cohere_model,
        top_n=top_n
    )

    # compressor = CohereRerank(
    #     cohere_api_key=cohere_api_key
    # )

    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return retriever_Cohere




def retrieval_blocks(
    build_vectorstore=True,# if True a Chroma vectorstore is created, else the Chroma vectorstore will be loaded
    LLM_service="OpenAI",
    vectorstore_name="Vit_All_OpenAI_Embeddings",
    chunk_size = 1600, chunk_overlap=200, # parameters of the RecursiveCharacterTextSplitter
    retriever_type="vectorstore_backed_retriever",
    base_retriever_search_type="similarity", base_retriever_k=10, base_retriever_score_threshold=None,
    compression_retriever_k=16,
    cohere_api_key="***", cohere_model="rerank-multilingual-v2.0", cohere_top_n=8,
):
    """
    Rertieval includes: document loaders, text splitter, vectorstore and retriever. 
    
    Parameters: 
        create_vectorstore (boolean): If True, a new Chroma vectorstore will be created. Otherwise, an existing vectorstore will be loaded.
        LLM_service: OpenAI, Google or HuggingFace.
        vectorstore_name (str): the name of the vectorstore.
        chunk_size and chunk_overlap: parameters of the RecursiveCharacterTextSplitter, default = (1600,200).
        
        retriever_type (str): in [Vectorstore_backed_retriever,Contextual_compression,Cohere_reranker]
        
        base_retriever_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retriever_k: The most similar vectors to retrieve (default k = 10).  
        base_retriever_score_threshold: score_threshold used by the base retriever, default = None.

        compression_retriever_k: top k documents returned by the compression retriever, default=16
        
        cohere_api_key: Cohere API key
        cohere_model (str): The Cohere model can be either 'rerank-english-v2.0' or 'rerank-multilingual-v2.0', with the latter being the default.
        cohere_top_n: top n results returned by Cohere rerank, default = 8.
   
    Output:
        retriever.
    """
    try:
        # Create new Vectorstore (Chroma index)
        if build_vectorstore: 
            # 1. load documents
            documents = langchain_document_loader("./context")
            
            # 2. Text Splitter: split documents to chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators = ["\n\n", "\n", " ", ""],    
                chunk_size = chunk_size,
                chunk_overlap= chunk_overlap
            )
            chunks = text_splitter.split_documents(documents=documents)
            
            # 3. Embeddings
            embeddings = select_embeddings_model(LLM_service=LLM_service)
        
            # 4. Vectorsore: create Chroma index
            vector_store = create_vectorstore(
                embeddings=embeddings,
                documents = chunks,
                vectorstore_name=vectorstore_name,
            )
            print("here")
    
        # 5. Load a Vectorstore (Chroma index)
        else: 
            embeddings = select_embeddings_model(LLM_service=LLM_service)        
            vector_store = Chroma(
                persist_directory = os.path.join("./") + vectorstore_name,
                embedding_function=embeddings
            )
            
        # 6. base retriever: Vector store-backed retriever 
        base_retriever = vectorstore_backed_retriever(
            vector_store,
            search_type=base_retriever_search_type,
            k=base_retriever_k,
            score_threshold=base_retriever_score_threshold
        )
        retriever = None
        if retriever_type=="vectorstore_backed_retriever": 
            retriever = base_retriever
    
        # 7. Contextual Compression Retriever
        if retriever_type=="Contextual_compression":    
            retriever = create_compression_retriever(
                embeddings=embeddings,
                base_retriever=base_retriever,
                k=compression_retriever_k,
            )
    
        # 8. CohereRerank retriever
        if retriever_type=="Cohere_reranker":
            retriever = CohereRerank_retriever(
                base_retriever=base_retriever, 
                cohere_api_key=cohere_api_key, 
                cohere_model=cohere_model, 
                top_n=cohere_top_n
            )
    
        print(f"\n{retriever_type} is created successfully!")
        print(f"Relevant documents will be retrieved from vectorstore ({vectorstore_name}) which uses {LLM_service} embeddings \
and has {vector_store._collection.count()} chunks.")
        
        return retriever
    except Exception as e:
        print(e)