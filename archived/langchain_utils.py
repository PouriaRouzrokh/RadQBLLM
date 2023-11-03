##########################################################################################
# Description: functionalities to build an end-to-end QA service based on vector-bsaed
# retrieval augmentation from docs.
##########################################################################################

import datetime
import os
import shutil
from typing import List, Union

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings, Embeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter
from langchain.schema.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore, DocArrayInMemorySearch

import radqg.configs as configs
from radqg.prompts import create_prompt

# ----------------------------------------------------------------------------------------
# Chunking the text
# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
# F: get_text


def get_text(input_obj: str, temp_dir: str = "temp") -> List[Document]:
    """Extract the raw text from a single PDF or text document.

    Args:
        input_obj (str): absolute path to a pdf or text document.

    Returns:
        List[Document]: list of langchain Document obejcts.
    """
    filepath = None
    if not os.path.isfile(input_obj):
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        now = datetime.datetime.now()
        filename = now.strftime("%Y%m%d_%H%M%S.txt")
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, "w") as file:
            file.write(input_obj)
        loader = TextLoader(filepath)
    elif input_obj.endswith(".txt"):
        loader = TextLoader(input_obj)
    elif input_obj.endswith(".pdf"):
        loader = PyPDFLoader(input_obj)

    docs = loader.load()
    if filepath is not None:
        os.remove(filepath)
    return docs


# ----------------------------------------------------------------------------------------
# F: chunk_docs


def chunk_docs(
    docs: List[Document], chunk_size: int = 1500, chunk_overlap: int = 200
) -> List[Document]:
    """Split each docuemnt into chunks, based on the chunk size and chunk_overlap.

    Args:
        docs (List[Document]): list of input docs, usually the pdf pages.
        chunk_size (int, optional): chunk size. Defaults to 1500.
        chunk_overlap (int, optional): chunk overlap. Defaults to 200.

    Returns:
        List[Document]: list of chunks as Documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(docs)

    return docs


# ----------------------------------------------------------------------------------------
# F: get_all_chunks


def get_all_chunks(paths: List[str], **kwargs) -> List[Document]:
    """Split all the pdf files to chunks

    Args:
        paths (List[str]): List of pdf paths.
        kwargs: to feed chunk_size and chunk_overlap to chunk_docs.

    Returns:
        List[Document]: List of chunked pieces as Documents.
    """

    docs = []
    for path in paths:
        docs_ = get_text(path)
        docs.extend(docs_)
    docs = chunk_docs(docs, **kwargs)

    return docs


# ----------------------------------------------------------------------------------------
# RAG
# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
# F: get_vector_db


def get_vector_db(
    docs: List[Document],
    db_name: str = "chroma",
    embeddings: Embeddings = OpenAIEmbeddings(openai_api_key=configs.OPENAI_API_KEY),
    persist_dir: str = configs.VECTOR_DB_DIR,
    load_from_existing: bool = False,
) -> VectorStore:
    """Initialize vector database.

    Args:
        docs (List[Document]): list of Documents.
        db_name (str, optional): Name of vector database to use. Currently one of:
                                 ['in-memory', 'chroma']. Defaults to "in-memory".
        embeddings (Embeddings, optional): langchain class to get embeddings.
                                           Defaults to OpenAIEmbeddings().
        persist_dir (str, optional): dir to save db. Defaults to "./data/vector_db".
        load_from_existing (bool, optional): whether to load from exisiting database.
                                             Defaults to False.

    Returns:
        VectorStore: a vector store instance.
    """
    db_name2db_cls = {"in-memory": DocArrayInMemorySearch, "chroma": Chroma}

    db_cls = db_name2db_cls[db_name]
    if db_name == "in-memory":
        vector_db = db_cls.from_documents(documents=docs, embedding=embeddings)
    else:
        if load_from_existing:
            vector_db = db_cls(
                persist_directory=persist_dir, embedding_function=embeddings
            )
        else:
            # Removes existing db in the dir to prevent duplicating the db.
            if os.path.isdir(persist_dir):
                shutil.rmtree(persist_dir)
            os.makedirs(persist_dir)
            vector_db = db_cls.from_documents(
                documents=docs, embedding=embeddings, persist_directory=persist_dir
            )
            vector_db.persist()
            # pylint: disable=protected-access
            assert (
                len(docs) == vector_db._collection.count()
            ), "Number of items in db differs from input docs"

    return vector_db


# ----------------------------------------------------------------------------------------
# F: get_retriever


def get_retriever(
    vector_db: VectorStore,
    search_type: str = "similarity",
    k: int = 4,
    fetch_k: Union[int, None] = None,
    contextual_compressor: Union[str, None] = None,
    openai_api_key: str = configs.OPENAI_API_KEY,
) -> BaseRetriever:
    """Create a retriever object out of the vector db.

    Args:
        vector_db (VectorStore): vector db object.
        search_type (str, optional): main search strategy;
                           should be one of "similarity",
                           "similarity_score_threshold", or "mmr".
                           Defaults to "similarity".
        k (int, optional): number of docs to return. Defaults to 4.
        fetch_k (Union[int, None], optional): number of initial docs to return;
                                              used only in "mmr".
                                              Defaults to None.

        contextual_compressor(Union[str, None], optional): Whether to use a contextual
                                compressor wrapping the retriever.
                                Should be one of "extractor" or "filter".
                                Defaults to None, disabling this feature.
                                Remeber that when using compressors,
                                the number of docs could differ from the k arg.

    Returns:
        BaseRetriever: retriever object.
    """

    assert search_type in [
        "similarity",
        "similarity_score_threshold",
        "mmr",
    ], "Search Type not recognized!"

    if contextual_compressor is not None:
        assert contextual_compressor in [
            "extractor",
            "filter",
        ], "Contextual Compressor not recognized!"

    if search_type in ["similarity", "similarity_score_threshold"]:
        search_kwargs = {"k": k}
    elif search_type == "mmr":
        search_kwargs = {"k": k, "fetch_k": fetch_k}

    retriever = vector_db.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    if contextual_compressor:
        str2cls = {"extractor": LLMChainExtractor, "filter": LLMChainFilter}
        compressor_cls = str2cls[contextual_compressor]
        llm = OpenAI(temperature=0.0, openai_api_key=openai_api_key)
        compressor = compressor_cls.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    return retriever


# ----------------------------------------------------------------------------------------
# F: retrieval_qa


def retrieval_qa(
    retriever: BaseRetriever,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    chain_type: str = "stuff",
    prompt_keywords: dict = {"format": "", "difficulty_level": "", "criteria": ""},
    openai_api_key: str = configs.OPENAI_API_KEY,
) -> RetrievalQA:
    """Create a retrieval QA chain against a retriever.

    Args:
        retriever (BaseRetriever): retriever object created from a vector db.
        model (str, optional): the LLM model name to use for QA.
        temperature (float, optional): LLM's temperature. Defaults to 0.0.
        chain_type (str, optional): chain type to use for combining docs.
                                    Could be one of "stuff",
                                    "map_reduce", "map_rerank", and "refine".
                                    Defaults to "stuff".
        prompt_keywords (dict, optional): keywords to use in the prompt.
        openai_api_key (str, optional): OpenAI API key. Defaults to OPENAI_API_KEY.

    Returns:
        RetrievalQA: retrieval QA chain object.
    """

    llm = ChatOpenAI(
        model_name=model,
        temperature=temperature,
        openai_api_key=openai_api_key,
    )

    chain_type_kwargs = {}
    if chain_type == "stuff":
        chain_type_kwargs["prompt"] = create_prompt(**prompt_keywords)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type=chain_type,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )

    return qa_chain


# ----------------------------------------------------------------------------------------
# F: prepare_pipeline


def prepare_pipeline(
    source_paths: List[str],
    prompt_keywords: dict = {"format": "", "difficulty_level": "", "criteria": ""},
    **kwargs,
) -> RetrievalQA:
    """Prepares the Retrieval QA pipeline.

    Args:
        source_paths (List[str]): list of source pdf files.
        prompt_keywords (dict, optional): keywords to use in the prompt.

    Returns:
        RetrievalQA: chain for retrieval QA.
    """

    docs = get_all_chunks(
        source_paths, chunk_size=configs.CHUNK_SIZE, chunk_overlap=configs.CHUNK_OVERLAP
    )
    embedding_llm = OpenAIEmbeddings(
        model=configs.EMBEDDING_MODEL, openai_api_key=configs.OPENAI_API_KEY
    )
    vector_db = get_vector_db(
        docs,
        db_name=configs.VECTOR_DB,
        load_from_existing=False,
        embeddings=embedding_llm,
    )
    retriever = get_retriever(
        vector_db,
        search_type=configs.SEARCH_TYPE,
        k=configs.K if "K" not in kwargs else kwargs["K"],
        fetch_k=configs.FETCH_K if "FETCH_K" not in kwargs else kwargs["FETCH_K"],
        contextual_compressor=configs.COMPRESSOR,
    )
    chain = retrieval_qa(
        retriever,
        model=configs.MODEL if "MODEL" not in kwargs else kwargs["MODEL"],
        temperature=configs.TEMPERATURE
        if "TEMPERATURE" not in kwargs
        else kwargs["TEMPERATURE"],
        chain_type=configs.CHAIN_TYPE,
        prompt_keywords=prompt_keywords,
    )
    print(f"The pipeline is ready using the {configs.MODEL.upper()} model.")
    return chain
