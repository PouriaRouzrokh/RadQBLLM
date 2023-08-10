##########################################################################################
# Description: A script containing functionalities to process the text data.
##########################################################################################

from typing import List

from langchain.schema.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# ----------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------


def count_tokens(string: str, encoding_name: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a string.

    Args:
        string (str): the string to count tokens in.
        encoding_name (str, optional): Name of the LLM model. Defaults to "gpt-3.5-turbo".

    Returns:
        int: number of tokens in the string.
    """

    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens


# ----------------------------------------------------------------------------------------
# Chunking the text
# ----------------------------------------------------------------------------------------


def get_text(file_path: str) -> List[Document]:
    """Extract the raw text from a single PDF or text document.

    Args:
        file_path (str): absolute path to a pdf or text document.

    Returns:
        List[Document]: list of langchain Document obejcts.
    """
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    docs = loader.load()

    return docs


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
