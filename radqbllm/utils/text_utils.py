##########################################################################################
# Description: A script containing functionalities to process the text data.
##########################################################################################


from typing import List

from langchain.schema.document import Document
from langchain.document_loaders import TextLoader
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

def get_text(txt_file_path: str) -> List[Document]:
    """Extract the raw text from a single PDF document, using pypdf library.

    Args:
        pdf_path (str): absolute path to pdf document.

    Returns:
        List[Document]: list of langchain Document obejcts.
    """

    loader = TextLoader(txt_file_path)
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


def get_all_chunks(data: List[dict], **kwargs) -> List[Document]:
    """Split all the pdf files to chunks

    Args:
        data (List[str]): List of document dictionaris
        kwargs: to feed chunk_size and chunk_overlap to chunk_docs.

    Returns:
        List[Document]: List of chunked pieces as Documents.
    """

    # Building a dictionary from the file path to the other fields.
    path_dict = dict()
    for data_point in data:
        records = dict()
        for key in data_point:
            if key != 'path':
                records[key] = data_point[key]
        path_dict[data_point['path']] = records
    
    # Building the chunks
    docs = []
    for data_point in data:
        doc = get_text(data_point['path'])
        docs.extend(doc)
    chunks = chunk_docs(docs, **kwargs)
    for chunk in chunks:
         chunk.metadata.update(path_dict[chunk.metadata['source']])
    
    return chunks
