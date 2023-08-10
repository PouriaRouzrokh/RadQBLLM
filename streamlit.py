##########################################################################################
# Description: GUI app code based on Streamlit demonstrating the pipeline.
##########################################################################################

import os

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from radqbllm.utils.text_utils import get_all_chunks
from radqbllm.rag import (
    get_vector_db,
    get_retriever,
    retrieval_qa,
)

# ----------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------


def save_file(uploaded_file: UploadedFile, save_dir: str = "./data/.tmp") -> str:
    """Save uploaded file to disk.

    Args:
        uploaded_file (UploadedFile): Streamlit UploadedFile instance.
        save_dir (str, optional): directory to save the file.
        Defaults to './data/.tmp'.

    Returns:
        str: abs saved path.
    """
    path = os.path.join(save_dir, uploaded_file.name)
    with open(path, "wb") as pdf:
        pdf.write(uploaded_file.getbuffer())

    return path


# ----------------------------------------------------------------------------------------
# Steamlit App
# ----------------------------------------------------------------------------------------


def main():
    """Main StreamLit app."""
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    st.title("Automated Radiology Question and Answering with Large Language Models")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("./data/.tmp", exist_ok=True)
        paths = [
            save_file(uploaded_file=file, save_dir="./data/.tmp")
            for file in uploaded_files
        ]
        docs = get_all_chunks(paths)

        if st.session_state.vector_db is not None:
            vector_db = st.session_state.vector_db
        else:
            with st.spinner("Building the vectore store ..."):
                vector_db = get_vector_db(docs, db_name="in-memory")
                st.session_state.vector_db = vector_db

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            output2type = {
                "similarity": "Similarity Search",
                "mmr": "Maximal Marginal Relevance (MMR)",
                "similarity_score_threshold": "Similarity Score Threshold",
            }
            search_type = st.radio(
                "Search Type",
                options=["similarity", "mmr", "similarity_score_threshold"],
                format_func=lambda x: output2type[x],
                label_visibility="visible",
            )

            fetch_k = None
            if search_type == "mmr":
                with col2:
                    fetch_k = st.number_input(
                        "Num of initial docs to fetch", min_value=1, value=10
                    )
                    k = st.number_input("Num of docs to feed LLM", min_value=1, value=4)
            else:
                with col2:
                    k = st.number_input("Num of docs to feed LLM", min_value=1, value=4)

        with col3:
            output2compressor = {
                "extractor": "LLM Chain Extractor",
                "filter": "LLM Chain Filter",
            }
            contextual_compressor = st.radio(
                "Contextual Compression",
                options=[None, "extractor", "filter"],
                format_func=lambda x: output2compressor[x] if x else None,
                label_visibility="visible",
            )

        with col4:
            output2chain = {
                "stuff": "Stuff",
                "map_reduce": "Map Reduce",
                "map_rerank": "Map Rerank",
                "refine": "Refine",
            }
            chain_type = st.radio(
                "Chain Type",
                options=["stuff", "map_reduce", "map_rerank", "refine"],
                format_func=lambda x: output2chain[x],
                label_visibility="visible",
            )

        retriever = get_retriever(
            vector_db,
            search_type=search_type,
            k=k,
            fetch_k=fetch_k,
            contextual_compressor=contextual_compressor,
        )
        qa_chain = retrieval_qa(
            retriever=retriever, temperature=0.0, chain_type=chain_type
        )
        question = st.text_input("Enter your question")

        if question:
            result = qa_chain({"query": question})
            st.write(result["result"])
            for i, doc in enumerate(result["source_documents"]):
                st.subheader(f"Doc: {i+1}")
                st.text(f"Source File: {doc.metadata['source']}")
                st.text(f"Page Num: {doc.metadata['page']}")


# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    st.set_page_config(page_title="Radiology QA", layout="wide")
    main()
