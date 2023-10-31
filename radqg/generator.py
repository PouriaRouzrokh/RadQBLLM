##########################################################################################
# Description: A script containing the Q/A Generator class.
##########################################################################################

import datetime
import random
from typing import Union
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import radqg.configs as configs
from radqg.parse_html import retrieve_figures, retrieve_articles


# ----------------------------------------------------------------------------------------
# Generator


class Generator:
    """A class for generating questions and answers from a given directory of
    RadioGraphics articles saved as HTML files and supporting directories in a given
    folder.
    """

    def __init__(
        self,
        data_dir: str,
        embed_fn: bool,
        chunk_size: int = configs.CHUNK_SIZE,
        chunk_overlap: int = configs.CHUNK_OVERLAP,
        collection_name: str = None,
        generator_model: str = configs.OPENAI_GENERATOR_MODEL,
        content_editor_model: str = configs.OPENAI_CONTENT_EDITOR_MODEL,
        format_editor_model: str = configs.OPENAI_FORMAT_EDITOR_MODEL,
    ):
        """The constructor of the Generator class."""

        self.data_dir = data_dir
        self.embed_fn = embed_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.generator_memory = dict()
        self.collection = self.create_collection()
        self.generator_model = generator_model
        self.content_editor_model = content_editor_model
        self.format_editor_model = format_editor_model

    def create_collection(self) -> chromadb.Collection:
        """A method to create a collection of articles and figures from a given
        directory of saved RadioGraphics articles in the format of HTML files."""

        # Retrieving articles and figures
        self.article_list = retrieve_articles(self.data_dir)
        self.fig_list = retrieve_figures(self.data_dir)

        # Building the collection
        if self.collection_name is None:
            now = datetime.datetime.now()
            collection_name = now.strftime("%Y%m%d_%H%M%S")
        else:
            collection_name = self.collection_name
        client = chromadb.PersistentClient(path=configs.VECTOR_DB_DIR)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embed_fn,
        )

        # Adding chunked articles' text to the collection
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        for article in self.article_list:
            chunks = text_splitter.split_text(article["article_full_text"])
            collection.add(
                documents=chunks,
                metadatas=[
                    {
                        "type": "article",
                        "article_path": article["article_file_path"],
                        "article_name": article["article_file_name"],
                        "chunk_index": i,
                    }
                    for i in range(len(chunks))
                ],
                ids=[f"{article['article_file_name']}_{i}" for i in range(len(chunks))],
            )

        # Adding figure captions to the collection
        figure_names = [item["figure_name"] for item in self.fig_list]
        figure_captions = [item["caption_text"] for item in self.fig_list]
        figure_paths = [item["figure_path"] for item in self.fig_list]
        figure_article_names = [item["article_file_name"] for item in self.fig_list]
        metadatas = [
            {
                "type": "figure_caption",
                "figure_path": figure_paths[i],
                "article_name": figure_article_names[i],
                "figure_names": figure_names[i],
            }
            for i in range(len(self.fig_list))
        ]
        collection.add(
            documents=figure_captions,
            metadatas=metadatas,
            ids=[
                f"{item['article_file_name']}_{item['figure_name']}"
                for item in self.fig_list
            ],
        )
        print(f'The collection "{collection_name}" has been created with:')
        print(
            f"    {len(self.fig_list)} figures from {len(self.article_list)} articles"
        )
        return collection

    def _weighted_sampler(self, distances: list) -> iter:
        """An internal method to generate a weighted sampler based on the distances of the
        figure caption and the user-specified topic of interest."""

        # Calculate weights: smaller distances are more likely to be chosen
        weights = [1 / (d + 1e-6) ** 50 for d in distances]

        # Normalizing weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        while True:
            # Randomly selecting an index based on weights
            selected_index = random.choices(
                range(len(distances)), weights=normalized_weights
            )[0]
            yield selected_index

    def _random_sampler(self, captions) -> iter:
        """An intenral method to generate a random sampler."""
        indices = list(range(len(captions)))
        random.shuffle(indices)
        for index in indices:
            yield index

    def setup_qbank(
        self,
        topic: str = None,
    ) -> tuple[list[str], list[str], list[str], iter]:
        """A method to set up the question bank depending on the user-specified topic."""

        if topic is not None:
            out = self.collection.query(
                query_texts=topic,
                n_results=len(self.fig_list),
                where={"type": "figure_caption"},
            )
            captions = out["documents"][0]
            article_names = [
                out["metadatas"][0][i]["article_name"] for i in range(len(captions))
            ]
            figure_paths = [
                out["metadatas"][0][i]["figure_path"] for i in range(len(captions))
            ]
            distances = out["distances"][0]
            sampler = self._weighted_sampler(distances)
        else:
            out = self.collection.get(
                where={"type": "figure_caption"},
            )
            captions = out["documents"]
            article_names = [
                out["metadatas"][i]["article_name"] for i in range(len(captions))
            ]
            figure_paths = [
                out["metadatas"][i]["figure_path"] for i in range(len(captions))
            ]
            sampler = self._random_sampler(captions)

        return article_names, figure_paths, captions, sampler

    def select_figure(
        self,
        article_names: list[str],
        figure_paths: list[str],
        captions: list[str],
        sampler: iter,
        max_q_per_fig: int = 1,
        reset_memory=False,
    ) -> tuple[str, str, str]:
        """A method to randomly select a figure from the question bank."""

        if reset_memory:
            self.generator_memory = dict()
        while True:
            selected_idx = next(sampler)
            current_q_count = self.generator_memory.get(selected_idx, 0)
            if current_q_count < max_q_per_fig:
                self.generator_memory[selected_idx] = current_q_count + 1
                break
        selected_article_name = article_names[selected_idx]
        selected_figpath = figure_paths[selected_idx]
        selected_caption = captions[selected_idx]

        return selected_article_name, selected_figpath, selected_caption

    def generate_qa(
        self,
        qa_fn: callable,
        article_name: str,
        caption: str,
        type_of_question: str,
        complete_return: bool = False,
    ) -> Union[dict, tuple[dict, str]]:
        """A method to generate a question-answer pair from a given figure caption."""

        # Retrieving the closest chunks to the caption
        out = self.collection.query(
            query_texts=caption,
            n_results=configs.NUM_RETRIEVED_CHUNKS,
            where={"$and": [{"type": "article"}, {"article_name": article_name}]},
        )

        # Building the context from the retrieved chunks
        chunks = out["documents"][0]
        metadata = out["metadatas"][0]
        chunk_indices = [metadata[i]["chunk_index"] for i in range(len(metadata))]
        chunks_copy = chunks.copy()
        chunks.sort(key=lambda x: chunk_indices[chunks_copy.index(x)])
        metadata_copy = metadata.copy()
        metadata.sort(key=lambda x: chunk_indices[metadata_copy.index(x)])
        context = "..." + "...".join(chunks) + "..."

        # Generating the question and answer
        qa_dict, llm1_response, llm2_response, llm3_response, total_tokens, total_price = qa_fn(
            caption, context, type_of_question, self.generator_model, self.content_editor_model, self.format_editor_model
        )
        if complete_return:
            return qa_dict, llm1_response, llm2_response, llm3_response, context, total_tokens, total_price
        return qa_dict
