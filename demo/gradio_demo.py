##########################################################################################
# Description: GUI app code based on GradIO demonstrating the pipeline.
##########################################################################################

import os
import sys

sys.path.append("../")
import gradio as gr
import radqg.configs as configs
from radqg.generator import Generator
from radqg.llm.openai import embed_fn as openai_embed_fn
from radqg.llm.openai import qa as openai_qa

# ----------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# initialize_app


def initialize_app(openai_api_key, chnuk_size, chunk_overlap, k):
    global chain
    global retriever

    docs = get_all_chunks(
        file_paths, chunk_size=int(chnuk_size), chunk_overlap=int(chunk_overlap)
    )
    embedding_llm = OpenAIEmbeddings(
        model=configs.EMBEDDING_MODEL, openai_api_key=openai_api_key
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
        k=int(k),
        fetch_k=configs.FETCH_K,
        contextual_compressor=configs.COMPRESSOR,
    )
    return gr.Textbox.update("The app is initialized.")


# ----------------------------------------------------------------------------------------
# initialize_qbank


def initialize_qbank(openai_api_key, chnuk_size, chunk_overlap, k):
    global chain
    global retriever

    docs = get_all_chunks(
        file_paths, chunk_size=int(chnuk_size), chunk_overlap=int(chunk_overlap)
    )
    embedding_llm = OpenAIEmbeddings(
        model=configs.EMBEDDING_MODEL, openai_api_key=openai_api_key
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
        k=int(k),
        fetch_k=configs.FETCH_K,
        contextual_compressor=configs.COMPRESSOR,
    )
    return gr.Textbox.update("The app is initialized.")


# ----------------------------------------------------------------------------------------
# load_articles


def load_articles():
    file_names = []
    full_file_names = []
    for file in os.listdir(configs.TOY_DATA_DIR):
        if file.endswith(".html"):
            file_name = file.split(" _ RadioGraphics.html")[0]
            file_names.append(file_name)
            full_file_names.append(file)
    return file_names, full_file_names


# ----------------------------------------------------------------------------------------
# generate_question


def generate_question(
    model,
    tempreture,
    openai_api_key,
    topic,
    format,
    level_of_difficulty,
    other_characteristics,
):
    chain = retrieval_qa(
        retriever,
        model=model,
        temperature=float(tempreture),
        chain_type=configs.CHAIN_TYPE,
        prompt_keywords={
            "format": format,
            "difficulty_level": level_of_difficulty,
            "criteria": other_characteristics,
        },
        openai_api_key=openai_api_key,
    )

    result = chain({"query": topic})

    return gr.Textbox.update(result["result"])


# ----------------------------------------------------------------------------------------
# GradIO App
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# run_gui


def run_gui():
    global chain

    try:
        gr.close_all()
    except:
        pass

    css_style = """
    #accordion {font-weight: bold; font-size: 18px; background-color: AliceBlue}
    #model {background-color: MintCream}
    #central {font-weight: normal; text-align: center; font-size: 16px}
    #normal {font-weight: normal; font-size: 16px}
    #generate_button {background-color: LightSalmon; font-size: 18px}
    #upload_button {background-color: LightSalmon; font-size: 18px}
    #articles {background-color: MintCream; font-size: 16px}
    #title {
        text-align: center;
        width: 90%;
        margin: auto !important;
        font-style: italic;
    }
    #link_out, #authors_list, #authors_affiliation{
        text-align: center;
        min-height: none !important;
    }
    """

    with gr.Blocks(css=css_style) as app:
        gr.Markdown(
            "# RadQG: Radiology Question Generator with Large Language Models",
            elem_id="title",
        )
        gr.HTML(
            "Pouria Rouzrokh, MD, MPH, MHPE<sup>1,2</sup>, Mark M. Hammer, MD<sup>1</sup>, Christine (Cooky) O. Menias, MD<sup>1</sup>",
            elem_id="authors_list",
        )
        gr.HTML(
            """1. RadioGraphics Trainee Editorial Board (RG Team)<br>\
            2. Mayo Clinic Artificial Intelligence Lab, Department of Radiology,\
                Mayo Clinic""",
            elem_id="authors_affiliation",
        )

        with gr.TabItem("Welcome"):
            gr.Markdown(
                """
                    Welcome to RadQG!
                    <br>Here you can use large language artificial intelligence (AI) models to test your knowledge of radiology. RadQG is 
                    designed to generate questions based on the content and figures of the article published at the [RadioGraphics](https://pubs.rsna.org/journal/radiographics) jorunal.
                    
                    >**Important disclaimers**:
                        <ol>
                            <li style="padding-left: 20px;">RadQG is designed to work with the articles published at RadioGraphics, and not articles from other journals.</li>
                            <li style="padding-left: 20px;">The current large language models (LLMs) that RadQG uses are provided by [OpenAI](https://openai.com) and are not free to use. Using the default models, you will spend ~0.17 USD per question generated.</li>
                            <li style="padding-left: 20px;">The text and the figure captions of the articles you upload will be sent to OpenAI servers for processing. OpenAI claims that it does not store the user inputs; still, please make sure that you have the right to use the articles you upload. No figures will be processed by OpenAI.</li>
                            <li style="padding-left: 20px;">Although we gurantee the overal quality of the questions, we do not gurantee the quality and correctness of each individual questions. Please use your own judgement and check the original articles as references for validating the questions.</li>
                            <li style="padding-left: 20px;">RadQG is not a replacement for the RadioGraphics journal. It is a tool to help you test your knowledge of radiology.</li>
                            <li style="padding-left: 20px;">If this is the first time you are using this app, please visit the **"About"** tab to learn how to use the app.</li>
                        </ol>    
                    """,
                elem_id="normal",
            )

            with gr.Row():
                gr.Markdown(
                    """ 
                    To start, please enter your **OpenAI API key** and click on "Start the App!" button below.<sup>*</sup>.
                    <br><sup>*</sup> If you do not have an OpenAI API key or do not know what it is, please visit [here](https://openai.com/blog/openai-api).
                    """
                )

            with gr.Row():
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    value=configs.OPENAI_API_KEY,
                    interactive=True,
                    type="password",
                    elem_id="normal",
                )

            with gr.Row():
                gr.Markdown(
                    """ 
                    Optionally, you can also configure the RadQG advanced settings before starting the app.
                    """
                )

            with gr.Row():
                with gr.Accordion(
                    label="Advanced Settings", elem_id="accordion", open=False
                ):
                    with gr.Row():
                        generator_model = gr.Dropdown(
                            choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"],
                            label="Generator LLM:",
                            value="gpt-4",
                        )

                    with gr.Row():
                        content_editor_model = gr.Dropdown(
                            choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"],
                            label="Content Editor LLM:",
                            value="gpt-4",
                        )

                    with gr.Row():
                        format_editor_model = gr.Dropdown(
                            choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"],
                            label="Format Editor LLM:",
                            value="gpt-3.5-turbo",
                        )

                    with gr.Row():
                        chnuk_size = gr.Number(
                            label="Chunk Size (integer number):",
                            value=configs.CHUNK_SIZE,
                            interactive=True,
                            elem_id="normal",
                        )
                    with gr.Row():
                        chunk_overlap = gr.Number(
                            label="Chunk Overlap (integer number):",
                            value=configs.CHUNK_OVERLAP,
                            interactive=True,
                            elem_id="normal",
                        )
                    with gr.Row():
                        num_retrieved_chunks = gr.Number(
                            label="Nubmer of article chunks to retrieve (integer number):",
                            value=configs.NUM_RETRIEVED_CHUNKS,
                            interactive=True,
                            elem_id="normal",
                        )
            with gr.Row():
                app_button = gr.Button("Start the App!", elem_id="generate_button")

            with gr.Row():
                log_textbox1 = gr.Textbox(
                    "Currently, the app is not initialized.",
                    label="App Status",
                )

            # Events
            app_button.click(
                initialize_app,
                [openai_api_key, chnuk_size, chunk_overlap, num_retrieved_chunks],
                [log_textbox1],
            )

        with gr.TabItem("Question Bank"):
            with gr.Row():
                gr.Markdown(
                    """ 
                        Please select the articles that you want to use as the source for question generation.
                        > **Note**: This app is a demo version of the RadQG and contains a limited number of articles from the RadioGraphics top 10 gastrointestinal list to choose from. 
                        """
                )

            # article_checkboxes = gr.CheckboxGroup(
            #     choices=load_articles()[0],
            #     label="Articles:",
            #     interactive=True,
            #     container=False,
            #     elem_id="articles",
            # )

            with gr.Row():
                gr.Markdown("Articles:", elem_id="normal")
            with gr.Group():
                article_checkboxes = []
                for article in load_articles()[0]:
                    with gr.Row():
                        article_checkboxes.append(
                            gr.Checkbox(
                                label=article,
                                value=True,
                                interactive=True,
                                elem_id="articles",
                            )
                        )

            with gr.Row():
                gr.Markdown(
                    """ 
                    Optinally, you can also provide a topic or keyword to see more questions related to that topic.
                    """
                )
            with gr.Row():
                topic = gr.Textbox(
                    label="Topic of the question:", interactive=True, elem_id="normal"
                )
            with gr.Row():
                gr.Markdown(
                    """
                    > **Note**: If the topic is left blank, the app will generate random questions from the entire content of articles.
                    """
                )
            with gr.Row():
                qbank_button = gr.Button(
                    "Set up the question bank!", elem_id="generate_button"
                )

            with gr.Row():
                log_textbox2 = gr.Textbox(
                    "Currently, the app is not initialized.",
                    label="App Status",
                )

            # Events
            qbank_button.click(
                initialize_qbank,
                [topic, *article_checkboxes],
                [log_textbox2],
            )

        with gr.TabItem("Question Generator"):
            # Elements
            with gr.Row():
                gr.Markdown(
                    "Please enter the type of question you desire:",
                    elem_id="normal",
                )
            with gr.Row():
                question_type = gr.Dropdown(
                    choices=["MCQ", "Fill_in_the_Blanks", "Open-Ended", "Anki"],
                    value="MCQ",
                    label="Qestion type:",
                    interactive=True,
                    elem_id="normal",
                )
            with gr.Row():
                generate_button = gr.Button(
                    "Generate a question!", elem_id="generate_button"
                )
            with gr.Row():
                image_box = gr.Image(
                    value="../data/fig_placeholder.png",
                    label="Figure",
                    interactive=False,
                    width=500,
                    height=500,
                )
            with gr.Row():
                question_box = gr.Textbox(
                    "Currently, no questions are generated.",
                    label="Generated a question!",
                    elem_id="normal",
                    max_lines=40,
                )
            with gr.Accordion(
                label="Click to see the answer!", elem_id="accordion", open=False
            ):
                with gr.Row():
                    answer_box = gr.Textbox(
                        "Currently, no answer is generated.",
                        label="Generated Answer",
                        elem_id="normal",
                        max_lines=40,
                    )

            # Events
            generate_button.click(
                generate_question,
                [question_type],
                [image_box, question_box, answer_box],
            )

        with gr.TabItem("About"):
            gr.Markdown(
                """To be developed...!""",
                elem_id="normal",
            )

    try:
        app.close()
        gr.close_all()
    except:
        pass

    app.queue(
        status_update_rate="auto",
    )

    out = app.launch(
        # max_threads=4,
        share=configs.GR_PUBLIC_SHARE,
        inline=False,
        show_api=False,
        show_error=True,
        server_port=configs.GR_PORT_NUMBER,
        server_name=configs.GR_SERVER_NAME,
    )
    return out


# ----------------------------------------------------------------------------------------
# main

if __name__ == "__main__":
    _ = run_gui()
