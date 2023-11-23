##########################################################################################
# Description: A script containing general functionalites for working with OpenAI API.
##########################################################################################

import json
import openai
import radqg.configs as configs
from radqg.prompts import get_radiologist_base_prompt
from radqg.prompts import get_educationist_base_prompt
from radqg.utils import count_tokens

# ----------------------------------------------------------------------------------------
# Configurations

openai.api_key = configs.OPENAI_API_KEY
price_for_gpt4_tokens_input: float = 0.03 / 1000  # $0.03 per 1000 tokens
price_for_gpt4_tokens_output: float = 0.06 / 1000  # $0.03 per 1000 tokens
price_for_gpt4turbo_tokens_input: float = 0.01 / 1000  # $0.01 per 1000 tokens
price_for_gpt4turbo_tokens_output: float = 0.03 / 1000  # $0.01 per 1000 tokens
price_for_chatgpt_tokens_input: float = 0.0010 / 1000  # $0.0015 per 1000 tokens
price_for_chatgpt_tokens_output: float = 0.0020 / 1000  # $0.0015 per 1000 tokens

# ----------------------------------------------------------------------------------------
# get_price_for_tokens


def get_price_for_tokens(total_tokens: int, model: str, type: str) -> float:
    """A function to calculate the price for a given number of tokens."""

    if model == "gpt-4" and type == "input":
        return total_tokens * price_for_gpt4_tokens_input
    elif model == "gpt-4" and type == "output":
        return total_tokens * price_for_gpt4_tokens_output
    elif model == "gpt-3.5-turbo-1106" and type == "input":
        return total_tokens * price_for_chatgpt_tokens_input
    elif model == "gpt-3.5-turbo-1106" and type == "output":
        return total_tokens * price_for_chatgpt_tokens_output
    elif model == "gpt-4-1106-preview" and type == "input":
        return total_tokens * price_for_gpt4turbo_tokens_input
    elif model == "gpt-4-1106-preview" and type == "output":
        return total_tokens * price_for_gpt4turbo_tokens_output
    else:
        raise ValueError("Invalid model or type.")


# ----------------------------------------------------------------------------------------
# emb_fn


def embed_fn(
    text_list_to_embed: list[str], model: str = configs.OPENAI_EMBEDDING_MODEL
) -> list[float]:
    """A function to embed a list of texts using OpenAI API."""

    response = openai.Embedding.create(model=model, input=text_list_to_embed)
    embeddings = [
        response["data"][i]["embedding"] for i in range(len(response["data"]))
    ]

    return embeddings


# ----------------------------------------------------------------------------------------
# get_qa_string


def get_qa_string(json_data, question_type):
    # Generate the required string
    if question_type.lower() == "mcq":
        formatted_string = f"Question stem:\n{json_data['question']}\nOptions:\n{json_data['options']}\nAnswer:\n{json_data['answer']}"
    else:
        formatted_string = (
            f"Question stem:\n{json_data['question']}\nAnswer:\n{json_data['answer']}"
        )

    return formatted_string


# ----------------------------------------------------------------------------------------
# qa


def qa(
    fignum,
    caption: str,
    context: str,
    type_of_question: str,
    radiologist_model: str = configs.OPENAI_RADIOLOGIST_MODEL,
    educationist_model: str = configs.OPENAI_EDUCATOR_MODEL,
    verbose=True,
) -> dict:
    """A function to generate a question from a figure caption using the OpenAI LLMs."""

    # To check the total number of tokens and budget used.
    total_cost = 0.0

    # To store the conversation log
    conversation_log = []

    # Asking for the initial question and answer generation
    radiologist_prompt = get_radiologist_base_prompt(
        figure_number=fignum,
        figure_caption=caption,
        context=context,
        type_of_question=type_of_question,
    )
    # Log the radiologist's prompt
    conversation_log.append(radiologist_prompt)

    radiologist_message = [
        {
            "role": "system",
            "content": "You are an expert radiologist and your job is to design a question for a radiology figure to be answered by a radiology resident.",
        },
        {"role": "user", "content": radiologist_prompt},
    ]
    educationist_response_str = None
    num_revised = 0

    while num_revised < 10:
        # Ask radiologist agent to generate or modify a question and answer
        radiologist_response = openai.ChatCompletion.create(
            model=radiologist_model,
            messages=radiologist_message,
            temperature=0.5,
            max_tokens=2000,
            frequency_penalty=0.0,
        )
        radiologist_response_str = radiologist_response.choices[0]["message"]["content"]

        # Check if the radiologist's response is valid JSON
        try:
            radiologist_response_json = json.loads(radiologist_response_str)
            qa_string = get_qa_string(radiologist_response_json, type_of_question)
        except:
            raise ValueError(f"Invalid JSON string: {radiologist_response_str}")

        # Log the radiologist's generated question and answering
        generation = f"Radiologist > {qa_string}"
        if verbose:
            print(f"\n---------Round {num_revised+1}---------\n")
            print(f"Radiologist output: {generation}")
        conversation_log.append(generation)

        # Calculate the total cost for the radiologist response
        total_cost += get_price_for_tokens(
            count_tokens(radiologist_message),
            model=configs.OPENAI_RADIOLOGIST_MODEL,
            type="input",
        ) + get_price_for_tokens(
            count_tokens(radiologist_response_str),
            model=configs.OPENAI_RADIOLOGIST_MODEL,
            type="output",
        )

        # Asking educationist agent to evaluate the question and answer
        if educationist_response_str is None:  # First time asking educationist
            educationist_prompt = get_educationist_base_prompt(
                caption, qa_string, type_of_question
            )
            educationist_message = [
                {
                    "role": "system",
                    "content": "You are an expert educationist and your job is to double-check the question and answer generated by the radiologist.",
                },
                {"role": "user", "content": educationist_prompt},
            ]
        else:  # Subsequent times asking educationist
            educationist_message = educationist_message + [
                {"role": "assistant", "content": educationist_response_str},
                {"role": "user", "content": qa_string},
            ]
        educationist_response = openai.ChatCompletion.create(
            model=educationist_model,
            messages=educationist_message,
            temperature=0.1,
            max_tokens=2000,
            frequency_penalty=0.0,
        )
        educationist_response_str = educationist_response.choices[0]["message"][
            "content"
        ]

        # Check if the educationist's response is valid JSON
        try:
            educationist_response_json = json.loads(educationist_response_str)
        except:
            raise ValueError(f"Invalid JSON string: {educationist_response_str}")

        # Check for the length of the radiologist's "answer" to the question
        if (
            count_tokens(radiologist_response_json["answer"]) < 50
            and type_of_question == "MCQ"
        ):
            if educationist_response_json["Status"] == "Pass":
                educationist_response_json["Status"] = "Fail"
                educationist_response_json[
                    "Message"
                ] = 'Your question and options are good, but your provided answer is too short. The answer must be a detailed response to the question in the "answer" key of the output `json` and it should explain why the right option is correct based on the figure caption and/or the provided context!'
            else:
                educationist_response_json[
                    "Message"
                ] += ' Also, your provided answer is too short. The answer must be a detailed response to the question in the "answer" key of the output `json` and it should explain why the right option is correct based on the figure caption and/or the provided context!'
        elif (
            count_tokens(radiologist_response_json["answer"]) > 50
            and type_of_question == "Short-Answer"
        ):
            if educationist_response_json["Status"] == "Pass":
                educationist_response_json["Status"] = "Fail"
                educationist_response_json[
                    "Message"
                ] = "Your question is good but your provided answer is too long. Please provide a shorter and more concise answer that is suitable for writing in a flash card!"
            else:
                educationist_response_json[
                    "Message"
                ] += " Also, your provided answer is too long. Please provide a shorter and more concise answer that is suitable for writing in a flash card!"

        # Log the educationist's response (which is the next prompt for the radiologist)
        feedback = f'Educationist > Status: {educationist_response_json["Status"]}: {educationist_response_json["Message"]}'
        if verbose:
            print(feedback)
        conversation_log.append(feedback)

        # Calculate the total cost for the educationist response
        total_cost += get_price_for_tokens(
            count_tokens(educationist_message),
            model=configs.OPENAI_EDUCATOR_MODEL,
            type="input",
        ) + get_price_for_tokens(
            count_tokens(educationist_response_str),
            model=configs.OPENAI_EDUCATOR_MODEL,
            type="output",
        )

        # If the educationist approves the question and answer, return the question and answer, total cost, and conversation log
        if educationist_response_json["Status"] == "Pass":
            return (
                radiologist_response_json,
                total_cost,
                conversation_log,
            )

        # If the educationist rejects the question and answer, add the educationist's feedback to the radiologist's next prompt
        num_revised += 1
        radiologist_message += [
            {"role": "assistant", "content": radiologist_response_str},
            {"role": "user", "content": educationist_response_json["Message"]},
        ]

    raise ValueError("The question and answer could not be revised in 10 rounds.")
