#########################################################################################
# Description: Custom prompts to be used in chains.
##########################################################################################

# ----------------------------------------------------------------------------------------
# Imports and configs

from langchain.prompts import PromptTemplate

# pylint: disable=line-too-long

# ----------------------------------------------------------------------------------------
# STUFF_PROMPT_TEMPLATE


def create_prompt(
    format: str = "", difficulty_level: str = "", criteria: str = ""
) -> PromptTemplate:
    context = "{context}"
    question = "{question}"

    STUFF_PROMPT_TEMPLATE = f"""You are an expert radiologist in the academia that are going to
  design radiology board questions. You are given a textual context containing scientitfic 
  facts about radiology (e.g., part of a peer-reviewed article or part of a book chapter), 
  and a set of user-desired configurations:

  - The topic of the question(s) you are going to design is: {question},
  - The desired difficulty of the question(s) you are going to design is: {difficulty_level},
  - The desired format of the question(s) (you are going to design is: {format},
  - Other criteria that you should follow for question designing are: {criteria}. 

  Your job is to design the questions based on the user-specified configurations and in a
  way that the questions can be answered based on the provided context.

  If there is no information in the context that could be used to design a question about 
  the given topic, write "No information in the context could be used to design a question 
  about the given topic." as the answer, and let the user know which part of the information
  was missing.
  
  If the user has not specified some of the configurations, you can choose them yourself,
  in a way that the other configurations are satisfied the best.

  After returning the question, provide the correct answer, the part of the context that 
  supports the answer, and a step-by-step reasoning on why this is the correct answer 
  at the end. 

  Context:
  {context}

  Output your answer in the following format:

  ```
  Question: (The question you designed)
  Answer: (The correct answer)
  Source: (the part of the context that supports the answer)
  Step by step reasoning on why this is the correct answer: ...
  ```

  Your answer in the above format:
  """

    STUFF_PROMPT = PromptTemplate(
        template=STUFF_PROMPT_TEMPLATE, input_variables=["question", "context"]
    )
    return STUFF_PROMPT
