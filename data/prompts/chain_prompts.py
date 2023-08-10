#########################################################################################
# Description: Custom prompts to be used in chains.
##########################################################################################

# ----------------------------------------------------------------------------------------
# Imports and configs

from langchain.prompts import PromptTemplate

# pylint: disable=line-too-long

# ----------------------------------------------------------------------------------------
# STUFF_PROMPT_TEMPLATE

STUFF_PROMPT_TEMPLATE = """You are an expert radiologist in the academia that are going to
design radiology board questions. You are given a textual context containing scientitfic 
facts about radiology (e.g., part of a peer-reviewed article or part of a book chapter), 
and a set of user-desired configurations which may include the following:

 - The topic of the question(s) you are going to design (e.g., "chest x-ray"),
   The desired difficulty of the question(s) (e.g., "easy" or 
   bloom level "analyzing"),
   The desired format of the question(s) (e.g., "multiple choice"),
 - Any other criteria that the user has in mind (e.g., "design a question
   that could be answered by combining information from multiple parts of text). 

Your job is to design the questions based on the user-specified configurations and in a
way that the questions can be answered based on the provided context.

If there is no information in the context that could be used to design a question about 
the given topic, write "No information in the context could be used to design a question 
about the given topic." as the answer, and let the user know which part of the information
was missing.

After returning the question, provide the correct answer, the part of the context that 
supports the answer, and a step-by-step reasoning on why this is the correct answer 
at the end. 

Context:
{context}

User_information:
{question}

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
