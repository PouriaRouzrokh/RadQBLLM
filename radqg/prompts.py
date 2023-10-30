##########################################################################################
# Description: A script containing prompts for generating questions with the LLM.
##########################################################################################

# ----------------------------------------------------------------------------------------
# get_qa_prompt


def get_qa_prompt(figure_caption: str, context: str, type_of_question: str) -> str:
    assert type_of_question in ["MCQ", "Fill_in_the_Blanks", "Open-Ended", "Anki"]

    question_instructions = {
        "MCQ": """
        - Craft a question that is clear, concise, and directly related to the figure provided.\n
        - Provide five answer options, labeling them as A, B, C, D, and E. Ensure that only one of these is the correct answer.\n
        - Include all components of the question (stem and options) under the "question" key in the output dictionary.
        - Avoid using 'All of the above' or 'None of the above' as answer options to maintain the question's educational effectiveness.
        """,
        "Fill_in_the_Blanks": """ 
        - Formulate a question or statement that has one or more blanks for the trainee to complete.
        - Ensure that the blank spaces are logically placed, and their answers can be derived from analyzing the figure and the context.
        - Provide clear instructions on what is expected in the blanks, and ensure the required information is visible or inferable from the figure.
        """,
        "Open-Ended": """ 
        - Design a question that encourages a detailed response, analysis, or explanation from the trainee.
        - Ensure the question is open to thoughtful interpretation and analysis, promoting critical thinking.
        - Make sure the question is structured in a way that the answer can be derived through a detailed examination of the figure and context.
        """,
        "Anki": """ 
        - Create a question suitable for a flashcard, focusing on memorization and recall.
        - The question should be concise, focusing on a single fact or concept that is evident from the figure and context.
        - Ensure the answer to the question is brief and directly related to the question, facilitating easy memorization and recall for the trainee.
        """,
    }

    prompt = f"""
    ## Figure Caption
    {figure_caption}

    ## Context
    {context}

    ## Type of Question
    {type_of_question}

    ## Question Instructions
    {question_instructions[type_of_question]}

    ---

    Based on the figure caption and the context provided:

    - Develop a very difficult question that is directly related to the visual content in the figure.
    - If there is any diagnosis or imaging findings mentioned in the figure caption, ensure that you do not disclose it in the question.
    - The question may draw upon details from the context to enhance its educational value, but it must remain grounded in the figure.
    - Ensure the question aligns with the specified type and follows the instructions and guidelines given for that type.
    - The output should be a Python dictionary formatted as follows:
    {{'question': 'Your question here', 'answer': 'The answer here'}}
    
    
    Your output:
    
    """

    return prompt


# ----------------------------------------------------------------------------------------
# edit_qa_prompt


def edit_qa_prompt(figure_caption: str, qa_dict_string: str) -> str:
    prompt = f"""
    Read tge figure caption and the question-answer pair provided below and edit 
    the question to ensure it is not disclosing the diagnosis or imaging findings mentioned
    in the figure caption. 
    
    --- Input ---
    
    - Figure Caption: {figure_caption}
    - Question-Answer Pair: {qa_dict_string}
    
    --- Further Instructions ---
    
    - You should also ensure that the the figure is pointed to in the
    question but the figure number is not disclosed; e.g., instead of Figure 9.a you should
    use the phrase "the figure provided" or "the figure above".
    
    - Do not change the answer and only edit the question.
    
    - The output should be a Python dictionary formatted as follows:
    {{'question': 'Your modified question here', 'answer': 'The answer here'}}
    
    --- Example ---
    
    **Input:**
    
    - Figure Caption: 
    "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites."
    
    - Generated QA:
    'Question': 'A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen reveals a linear density in a loop of small bowel in the mid-left abdomen. Based on the imaging findings, which of the following is the most likely diagnosis? A) Small bowel obstruction B) Foreign body perforation C) Acute mesenteric ischemia D) Inflammatory bowel disease E) Small bowel ischemia' 
    'Answer': 'B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation.'
    
    **Exected Output:**
    
    'Question': 'A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen is provided. Based on the imaging findings, which of the following is the most likely diagnosis? A) Small bowel obstruction B) Foreign body perforation C) Acute mesenteric ischemia D) Inflammatory bowel disease E) Small bowel ischemia' 
    'Answer': 'B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation.'
    
    --- End of Example ---
    
    Please provide the modified question-answer pair below:
    
    """

    return prompt


# ----------------------------------------------------------------------------------------
# get_dict_formatting_prompt


def get_dict_formatting_prompt(qa_dict_string: str) -> str:
    prompt = f"""
    ---
    You will receive a string as input and should if it can be converted into a Python dictionary with the eval() function.
    If not, you should modify the string so that it can be converted into a dictionary.
    The output dictionary must contain at least two keys: 'question' and 'answer'.

    instructions:
        1. Examine the input string to ensure that it is formatted correctly to be interpreted as a dictionary by the eval() function.
        2. Ensure that the resulting dictionary contains the necessary keys: 'question' and 'answer'.
        3. Make sure that the ' and " characters are matched correctly.
        4. If there is any ' or " in the string except for the ones that are used to enclose the keys and values, replace them with `.
        5. If the input string is not correctly formatted or lacks the necessary keys, correct the format and/or include the missing elements.
        6. The output should be a string that, when passed to the eval() function, results in a valid dictionary with the necessary keys.

    example:
        input: "{{'question': 'What is radiologists' job?', 'answer: 'Reading medical images'}}"
        output: "{{'question': 'What is radiologists` job?', 'answer': 'Reading medical images'}}"
    ---

    Input String: "{qa_dict_string}"

    Output:
    "Your reformatted string here"
    """

    return prompt
