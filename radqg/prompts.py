##########################################################################################
# Description: A script containing prompts for generating questions with the LLM.
##########################################################################################

# ----------------------------------------------------------------------------------------
# get_generator_prompt


def get_generator_prompt(
    figure_caption: str, context: str, type_of_question: str
) -> str:
    assert type_of_question in ["MCQ", "Short-Answer", "Long-Answer"]

    question_instructions = {
        "MCQ": """ 
        - Craft a question that is clear, concise, and directly related to the figure and context provided.\n
        - Provide five options, labeling them as A, B, C, D, and E. Ensure that only one of these is the correct answer.\n
        - Include all five options in the output dictionary under the "question" key.\n
        - Avoid using 'All of the above' or 'None of the above' among the options.
        - Provide a detailed response to the question in the "answer" key of the question output dictionary.
        """,
        "Short-Answer": """ 
        - Craft a question or statement related to the figure and context that has a one-word or short phrase response.
        - The question and answer should emphasize recall and be useful for an Anki flashcard.
        """,
        "Long-Answer": """ 
        - Craft a question related to the figure and context that encourages a detailed response, analysis, or explanation from the trainee.
        - Provide a detailed response to the question in the "answer" key of the question output dictionary.
        """,
    }

    prompt = f"""
    
    -- Instructions --
    
    You will receive two inputs: a figure caption and a related context text to the figure.
    Based on these inputs, your job is to do the following tasks:
    
    1) Develop a very difficult clinical scenario-based {type_of_question} question that is directly related to the visual content in the figure.
    2) You can also ask about the information provided in the context, but the question should still need the user to figure out the diagnosis or some imaging findings from the figure.
    3) You must not mention more than one clinical scenario in the stem of question.
    4) You must not introduce more than one patient in the stem of the question; e.g., avoide stems that contain phrases like this: "A 67-year-old man and a 44-year-old man both present with epigastric pain..."
    5) You must not disclose any diagnosis and imaging findings that are mentioned in the figure caption within the question.
    6) You must not describe the imaging findings that are mentioned in the figure caption within the question.
    7) Ensure the question aligns with the following instructions: {question_instructions[type_of_question]}
    8) Your output should be in the following dictionary format if the question is MCQ:
       {{'question': 'Your question here (including the choices if MCQ question)', 'options': "The A), B), C), D), and E) options here', 'answer': 'The answer here'}}
       and the in the follwing dictionary format if the question is not MCQ:
       {{'question': 'Your question here (including the choices if MCQ question)', 'answer': 'The answer here'}}
    
    -- Inputs --
    
    ## Figure Caption
    {figure_caption}

    ## Context
    {context}

    --- Your output ---
    
    Please provide the output dictionary below:
    
    """

    return prompt


# ----------------------------------------------------------------------------------------
# get_contenteditor_prompt


def get_contenteditor_prompt(
    figure_caption: str, 
    qa_dict_string: str,
    question_type: str
    ) -> str:
    prompt = f"""
    
    -- Instructions --
    
    You will receive three inputs: a figure caption, a question-answer dictionary, a question type. 
    Your job is to do the following tasks:
    
    1) Read the figure caption and the question-answer dictionary and edit or change the question to ensure it is not disclosing the diagnosis and imaging findings mentioned
    in the figure caption. If the original question is describing some imaging findings, remove them.
    
    2) Do not change the "answer" or "options" key in the dictionary and only edit the "question".
    
    3) Ensure that the figure is referenced in the question but the figure number is not disclosed; e.g., instead of Figure 9.a you should use the phrase "the figure provided" or "the figure above".
    
    4) Ensure that the question matches the type of question: 
        - If the question is an MCQ, ensure that the question has five options mentioned in the "options" key of the dictionary.
        - If the question is a fill-in-the-blanks question, ensure that the text in the question has one or more blanks (denoted by "-----") in it.
        - If the question is an open-ended question, ensure that the question is open-ended and does not have any blanks or options.
    
    5) Your output should be in the following format if the question type is MCQ:
    {{'question': 'Your question here (including the choices if MCQ question)', 'options': "The A), B), C), D), and E) options here', 'answer': 'The answer here'}}
    and the in the follwing format if the question type is not MCQ:
    {{'question': 'Your question here', 'answer': 'The answer here'}}
    
    --- Example 1 ---
    
    **Input:**
    
    - Figure Caption: 
    "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites."
    
    - Question-answer dictionary:
    {{"question": "A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen reveals a linear density in a loop of small bowel in the mid-left abdomen. Based on the imaging findings, which of the following is the most likely diagnosis? 
    "options": "A) Small bowel obstruction B) Foreign body perforation C) Acute mesenteric ischemia D) Inflammatory bowel disease E) Small bowel ischemia"
    "answer": "B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation."}}
    
    - Question type:
    MCQ
    
    **Exected Output:**
    
    {{"question": "A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen is provided. Based on the imaging findings, which of the following is the most likely diagnosis?", 
    "options": "A) Small bowel obstruction B) Foreign body perforation C) Acute mesenteric ischemia D) Inflammatory bowel disease E) Small bowel ischemia",
    "answer": "B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation."}}
    
    
    --- Example 2 ---
    
    **Input:**
    
    - Figure Caption: 
    "Figure 4b.Acute mesenteric venous ischemia in a 43-year-old woman with positive test results for lupus anticoagulant who presented with acute abdominal pain. Axial(a)and coronal(b)intravenous contrast-enhanced CT images show an occlusive thrombus within the superior mesenteric vein (white arrow) and its jejunal branches, with several loops of thick-walled jejunum (arrowheads) that show marked mural edema and a target appearance of the bowel, which is poorly enhancing. Note the mesenteric edema and fluid (black arrow inb), common features of mesenteric venous ischemia."
    
    - Question-answer dictionary:
    {{"question": "A 43-year-old woman with a positive test for lupus anticoagulant presents with acute abdominal pain. The figure provided shows CT images with several loops of thick-walled jejunum and mesenteric edema. What could be a possible cause for her condition based on the imaging findings?", 
    "answer": "Acute mesenteric venous ischemia."}}
    
    - Question type:
    Short-Answer
    
    **Exected Output:**
    
    {{"question': "A 43-year-old woman with a positive test for lupus anticoagulant presents with acute abdominal pain. The figure provided shows CT images for the patient. What could be a possible cause for her condition?",
    "answer": "Acute mesenteric venous ischemia."}}
   
    
    --- Input ---
        
    - Figure Caption: 
    {figure_caption}
    
    - Question-Answer Pair: 
    {qa_dict_string}
    
    - Type of Question:
    {question_type}
    
    --- Your output ---
    Please provide the modified question-answer pair below:
    
    """

    return prompt


# ----------------------------------------------------------------------------------------
# get_formateditor_prompt


def get_formateditor_prompt(qa_dict_string: str) -> str:
    prompt = f"""
    ---
    Read the provided input string below and modify it so that it can be converted into a Python dictionary with the eval() function.
    The output dictionary must contain at least two keys: 'question' and 'answer', and it could optionally have an "options" key if the question is an MCQ.
    
    Input String: "{qa_dict_string}"

    --- Further instructions ---
        1. Examine the input string to ensure that it is formatted correctly to be interpreted as a dictionary by the eval() function.
        2. Ensure that the resulting dictionary contains the necessary keys: 'question' and 'answer'.
        3. If the question is an MCQ, ensure it also contains the key `options`.
        3. Make sure that the ' and " characters are matched correctly.
        4. If there is any ' or " in the string except for the ones that are used to enclose the keys and values, replace them with `.
        5. If the input string is not correctly formatted or lacks the necessary keys, correct the format and/or include the missing elements.
        6. The output should be a Python dictionary with "question", "answer", and an optional "options" keys.

    --- Example ---
        input: "{{'question': 'What is radiologists' job?', 'answer: 'Reading medical images'}}"
        output: {{'question': 'What is radiologists` job?', 'answer': 'Reading medical images'}}
    ---

    Please provide the output below:
    """

    return prompt
