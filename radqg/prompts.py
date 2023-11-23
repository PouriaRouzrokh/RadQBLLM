##########################################################################################
# Description: A script containing prompts for generating questions with the LLM.
##########################################################################################

# ----------------------------------------------------------------------------------------
# get_radiologist_base_prompt


def get_radiologist_base_prompt(
    figure_number: str, figure_caption: str, context: str, type_of_question: str
) -> str:
    assert type_of_question in ["MCQ", "Short-Answer"]

    question_instructions = {
        "MCQ": """ 
        - Craft a question that is clear, concise, and directly related to the figure and context provided.\n
        - Provide five options, labeling them as A, B, C, D, and E. Ensure that only one of these is the correct answer.\n
        - Avoid using 'All of the above' or 'None of the above' among the options.
        - The question must require the examinee to deduce something from the figure, and it should not be answerable without viewing the figure!
        - The question must be based on a clinical scenario that is directly related to the visual content in the figure.
        - The answer must be a detailed response to the question in the "answer" key of the question output dictionary.
        """,
        "Short-Answer": """ 
        - Craft a question or statement related to the figure and context that has a one-word or short phrase response.
        - The question must be based on a clinical scenario that is directly related to the visual content in the figure.
        - The question must require the examinee to deduce something from the figure, and it should not be answerable without viewing the figure!
        - The question and answer must be useful for an Anki flashcard.
        - The question could also be a fill-in-the-blank question.
        - The answer must be a one-word or short phrase response in the "answer" key of the question output dictionary.
        """,
    }

    dict_instructions = {
        "MCQ": """
            {
            "question": "Your question here",
            "options": {
                "A": "Option A details here",
                "B": "Option B details here",
                "C": "Option C details here",
                "D": "Option D details here",
                "E": "Option E details here"
            },
            "answer": "The answer here"
            }
        """,
        "Short-Answer": """
            {
            "question": "Your question here",
            "answer": "The answer here"
            }
        """,
    }

    prompt = f"""
    
    -- Instructions --
    
    You will receive three inputs: a string containing the sub-figure number, a figure caption and a related context text to the figure.
    Based on these inputs, your job is to design a question answer in the following ```json``` format: {dict_instructions[type_of_question]}
    
    1) Develop a very difficult clinical scenario-based {type_of_question} question that is directly related to the visual content in the figure.
    2) You can also ask about the information provided in the context, but the question should still need the user to figure out the diagnosis or some imaging findings from the figure.
    3) The user is only going to see the subfigure with the number provided. Therefore, you must not ask about information that is provided in other subfigures. For example, if the subfigure number is Figure 15d, do not ask questions that need the user to see Figure 15a,b,c.
    4) Avoid creating questions that needs the user to see any figure other than the one you have the subfigure number for. For example, if the subfigure number is Figure 15d, do not ask questions that need the user to see Figure 15a,b,c.
    5) If there is a mentioning of an annotation (e.g., arrow, arrow head, star, etc.) in the figure caption, you can ask about it. However, you must not disclose the diagnosis or imaging findings mentioned in the figure caption.
    6) UNDER NO CIRCUMSTANCES, mention more than one clinical scenario in the stem of question.
    7) UNDER NO CIRCUMSTANCES, introduce more than one patient in the stem of the question; e.g., avoide stems that contain phrases like this: "A 67-year-old man and a 44-year-old man both present with epigastric pain..."
    8) UNDER NO CIRCUMSTANCES, disclose any diagnosis and imaging findings that are mentioned in the figure caption within the question. That's for the user to understand.
    9) UNDER NO CIRCUMSTANCES, describe the imaging findings that are mentioned in the figure caption within the question.
    10) Ensure the question aligns with the following instructions: {question_instructions[type_of_question]}

    
    -- Inputs --
    
    ## String Containing the Figure Number
    {figure_number}
    
    ## Figure Caption
    {figure_caption}

    ## Context
    {context}

    --- Your output ---
    
    Please provide the output ```json``` below:
    
    """

    return prompt


# ----------------------------------------------------------------------------------------
# get_educationist_base_prompt


def get_educationist_base_prompt(
    figure_caption: str, qa_dict_string: str, question_type: str
) -> str:
    question_instructions = {
        "MCQ": """ 
        - Craft a question that is clear, concise, and directly related to the figure and context provided.\n
        - Provide five options, labeling them as A, B, C, D, and E. Ensure that only one of these is the correct answer.\n
        - Avoid using 'All of the above' or 'None of the above' among the options.
        - The question must require the examinee to deduce something from the figure, and it should not be answerable without viewing the figure!
        - The question must be based on a clinical scenario that is directly related to the visual content in the figure.
        """,
        "Short-Answer": """ 
        - Craft a question or statement related to the figure and context that has a one-word or short phrase response.
        - The question must be based on a clinical scenario that is directly related to the visual content in the figure.
        - The question must require the examinee to deduce something from the figure, and it should not be answerable without viewing the figure!
        - The question and answer must be useful for an Anki flashcard.
        - The question could also be a fill-in-the-blank question.
        """,
    }

    example_instructions = {
        """MCQ""": """
        
        -- Example 1 --
        
        **Input:**
    
        - Figure Caption: 
        "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites."
        
        - Question-answer:
        "question stem": "A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen reveals a linear density in a loop of small bowel in the mid-left abdomen. Based on the imaging findings, which of the following is the most likely diagnosis? 
        "options": "A) Small bowel obstruction B) Foreign body perforation C) Acute mesenteric ischemia D) Inflammatory bowel disease E) Small bowel ischemia"
        "answer": "B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation."
        
        **Exected Output:**
        {{"Status": "Fail", "Message": "The sentence 'A non-enhanced CT scan of the abdomen reveals a linear density in a loop of small bowel in the mid-left abdomen' is revealing an imaging finding; This is the job of the examenee to guess the imaging findnigs. Revise your quesiton like like this: A non-enhanced CT scan of the abdomen is provided. Regnerate the ```json`` with modified question stem, and if needed, quesiton options or answer."}}
        
        -- Example 2 --
         **Input:**
    
        - Figure Caption: 
        "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites."
        
        - Question-answer:
        "question stem": "A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen is provided in figure 2. Based on the imaging findings, which of the following is the most likely diagnosis? 
        "options": "A) Small bowel obstruction B) Foreign body perforation C) Acute mesenteric ischemia D) Inflammatory bowel disease E) Small bowel ischemia"
        "answer": "B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation."
        
        -- Example 3 --
        
        ** Input: **
        
        - Figure Caption: 
        "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites.
        
        - Question-answer:
        "question stem": "Based on the imaging findings, what is the most likely diagnosis? 
        "options": "A) Small bowel obstruction B) Foreign body perforation C) Acute mesenteric ischemia D) Inflammatory bowel disease E) Small bowel ischemia"
        "answer": "B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation."
        
        **Exected Output:**
        {{"Status": "Fail", "Message": "Your question is not based on a clinical scenario. You can change it like this: A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen is provided. Based on the imaging findings, what is the most likely diagnosis? Regenerate the ```json``` with modified question stem, and if necessary, modified options and answer."}}

        -- Example 4 --
        
        **Input:**
    
        - Figure Caption: 
        "Figure 11d. Peptic stricture. (a) Photomicrograph of the esophagus shows diffuse deposition of pink-stained fibrous tissue (curved arrows) in the splayed muscularis propria (straight arrow). There is lymphocytic infiltration of the lamina propria and epithelium (circle), which is now composed of glands containing goblet cells (arrowheads), indicative of chronic inflammation and intestinal metaplasia. (H-E stain; original magnification, ×40.) (b) Photograph of a gross specimen shows a focal area of luminal narrowing (arrows) just above the squamocolumnar junction. (c) Endoscopic image shows a focal area of smooth narrowing (arrow) adjacent to nodular reddened mucosa, consistent with erosive esophagitis. Note the finger-like projections of salmon-colored velvety mucosa (*), consistent with areas of Barrett esophagus. (d) Prone single-contrast radiograph shows smooth concentric narrowing of the midesophagus (arrows) with a tiny outpouching of contrast material (intramural pseudodiverticulosis)."
        
        - Question-answer:
        "question stem": "A patient with a history of GERD presents with dysphagia. What is the arrow on the gross specimen provided in the figure pointng to?" 
        "Options" : "A) Luminal narrowing B) Barrett esophagus C) Intramural pseudodiverticulosis D) Erosive esophagitis E) Intestinal metaplasia"
        "answer": "Luminal narrowing"
        
        **Exected Output:**
        {{"Status": "Fail", "Message": "Your question, options, and answer are wrong. Note that the user is seeing figure 15d, and accoding to the caption, figure 15d is the image of a single-contrast radiograph, not a gross specimen. Revise your answer and regenerate the ```json```. If necessary, revise the question stem."}}

        """,
        """Short-Answer""": """  
       -- Example 1 --
        
        **Input:**
    
        - Figure Caption: 
        "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites."
        
        - Question-answer:
        "question stem": "A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen reveals a linear density in a loop of small bowel in the mid-left abdomen. Based on the imaging findings, what is the most likely diagnosis? 
        "answer": "Foreign body perforation"
        
        **Exected Output:**
        {{"Status": "Fail", "Message": "The sentence 'A non-enhanced CT scan of the abdomen reveals a linear density in a loop of small bowel in the mid-left abdomen' is revealing an imaging finding; This is the job of the examenee to guess the imaging findnigs. Revise your quesiton like like this: A non-enhanced CT scan of the abdomen is provided. Regnerate the ```json`` with modified question stem, and if needed, a modified answer."}}
        
        -- Example 2 --
         **Input:**
    
        - Figure Caption: 
        "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites."
        
        - Question-answer:
        "question stem": "A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen is provided in figure 2. Based on the imaging findings, what is the most likely diagnosis? 
        "answer": "Foreign body perforation"
        
        **Exected Output:**
        {{"Status": "Fail", "Message": "The sentence 'A non-enhanced CT scan of the abdomen is provided in figure 2' is mentioning the number of the figure. You must not mention the number of figure in the question stem. Revise your quesiton like like this: A non-enhanced CT scan of the abdomen is provided in the figure. Regnerate the ```json`` with modified question stem, but do not change the answer."}}
        
        -- Example 3 --
        
        **Input:**
    
        - Figure Caption: 
        "Figure 18b.Surgically-confirmed foreign body perforation in two patients.(a)Axial nonenhanced CT image in a 16-year-old girl who ingested a small bone shows a linear density in a loop of small bowel in the mid-left abdomen (arrow), extending through the bowel wall, with perforation that was later confirmed at surgery.(b)Axial intravenous contrast-enhanced CT image in a 78-year-old woman with right-sided abdominal pain shows a thin bone (arrow) causing focal small-bowel perforation, with associated mural edema, mesenteric fat stranding, and ascites."
        
        - Question-answer:
        "question stem": "Based on the imaging findings, what is the most likely diagnosis? 
        "answer": "B) Foreign body perforation. The CT scan findings of a linear density in a loop of small bowel extending through the bowel wall are consistent with a foreign body perforation. The history of the patient having eaten a chicken meal earlier in the day supports the possibility that a small bone might have been ingested and caused the perforation."
        
        **Exected Output:**
        {{"Status": "Fail", "Message": "Your question is not based on a clinical scenario. You can change it like this: A 16-year-old girl presents to the emergency department with severe left-sided abdominal pain. A non-enhanced CT scan of the abdomen is provided. Based on the imaging findings, what is the most likely diagnosis? Regenerate the ```json``` with modified question stem, and if necessary, modified options and answer."}}

        -- Example 4 --
        
         **Input:**
    
        - Figure Caption: 
        "Figure 11d. Peptic stricture. (a) Photomicrograph of the esophagus shows diffuse deposition of pink-stained fibrous tissue (curved arrows) in the splayed muscularis propria (straight arrow). There is lymphocytic infiltration of the lamina propria and epithelium (circle), which is now composed of glands containing goblet cells (arrowheads), indicative of chronic inflammation and intestinal metaplasia. (H-E stain; original magnification, ×40.) (b) Photograph of a gross specimen shows a focal area of luminal narrowing (arrows) just above the squamocolumnar junction. (c) Endoscopic image shows a focal area of smooth narrowing (arrow) adjacent to nodular reddened mucosa, consistent with erosive esophagitis. Note the finger-like projections of salmon-colored velvety mucosa (*), consistent with areas of Barrett esophagus. (d) Prone single-contrast radiograph shows smooth concentric narrowing of the midesophagus (arrows) with a tiny outpouching of contrast material (intramural pseudodiverticulosis)."
        
        - Question-answer:
        "question stem": "A patient with a history of GERD presents with dysphagia. What is the arrow on the gross specimen provided in the figure pointng to?" 
        "answer": "Luminal narrowing"
        
        **Exected Output:**
        {{"Status": "Fail", "Message": "Your question and answer are wrong. Note that the user is seeing figure 15d, and accoding to the caption, figure 15d is the image of a single-contrast radiograph, not a gross specimen. Revise your answer and regenerate the ```json```. If necessary, revise the question stem."}}
     
        """,
    }

    prompt = f"""
    
    -- Instructions --
    
    You will receive two inputs: a figure caption, and a question-answer created by a radiologist for the figure.
    Your job is to check the question for the following criteria and output a ```JSON``` with two keys:
    "Status": "Pass" or "Fail"
    "Message": "Your step-by-step evaluation of the current question and instructions for the radiologist to correct it, if needed."
    Always tell the radiologist to regenrate a ```json``` and provide instructions on what keys to change or not to change.
    
    Here are the criteria:
    
    1) UNDER NO CIRCUMSTANCES, the question stem must disclose any SPECIFIC diagnosis or imaging findings that are mentioned in the figure caption. Note that: 
    - The question stem IS ALLOWED to generally refer to imaging findings like: "in the context of radiographic findings provided in the figure..., based on the CT findings..., etc." but It must not say what the exact findings are.
    - The question stem IS ALLOWED to mention the clinical signs, symptoms, lab data, or the history of the patient.
    - The question options or the answer ARE ALLOWED to mention the diagnosis or imaging findings.
    - The question stem IS ALLOWED to ask the examinee about what an annotation (e.g., arrow, arrow head, bracket, star, etc.) in the figure is referring to, but it must not disclose the answer to that in the question stem.
        
    2) The figure must be referenced in the question but the figure number must not be disclosed; 
    e.g., `Figure 9.a` must change to a phrase like "the figure provided" or "the figure above". 
    
    3 Although the figure number must not be disclosed, the question must still be designed from the text
    related to the figure number in the figure caption. For example, if the figure caption includes information 
    about figures 15a,b,c,d, but your figure number is 15d, you must still design the question from the text related to 15d. 
    Also the user must not need to see any other figure than the one with the specified figure number to be able to answer the question.
    
    4) The question must be based on a clinical scenario that is directly related to the visual content in the figure.

    5) The question must meet the following criteria: {question_instructions[question_type]}
        
    --- Example ---
    
    {example_instructions[question_type]}
    
    --- Input ---
        
    - Figure Caption: 
    {figure_caption}
    
    - Question-Answer Pair: 
    {qa_dict_string}
    
    --- Your output ---
    Please provide the output ```json``` below:
    
    """

    return prompt
