# RadQG
RadQG is a small personal application to automatically generate radiology questions based
on retrieval augmentation of some user-input radiology texts. 
This repository contains the functionalities to load the Large Language Models (LLMs) 
and run the app for generating radiology questions. 

<p> Developer: Pouria Rouzrokh, MD, MPH, MHPE [rouzrokh.pouria@mayo.edu]

A brief demo of our app is available below:

<img src="demos/demo.gif" alt="demo" style="border: 1px solid black;">

## Run

- This repository is set up to mostly run with a GradIO application. Please run the 
`main.py` file to run the pipeline. You can change the local port number

- Feel free to change the parameters in the `configs.py` file according to your needs.
Don't forget to add your OpenAI API key to the `configs.py` file.

- If you want to run the application in a notebook or command line environment, please
see the `demo/demo.ipynb` file for an example.

## Development Logs

```python

### 10/28/2023:

- Added functionalities for HTML parsing and extracting the text, figures, and figure captions from the HTML files.
- Reorganized the repo to reflect the new version of RadQG which will be based on Q/A from RadioGraphics figures.
- Added the `html_utils.py` file to the `utils` folder.
- Merged all langchain utils into the `langchain_utils.py` file.
- Removed the `text_utils.py` file and merged its functionalities into the `general_utils.py` file.
- Added example HTML files to the `html_article` folder.
- Archived the previous version of RadQG GUI and demos in the `archived` folder.
- The `main.py` was renamed to `gradio_demo.py` and a `notebook_demo` was also added to both demonstrate the functionalities of the pipeline and to serve as a template for future development.
- The `demos` folder was deleted as the demo files are now available in the root directory.

### 10/27/2023:

- Set up pre-commit in the repo.
- Added `balck` and `black-nb` to the pre-commit hooks.

### 08/10/2023:

- Changed the name of the repository to "RadQG".
- Added the `main.py` file to run the pipeline using GradIO.
- Solved a bug that prevented correct query building and cosine similiarity calculation.
- Added the GIF demo of the pipeline to the `demo` folder.
- Added the `demo.mp4` and `demo.gif` files to the `demos` folder.

### 08/09/2023:

- Added the `configs.py` file.
- Added the `toy data`.
- Added the `chain_prompts.py`
- Added the `general_utils.py`
- Added the `rag.py`
- Added the `demo.ipynb`.
- Tested the pipeline on the toy data and in the demo notebook.
- Tested the functionality of the pipeline with "GPT-4" model.

### 08/08/2023:

- Created the repository.
- Cleaned and added the code for the text preprocessing.
- Added the `requirement.txt` baseline file. 