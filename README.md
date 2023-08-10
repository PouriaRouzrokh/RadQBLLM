# RadQG
RadQG is a small personal application to automatically generate radiology questions based
on retrieval augmentation of some user-input radiology texts. 
This repository contains the functionalities to load the Large Language Models (LLMs) 
and run the app for generating radiology questions. 

<p> Developer: Pouria Rouzrokh, MD, MPH, MHPE [rouzrokh.pouria@mayo.edu]

A brief demo of our app is available below:

![demo](demos/demo.gif)


## Run

- This repository is set up to mostly run with a GradIO application. Please rund the 
`main.py` file to run the pipeline. You can change the local port number

- Feel free to change the parameters in the `configs.py` file according to your needs.
Don't forget to add your OpenAI API key to the `configs.py` file.

- If you want to run the application in a notebook or command line environment, please
see the `demo/demo.ipynb` file for an example.

## Logs

```python
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