# RadQBLLM
In this repository, we develop automated pipelines based on Large Language Models (LLMs) 
for building radiology question banks. This README file currently contains 
logs of the development process. When the code is ready for public use, this README file
will be updated with instructions on how to use the code.

<p> Developer: Pouria Rouzrokh, MD, MPH, MHPE [rouzrokh.pouria@mayo.edu]

## Logs

### 08/10/2023:

- Added the `main.py` file to run the pipeline using GradIO.
- Solved a bug that prevented correct query building and cosine similiarity calculation.

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