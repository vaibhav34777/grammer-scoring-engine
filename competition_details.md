# SHL Hiring Assessment - Grammar Scoring Engine

## Overview
The goal of this competition is to develop a Grammar Scoring Engine for spoken data samples. 
The task is to build a machine learning/AI model that takes a 45-60 second audio file (`.wav`) as input and outputs a continuous Grammar Score ranging from 0 to 5.

## Dataset
- **Training Data:** 769 audio samples. (`train.csv` contains file names and labels).
- **Testing Data:** 216 audio samples. (`test.csv` contains file names).
- **Format:** Audio files are in standard `.wav` format.

## Grammar Score Rubric (0 to 5 Continuous Score)
| Score | Description |
| :---: | :--- |
| **1** | Struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns. |
| **2** | Limited understanding of sentence structure and syntax. Uses simple structures but consistently makes basic mistakes; sentences may be incomplete. |
| **3** | Decent grasp of sentence structure but makes errors in grammatical structure, OR decent grasp of grammatical structure but makes errors in sentence syntax/structure. |
| **4** | Strong understanding of sentence structure and syntax. Good control of grammar. Occasional minor errors that don't lead to misunderstandings; person can correct most of them. |
| **5** | High grammatical accuracy and adept control of complex grammar. Uses grammar accurately/effectively, seldom making noticeable mistakes. Handles complex language structures well. |

## Evaluation
- **Leaderboard Metrics:** Pearson Correlation and RMSE (Root Mean Squared Error).
- **Evaluation Criteria (for the notebook/code):**
  - **Correctness:** Does the solution work as expected?
  - **Code Quality:** Is the code clean, well-structured, and documented?
  - **Performance:** How well does the model perform on the test dataset?
  - **Interpretability:** Are the results well-explained with relevant visualizations?

## Submission Requirements
1. **Jupyter Notebook:** 
   - Well-documented and commented code.
   - Includes a brief report explaining the approach, preprocessing steps, pipeline architecture, and evaluation results.
   - **COMPULSORY:** You must add the RMSE score of the training data in your final submission notebook.
   - Must compute task-relevant metrics to benchmark model performance.
   - Add visualizations wherever applicable.
2. **GitHub Repository:**
   - Must contain your code and be **publicly accessible**.
3. **Qualtrics Form:** 
   - Fill out the mandatory form [here](https://shl1.fra1.qualtrics.com/jfe/form/SV_eWMhjeljEFzNqSy) when done.
   - Requires Kaggle username, submission details, GitHub URL, and the final submission/output file.

## Rules & AI Usage
- **Timeframe:** Steady work should reach the threshold within ~2 days, though there is no fixed deadline. 
- **Submissions:** Multiple submissions are encouraged to iteratively improve (subject to Kaggle's daily limits). Time and score of each submission are recorded. First to meet the hidden threshold gets contacted.
- **AI Tools:** You are welcome to use AI tools, but you **must understand and be able to explain your approach** during the interview. A simple, well-understood approach is preferred over a complex one that cannot be explained clearly.
