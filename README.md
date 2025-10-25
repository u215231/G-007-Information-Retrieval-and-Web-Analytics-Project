# Information Retrieval and Web Analytics: Project Set-up

---

## Group members

| Name and Surnames | Identifier | E-Mail |
|----------------------|--------------|------------------------------------|
| Marc Bosch Manzano | u215231 | marc.bosch03\@estudiant.upf.edu |
| Christopher Matienzo Chilo | u198726 | christopher.matienzo01\@estudiant.upf.edu |
| Àlex Roger Moya | u199765 | alex.roger01\@estudiant.upf.edu |

---

## Introduction to Installation Process

This README contains the previous information to the user to execute properly all the codes. This project uses Python notebooks and also several libraries for data and text processing. 

---

## Python and IDE installation

First of all, check if you have in your computer installed Python, version 3. You can check with `python --version` on the command interpreter. If it is not installed, install it on the Python from https://www.python.org/downloads/ or from the local app store of your computer.

To run the notebooks, you can use some Integrated Development Enviroment (IDE) such as Visual Studio Code (VSCode), which can be downloaded from https://code.visualstudio.com/ or from the local app store of the computer.


---

## Installation of GitHub repository

Run the following commands in your console to set up the environment:

```bash
# 1. Clone the repository
git clone https://github.com/u215231/Information-Retrieval-and-Web-Analytics-Project.git

# 2. Navigate to the project directory
cd Information-Retrieval-and-Web-Analytics-Project
```
---


## Project Part 1

### Requirements

```bash
# 1. Navigate to project part 1
cd IRWA-2025-part-1

# 2. Install the required libraries
pip install pandas nltk matplotlib seaborn wordcloud

# 3. Download additional NLTK resources
python -m nltk.downloader stopwords punkt
```

### Code Execution

When all libraries are installed, we can execute `IRWA-2025-part-1-final-solution.ipynb` notebook with `Run All` option of VSCode. We will choose our installed Python kernel to run the notebook.
