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
git clone https://github.com/u215231/G-007-Information-Retrieval-and-Web-Analytics-Project.git

# 2. Navigate to the project directory
cd Information-Retrieval-and-Web-Analytics-Project
```
---

## Project Part 1, 2, 3

### Requirements

```bash
# 1. Navigate to project part N (adapt N to be 1, 2, or 3)
cd IRWA-2025-part-N

# 2. Install the required libraries
pip install pandas nltk matplotlib seaborn wordcloud

# 3. Download additional NLTK resources
python -m nltk.downloader stopwords punkt
```

### Code Execution

When all libraries are installed, we can execute `IRWA-2025-part-N-final-solution.ipynb` notebook with `Run All` option of VSCode, where `N` is the current project part. We will choose our installed Python kernel to run the notebook.

## Project Part 4

### Basic set up 

You have to run the following commands in our console to set up the enviroment. First navigate to the project part 4 directory.

```bash
cd IRWA-2025-part-4
```

### Virtual Enviroment

Then, set up the Python environment (only for the first time you run the project).

```bash
# 1. In the project root directory execute:
pip3 install virtualenv
virtualenv --version

# 2. In the root of the project folder run to create a virtualenv named `irwa_venv`:
virtualenv irwa_venv

# 3. The next step is to activate your new virtualenv for the project:
source irwa_venv/bin/activate

# or for Windows (recomended to be run in Command Prompt Interpeter, not in Power Shell)...
irwa_venv\Scripts\activate.bat
```

### Flask installation

Then, we install Flask and other packages in your virtualenv. Make sure you are in the root of the project folder and that your virtualenv is activated (you should see `(irwa_venv)` in your terminal prompt).


### Required packages

And then install all the packages listed in `requirements.txt` with:
```bash
pip install -r requirements.txt
```

If you need to add more packages in the future, you can install them with pip and then update `requirements.txt` with:
```bash
pip freeze > requirements.txt
```

### Enviroment variables file

In `IRWA-2025-part-4` create a file called `.env` if it is not created. Set the following configurations in this file as below:

```bash
# === GROQ API CONFIGURATION ===
GROQ_API_KEY="groq generated key"
GROQ_MODEL=llama-3.1-8b-instant

# === FLASK CONFIGURATION ===
FLASK_ENV=development
SECRET_KEY="some random key of 37 alphanumerical characters" 
SESSION_COOKIE_NAME=IRWA_SEARCH_ENGINE
DATA_FILE_PATH=data/fashion_products_dataset.json
DATA_CSV_FILE_PATH=data/fashion_products_dataset_processed_review.csv
MAX_DOCUMENTS_DISPLAYED=50

# === OPTIONAL (debugging) ===
DEBUG=True
```

You can take as example key afgsreg86sr897b6st8b76va8er76fcs6g8d7. The Groq API Key has to be generated in https://groq.com/ web page, on section `Start Building`. You have to create a groq user and then obtain key in `API Keys` section.

### Run the app

Inside `IRWA-2025-part-4`, we activate the virtual environment with `virtualenv irwa_venv` in Linux/MacOs, or `irwa_venv\Scripts\activate.bat you` in Windows. Then, we call the following commands to run the web application:

```bash
# If you are in Linux/MacOS...
./run.sh

# ...or if you are in Windows...
run.bat

# or as a general way...
python web_app.py
```

In the terminal, you will see something like:

```bash
 * Serving Flask app 'web_app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8088
 * Running on http://192.168.1.14:8088
Press CTRL+C to quit
```

You have to click on http://127.0.0.1:8088 address to render the application in your browser.

### Attributions 
The configuration section for project Part 4 is adapted from the following source: [IRWA Template 2021](https://github.com/irwa-labs/search-engine-web-app)