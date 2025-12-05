import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("punkt")
nltk.download("stopwords")
import pandas as pd
import os
from pathlib import Path

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update({"nan", "none", "null", ""})
SELECTED_ATTRIBUTES = [
    "pid", "title", "description",  "brand",  "category", "sub_category", 
    "product_details", "seller", "out_of_stock", "selling_price", "discount", 
    "actual_price", "average_rating", "url"
]
MERGED_TEXT_FIELDS = [
    "title", "description", "category", "sub_category", "brand", "seller", "merged_product_details"
]
NUMERICAL_COLUMNS = [
    "selling_price", "discount", "actual_price", "average_rating"
]

DATA_DIRECTORY = Path("../../../data")
INPUT_PATH = DATA_DIRECTORY / "fashion_products_dataset.json"
OUTPUT_PATH = DATA_DIRECTORY / "fashion_products_dataset_processed_review.csv"

TOP_PRODUCT_DETAILS = 10

def build_terms(line: str) -> list[str]:
    """Function to process the text of a document."""
    line = line.lower()
    line = re.sub(r"[^a-zA-Z-]", " ", line)
    line = line.split(" ")
    line = [word for word in line if word not in STOP_WORDS]
    line = [STEMMER.stem(word) for word in line]
    return line

def build_terms_str(line: str) -> str:
    """Function to process the text of a document."""
    return " ".join(build_terms(line))


def flatten_details(details: list[dict]) -> dict:
    """
    Converts a list of dictionariess [{a:1}, {b:2}] to a unique dictionary {a:1, b:2}.
    
    Parameters:
        details (list[dict]): A list containing atomic dictionaries.

    Returns:
        dict: A merged dict.
    """
    if not isinstance(details, list):
        return {}
    merged = {}
    for detail in details:
        if isinstance(detail, dict):
            merged.update(detail)
    return merged

if __name__ == "__main__":
    print("opening %s..." % INPUT_PATH)
    df = pd.read_json(INPUT_PATH)

    print("processing title...")
    df['title'] = df["title"].apply(build_terms_str)
    
    print("processing description...")
    df['description'] = df["description"].apply(build_terms_str)

    df = df[SELECTED_ATTRIBUTES].copy()

    print("processing product details...")
    product_details_df = df["product_details"].apply(flatten_details).apply(pd.Series)
    product_details_df.columns = ['product_details_' + col for col in product_details_df.columns]

    most_frequent_product_details = product_details_df.notnull().sum().sort_values(ascending=False).head(TOP_PRODUCT_DETAILS)
    product_details_df = product_details_df[most_frequent_product_details.index]
    product_details_df["merged_product_details"] = product_details_df.astype(str).apply(" ".join, axis=1)
    product_details_df["merged_product_details"] = product_details_df["merged_product_details"].apply(build_terms_str)

    print("creating documents column...")
    df = pd.concat([df, product_details_df], axis=1)
    
    df["document"] = df[MERGED_TEXT_FIELDS].apply(" ".join, axis=1)
    df["document"] = df["document"].apply(build_terms_str)

    df = df.drop(columns=["product_details", "merged_product_details"])

    print("processing numerical columns...")
    for col in NUMERICAL_COLUMNS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
        )

    df.to_csv(OUTPUT_PATH, sep=",", index=False, encoding="utf-8")