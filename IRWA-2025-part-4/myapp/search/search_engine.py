import random
import numpy as np

from myapp.search.objects import Document
from myapp.search.solution.model.tfidf import TFIDF
import pandas as pd
from myapp.search.solution.processing import build_terms_str

class SearchEngine:
    """Class that implements the search engine logic"""

    @staticmethod
    def tfidf_search(
        search_query: str, 
        search_id: int, 
        corpus: pd.DataFrame, 
        topK: int = 50
    ) -> list[Document]:
        print("Search query:", search_query)
        
        documents = corpus.set_index("pid")["document"]
        query = pd.Series([build_terms_str(search_query)], [search_id])

        ranker = TFIDF(documents, query)
        scores_df = ranker.get_scores_df(topK)
        ranked_corpus = corpus.set_index("pid").loc[scores_df["document_id"]]

        results = []
        for doc_id, doc in ranked_corpus.iterrows():
            results.append(Document(
                pid=str(doc_id), 
                title=str(doc["title"]), 
                description="" if str(doc["description"]) == "nan" else str(doc["description"]), 
                selling_price=str(doc["selling_price"]), 
                actual_price=str(doc["actual_price"]),
                discount=str(doc["discount"]), 
                average_rating=str(doc["average_rating"]),
                url="doc_details?pid={}&search_id={}&param2=2".format(doc_id, search_id)
            ))
        return results

    @staticmethod
    def dummy_search(
        search_query: str, 
        search_id: int, 
        corpus: dict, 
        num_results: int = 50
    ) -> list[Document]:
        """
        Just a demo method, that returns random <num_results> documents from the corpus
        
        :param corpus: the documents corpus
        :param search_id: the search id
        :param num_results: number of documents to return
        :return: a list of random documents from the corpus
        """
        print("Search query:", search_query)

        results = []
        doc_ids = list(corpus.keys())
        docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
        for doc_id in docs_to_return:
            doc = corpus[doc_id]
            results.append(Document(
                pid=doc.pid, 
                title=doc.title, 
                description=doc.description, 
                selling_price=doc.selling_price, 
                discount=doc.discount, 
                average_rating=doc.average_rating,
                url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), 
                ranking=random.random()
            ))
        return results