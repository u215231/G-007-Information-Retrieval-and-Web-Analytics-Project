import pandas as pd
from .processing import build_terms 

class Indexing:
    """
    Class to manage index of terms.
    """

    documents: pd.Series
    """Series with (document_id, documnent_text). """

    index: dict
    """Dictionary structured as (term, list of documents)."""

    def __init__(self, documents: pd.Series = None) -> None:
        self.documents = documents
        self.index = None

    def reset(self, documents: pd.Series = None) -> None:
        """
        Changes the properties of the indexing object. Resets the the index
        dictionary
        """
        self.documents = self.documents if documents is None else documents
        self.index = None

    def build_inverted_index(self) -> dict:
        """
        Builds the inverted index from the documents of the object.
        """
        if self.index is not None:
            return self.index
        index = {} 
        for doc_id, doc_text in self.documents.items(): 
            terms = doc_text.split() 
            for term in set(terms): 
                if term not in index: 
                    index[term] = [doc_id] 
                else: 
                    index[term].append(doc_id) 
        self.index = dict(index)
        return index
    
    def print_inverted_index(self, max_terms: int = 5, max_documents: int = 5) -> None:
        """
        Prints the first max_documents of the first max_terms terms of the
        inverted index.
        """
        index = self.build_inverted_index() if self.index is None else self.index
        for term in list(index.keys())[0:max_terms]:
            print(f"{term}: {index[term][0: max_documents]} ...")

    def search_by_conjunctive_queries(self, query_text: str) -> pd.Series:
        """
        Retrieves the document idenfitifiers that contain the query terms.
        """
        index = self.build_inverted_index() if self.index is None else self.index
        terms = set(build_terms(query_text)) & set(index.keys())
        result = set()
        for term in terms:
            docs = set(index[term])
            result = docs if not len(result) else result & docs
        return self.documents[self.documents.index.isin(result)]
    