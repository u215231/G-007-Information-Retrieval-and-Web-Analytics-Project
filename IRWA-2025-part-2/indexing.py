import pandas as pd
from processing import build_terms 

class Indexing:
    """
    Class to manage index of terms.
    """

    documents: pd.DataFrame
    """
    Dataframe containing information about documents. Must have some 
    identifier column and some text columns.
    """
    identifier_column: str
    """The column that is set to be the identifier of documents."""
    text_column: str
    """The column that is set to be the text of documents."""
    index: dict
    """Dictionary structured as (term, list of documents)."""

    def __init__(self, documents, identifier_column, text_column) -> None:
        self.documents = documents
        self.identifier_column = identifier_column
        self.text_column = text_column
        self.index = None

    def reset(
            self,
            documents: pd.DataFrame = None, 
            identifier_column: str = None, 
            text_column: str = None
        ) -> None:
        """
        Changes the properties of the indexing object. Resets the the index
        dictionary
        """
        self.documents = self.documents if documents is None else documents
        self.identifier_column = self.identifier_column if identifier_column is None\
            else identifier_column
        self.text_column = self.text_column if text_column is None else text_column
        self.index = None

    def build_inverted_index(self) -> dict:
        """Builds the inverted index from the documents of the object."""
        if self.index is not None:
            return self.index
        index = {} 
        for _, row in self.documents.iterrows(): 
            doc_id = row[self.identifier_column] 
            terms = row[self.text_column].split() 
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
        index = self.index if self.index is not None else self.build_inverted_index()
        for term in list(index.keys())[0:max_terms]:
            print(f"{term}: {index[term][0: max_terms]} ...")

    def search_by_conjunctive_queries(self, query: str) -> list:
        """Retrieves the documents that contain the query terms."""
        index = self.index if self.index is not None else self.build_inverted_index()
        terms = build_terms(query)
        result = None
        for term in terms:
            if term in index:
                docs = set(index[term])
                result = docs if result is None else result & docs
            else:
                return []
        return sorted(result)