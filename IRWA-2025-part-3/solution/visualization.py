import pandas as pd
from typing import Literal
from IPython.display import display
import matplotlib.pyplot as plt
from .model.bm25 import BM25
from .model.custom import Custom
from .model.tfidf import TFIDF


class RankingVisualization:
    """
    Class for ranking visualizations of results.
    """
    
    tfidf: TFIDF
    bm25: BM25
    custom: Custom
    documents: pd.Series
    queries: pd.Series
    comparisons_df: pd.DataFrame
    topK: int
    boolean_filter: bool


    def __init__(
            self,
            documents: pd.Series | None = None,
            static_scores: pd.Series | None = None,
            queries: pd.Series | None = None
        ) -> None:
        """
        Constructor for ranking visualizator object.
        """
        self.documents = documents
        self.queries = queries
        self.tfidf = TFIDF(documents, queries)
        self.bm25 = BM25(documents, queries)
        self.custom = Custom(documents, static_scores, queries)
        self.comparisons_df = None
        self.topK = None
        self.boolean_filter = None


    def get_comparisons_df(
            self, 
            boolean_filter: bool = True,
            topK: int = 10
        ) -> pd.DataFrame:
        """
        Constructs a comparisons dataframe with columns ['Document ID', 
        'TF-IDF Rank', 'TF-IDF Score', 'BM25 Rank', 'BM25 Score', 
        'Custom Rank', 'Custom Score'].
        """
        if (
            self.comparisons_df is not None 
            and self.topK == topK 
            and self.boolean_filter == boolean_filter
        ):
            return self.comparisons_df
        
        tfidf_df = self.tfidf.get_scores_df(topK=topK, boolean_filter=boolean_filter)
        bm25_df = self.bm25.get_scores_df(topK=topK, boolean_filter=boolean_filter)
        custom_df = self.custom.get_scores_df(topK=topK, boolean_filter=boolean_filter)

        tfidf_df.drop(columns=["query_text", "document_text"], inplace=True)
        bm25_df.drop(columns=["query_text", "document_text"], inplace=True)
        custom_df.drop(columns=["query_text", "document_text"], inplace=True)

        if len(self.queries) == 1:
            tfidf_df.drop(columns="query_id", inplace=True)
            bm25_df.drop(columns="query_id", inplace=True)
            custom_df.drop(columns="query_id", inplace=True)

        tfidf_df.rename(columns={"document_id": "Document ID", "score": "TF-IDF Score"}, inplace=True)
        bm25_df.rename(columns={"document_id": "Document ID", "score": "BM25 Score"}, inplace=True)
        custom_df.rename(columns={"document_id": "Document ID", "score": "Custom Score"}, inplace=True)

        tfidf_df["TF-IDF Rank"] = tfidf_df.index + 1
        bm25_df["BM25 Rank"] = bm25_df.index + 1
        custom_df["Custom Rank"] = custom_df.index + 1

        comparisons_df = pd.merge(
            tfidf_df[['Document ID', 'TF-IDF Rank', 'TF-IDF Score']],
            bm25_df[['Document ID', 'BM25 Rank', 'BM25 Score']],
            on='Document ID',
            how='outer'
        )

        comparisons_df = pd.merge(
            comparisons_df,
            custom_df[['Document ID', 'Custom Rank', 'Custom Score']],
            on='Document ID',
            how='outer'
        )

        self.comparisons_df = comparisons_df
        self.topK = topK
        self.boolean_filter = boolean_filter
        return comparisons_df


    def print_comparisons_df(
            self, 
            sort_by: Literal["TF-IDF Rank", "BM25 Rank", "Custom Rank"],
            boolean_filter: bool = True,
            topK: int = 10,
            head: int | None = None
        ) -> None:
        """
        Print the comparisons dataframe sorted by some ranking method. ``topK`` is
        for computing the ranking for each method; ``head`` is only used as visualization
        for number of rows of comparisons dataframe. 
        """
        comparisons_df = self.get_comparisons_df(boolean_filter, topK)
        comparisons_df = comparisons_df.sort_values(by=sort_by).reset_index(drop=True)
        comparisons_df = comparisons_df.head(head) if head is not None else comparisons_df
        print("\n--- Top Document Rank Comparison (Sorted by BM25 Rank) ---")
        display(comparisons_df)


    def get_number_of_matchings_dict(
            self, 
            boolean_filter: bool = True,
            topK: int = 10
        ) -> dict:
        """
        Returns a dict of pairs of ranking models as keys and number of matchings as
        values {(model_name_1, model_name_2): num_of_matchings}.
        """
        
        comparisons_df = self.get_comparisons_df(boolean_filter, topK)
        model_names = ["TF-IDF Rank", "BM25 Rank", "Custom Rank"]
        matching_dict = {}

        for m1 in model_names:
            sorted_df = comparisons_df.sort_values(by=m1).reset_index(drop=True)
            selected_df = sorted_df.head(topK)

            for m2 in model_names:
                matching_dict[(m1, m2)] =  selected_df[m2].count()

        return matching_dict
    
    
    def get_number_of_matchings_heat_map(
            self, 
            boolean_filter: bool = True,
            topK: int = 10
        ) -> pd.DataFrame:
        """
        Gets a heat map as a dataframe of columns and index being the names
        of the ranking models: ["TF-IDF Rank", "BM25 Rank", "Custom Rank"].
        """

        data = self.get_number_of_matchings_dict(boolean_filter, topK)
        rows = sorted({row for row, _ in data})
        cols = sorted({col for _, col in data})

        matrix = [[data[(r, c)] for c in cols] for r in rows]

        return pd.DataFrame(matrix, index=rows, columns=cols)


    def plot_number_of_matchings_heat_map(
            self, 
            boolean_filter: bool = True,
            topK: int = 10
        ) -> None:
        """
        Plots the number of matches heat map.
        """

        df = self.get_number_of_matchings_heat_map(boolean_filter, topK)
        self.plot_heat_map(df)
        
    
    @staticmethod
    def plot_heat_map(df: pd.DataFrame) -> None:
        """
        Author: ChatGPT GPT-5.1.\n
        Plot a heat map expressed as a dataframe with same columns as 
        index values, including numeric values inside the cells and
        automatic text color selection (white/black) based on background.
        """
        fig, ax = plt.subplots()
        cax = ax.imshow(df.values)

        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index)

        cmap = cax.cmap
        norm = cax.norm

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):

                value = df.iat[i, j]
                r, g, b, _ = cmap(norm(value))

                luminance = 0.299*r + 0.587*g + 0.114*b

                text_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, str(value),
                        ha='center', va='center',
                        color=text_color)

        fig.colorbar(cax)
        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def print_query_rankings(scores_df: pd.DataFrame, queries: pd.Series) -> None:
        """
        Print scores for each query given input with columns 
        [query_id, query_text, document_id, document_text, score].
        """

        print("QUERY SCORES")
        for query_id in queries.index:
            
            query_scores_df = scores_df[scores_df["query_id"] == query_id]\
                .drop(columns=["query_id", "query_text"])

            print("=== query %d: %s ===" % (query_id, queries[query_id]))
            display(query_scores_df)

