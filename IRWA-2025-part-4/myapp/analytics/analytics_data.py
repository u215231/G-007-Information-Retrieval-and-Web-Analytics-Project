import json
import random
import altair as alt
import pandas as pd


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    def  __init__(self):
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
        self.fact_clicks = {}
        self.fact_queries = []
        self.click_events = []
        self.sessions = []
        self.requests = []

    ### Please add your custom tables here:

    def save_query_terms(self, terms: str) -> int:
        """
        Save a search query and return its query_id.
        """
        query_id = len(self.fact_queries) + 1
        self.fact_queries.append({
            "query_id": query_id,
            "terms": terms,
            "num_terms": len(terms.split()) if terms else 0
        })
        return query_id
    
    def register_click(self, query_id: int, doc_id: str, description: str = "") -> None:
        """
        Register a click on a document for a given query.
        Increments the counter and stores an event.
        """
        # update total clicks per doc
        self.fact_clicks[doc_id] = self.fact_clicks.get(doc_id, 0) + 1

        # store event (útil si luego quieres más métricas)
        self.click_events.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "description": description
        })

    def register_session(self, session_id):
        self.sessions.append({
            "session_id": session_id,
            "timestamp": pd.Timestamp.now()
        })

    def register_request(self, path, user_ip, user_agent):
        self.requests.append({
            "path": path,
            "ip": user_ip,
            "user_agent": user_agent,
            "timestamp": pd.Timestamp.now()
        })
    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        # Create Altair chart
        chart = alt.Chart(df).mark_bar().encode(
            x='Document ID',
            y='Number of Views'
        ).properties(
            title='Number of Views per Document'
        )
        # Render the chart to HTML
        return chart.to_html()
        
class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)