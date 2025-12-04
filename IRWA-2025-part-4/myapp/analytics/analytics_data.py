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
    
    def get_query_stats(self):
        """
        Returns a list of queries and their counts, sorted by popularity.
        """
        counts = {}
        for q in self.fact_queries:
            term = q['terms']
            if term: # Ensure we don't count empty queries if any
                counts[term] = counts.get(term, 0) + 1

        
        # Convert to a list of dictionaries for easier handling in the template
        stats = [{'term': term, 'count': count} for term, count in counts.items()]
        
        # Sort by count (descending)
        stats.sort(key=lambda x: x['count'], reverse=True)
        return stats

    def register_click(self, query_id: int, doc_id: str, description: str = "") -> int:
        """
        Register a click on a document for a given query.
        Returns the event_id (index) so we can update dwell time later.
        """
        # update total clicks per doc
        self.fact_clicks[doc_id] = self.fact_clicks.get(doc_id, 0) + 1

        # Store detailed event with timestamp and placeholder for dwell_time
        event = {
            "query_id": query_id,
            "doc_id": doc_id,
            "description": description,
            "timestamp": pd.Timestamp.now(),
            "dwell_time": 0  # Placeholder, will be updated when user leaves page
        }
        self.click_events.append(event)
        
        # Return the index of this event
        return len(self.click_events) - 1

    def update_dwell_time(self, event_id: int, dwell_time: float):
        """
        Update the dwell time for a specific click event.
        """
        if 0 <= event_id < len(self.click_events):
            self.click_events[event_id]["dwell_time"] = round(dwell_time, 2)

    def register_session(self, session_id):
        self.sessions.append({
            "session_id": session_id,
            "timestamp": pd.Timestamp.now()
        })

    def register_request(self, path, user_ip, user_agent, agent_parsed):

        # Extract info from parsed agent
        browser_info = agent_parsed.get('browser', {})
        os_info = agent_parsed.get('os', {})
        platform_info = agent_parsed.get('platform', {})
        
        # Get timestamp and split into date and time
        timestamp = pd.Timestamp.now()
        
        self.requests.append({
            "path": path,
            "ip": user_ip,
            "user_agent": user_agent,
            "browser": browser_info.get('name', 'Unknown'),
            "browser_version": browser_info.get('version', ''),
            "os": os_info.get('name', 'Unknown'),
            "os_version": os_info.get('version', ''),
            "platform": platform_info.get('name', 'Unknown'),
            "timestamp": timestamp,
            "date": timestamp.strftime('%Y-%m-%d'),
            "time": timestamp.strftime('%H:%M:%S')
        })
        
    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': str(doc_id), 'Number of Views': float(count)} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        if df.empty:
            return "<p>No views data available yet.</p>"
        # Create Altair chart with explicit types
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', title='Document ID'),
            y=alt.Y('Number of Views:Q', title='Number of Views')
        ).properties(
            title='Number of Views per Document'
        )
        # Render the chart to HTML
        return chart.to_html()
    

    def plot_browser_stats(self):
        df = pd.DataFrame(self.requests)
        if df.empty: return "<p>No data</p>"
        
        chart = alt.Chart(df).mark_arc().encode(
            theta=alt.Theta("count()", stack=True),
            color=alt.Color("browser"),
            tooltip=["browser", "count()"]
        ).properties(title="Visitors by Browser")
        return chart.to_html()
    
    def plot_os_stats(self):
        df = pd.DataFrame(self.requests)
        if df.empty: return "<p>No data</p>"
        
        chart = alt.Chart(df).mark_bar().encode(
            y='count()',
            x='os',
            color='os',
            tooltip=["os", "count()"]
        ).properties(title="Visitors by Operating System")
        return chart.to_html()
    
    def plot_hourly_traffic(self):
        df = pd.DataFrame(self.requests)
        if df.empty: return "<p>No data</p>"
        
        # Extract hour from timestamp
        df['hour'] = df['timestamp'].dt.hour
        
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('hour', axis=alt.Axis(title='Hour of Day')),
            y=alt.Y('count()', axis=alt.Axis(title='Number of Requests')),
            tooltip=['hour', 'count()']
        ).properties(title="Traffic Volume by Hour")
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