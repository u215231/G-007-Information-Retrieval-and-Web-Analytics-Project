import os
import uuid
from json import JSONEncoder
from flask import jsonify

import pandas as pd
import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session, redirect, url_for
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env

# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME") 

search_engine = SearchEngine()
analytics_data = AnalyticsData()
rag_generator = RAGGenerator()

full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
file_csv_path = path + "/" + os.getenv("DATA_CSV_FILE_PATH") 

my_corpus = pd.read_csv(file_csv_path)
corpus = load_corpus(file_path)
print("\nMy Corpus is loaded... \n", my_corpus.head(3))
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        # Register new session in analytics
        analytics_data.register_session(session["session_id"])
        print("New session registered:", session["session_id"])
    else:
        print("Existing session:", session["session_id"])

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)
    
    # Pass the parsed agent to register_request
    analytics_data.register_request(
        path=request.path,
        user_ip=user_ip,
        user_agent=user_agent,
        agent_parsed=agent
    )

    # debug prints as before
    print("Raw user browser:", user_agent)
    print("Remote IP:", user_ip)
    print("JSON browser:", httpagentparser.detect(user_agent))
    print("Current session:", session)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")




@app.route('/search', methods=['POST'])
def search_form_post():
    
    search_query = request.form['search-query']

    session['last_search_query'] = search_query

    search_id = analytics_data.save_query_terms(search_query)

    # Guarda l'search_id a la sessió
    session['current_search_id'] = search_id

    return redirect(url_for('search_results'))

@app.route('/results', methods=['GET'])
def search_results():

    search_query = session.get('last_search_query')
    search_id = session.get('current_search_id')

    results = search_engine.tfidf_search(search_query, search_id, my_corpus)

    rag_response = rag_generator.generate_response(search_query, results)

    found_count = len(results)
    session['last_found_count'] = found_count

    return render_template(
        'results.html',
        results_list=results,
        page_title="Results",
        found_counter=found_count,
        rag_response=rag_response,
        search_id=search_id
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    clicked_doc_id = request.args["pid"]
    search_id = request.args.get("search_id", type=int)

    event_id = analytics_data.register_click(search_id, clicked_doc_id)

    document = corpus.get(clicked_doc_id)
    doc = corpus.get(clicked_doc_id)
    print(doc.image)
    if not document:
        return "Document not found", 404

    return render_template(
        'doc_details.html', 
        doc=document, 
        event_id=event_id
    )

@app.route('/log_dwell_time', methods=['POST'])
def log_dwell_time():
    """
    New route to receive dwell time data from the frontend.
    """
    data = request.json
    event_id = data.get('event_id')
    dwell_time = data.get('dwell_time')
    
    if event_id is not None:
        analytics_data.update_dwell_time(event_id, float(dwell_time))
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(
            pid=row.pid, 
            title=row.title, 
            description=row.description, 
            url=row.url, 
            count=count
        )
        docs.append(doc)
    
    docs.sort(key=lambda doc: doc.count, reverse=True)

    id_to_terms = {q['query_id']: q['terms'] for q in analytics_data.fact_queries}
    
    doc_terms_map = {}
    for event in analytics_data.click_events:
        d_id = event['doc_id']
        q_id = event['query_id']
        
        term = id_to_terms.get(q_id)
        
        if term:
            if d_id not in doc_terms_map:
                doc_terms_map[d_id] = set()
            doc_terms_map[d_id].add(term)
            
    doc_queries_map = {k: sorted(list(v)) for k, v in doc_terms_map.items()}
    query_id_to_term = {q['query_id']: q['terms'] for q in analytics_data.fact_queries}

    return render_template('stats.html', 
                           clicks_data=docs, 
                           requests_data=analytics_data.requests, 
                           sessions_data=analytics_data.sessions, 
                           queries_data=analytics_data.fact_queries,
                           click_events_data=analytics_data.click_events,
                           doc_queries_map=doc_queries_map,
                           query_id_to_term=query_id_to_term)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    query_stats = analytics_data.get_query_stats()

    return render_template('dashboard.html', 
                           visited_docs=visited_docs, 
                           query_stats=query_stats)

# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()

@app.route('/plot_browsers', methods=['GET'])
def plot_browsers():
    return analytics_data.plot_browser_stats()

@app.route('/plot_os', methods=['GET'])
def plot_os():
    return analytics_data.plot_os_stats()

@app.route('/plot_time')
def plot_time():
    return analytics_data.plot_hourly_traffic()

if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))

"""Deprecated"""
if False:
    @app.route('/search', methods=['POST'])
    def search_form_post():
        search_query = request.form['search-query']
        session['last_search_query'] = search_query
        search_id = analytics_data.save_query_terms(search_query)
        results = search_engine.tfidf_search(search_query, search_id, my_corpus)
        # generate RAG response based on user query and retrieved results
        rag_response = rag_generator.generate_response(search_query, results)
        print("RAG response:", rag_response)
        found_count = len(results)
        session['last_found_count'] = found_count
        print(session)
        return render_template(
            'results.html', 
            results_list=results, 
            page_title="Results", 
            found_counter=found_count, 
            rag_response=rag_response
        )
