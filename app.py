import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from transformers import pipeline
import streamlit.components.v1 as components
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NASA Bioscience Explorer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL & DATA LOADING ---
# Use Streamlit's caching to load models and data only once.

@st.cache_resource
def load_models():
    """Loads the sentence transformer and summarization models."""
    st.write("Loading AI models... This may take a moment.")
    search_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Using a smaller, faster model for summarization to fit the 18-hour constraint
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    return search_model, summarizer

@st.cache_data
def load_data():
    """Loads the publication data and pre-computed embeddings."""
    try:
        df = pd.read_csv('publications.csv', encoding='latin-1') # Use latin-1 for safety
        embeddings = np.load('embeddings.npy')
        embeddings_tensor = torch.from_numpy(embeddings)
        return df, embeddings_tensor
    except FileNotFoundError:
        return None, None

# --- BACKEND AI FUNCTIONS ---

def search(query, search_model, embeddings_tensor, top_k=10):
    """Performs semantic search."""
    if not query:
        return []
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    hits = semantic_search(query_embedding, embeddings_tensor, top_k=top_k)
    return hits[0] # Get the first list of hits

def get_summary(text):
    """Generates a summary for a given text."""
    if not text or len(text.split()) < 50:
        return "The abstract is too short to generate a meaningful summary."
    # The model has a max input length of 1024 tokens
    summary = summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']


def generate_keyword_network_graph():
    """Generates and saves the keyword network graph HTML file."""
    # This is a simplified version of the graph logic for integration
    # For a real app, this would be more sophisticated
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    from itertools import combinations
    from pyvis.network import Network

    st.write("Generating keyword network... please wait.")
    stop_words = set(stopwords.words('english'))
    stop_words.update(['study', 'results', 'showed', 'effects', 'also', 'using', 'space', 'flight', 'data'])
    
    # Ensure dataframe is loaded
    if 'df' not in st.session_state or st.session_state.df is None:
        st.error("Dataframe not loaded. Cannot generate graph.")
        return

    corpus = " ".join(st.session_state.df['Abstract'].dropna().tolist()).lower()
    words = [word for word in word_tokenize(corpus) if word.isalpha() and word not in stop_words]
    most_common_words = [word for word, freq in Counter(words).most_common(50)]

    co_occurrences = Counter()
    for abstract in st.session_state.df['Abstract'].dropna():
        tokens = set([word for word in word_tokenize(abstract.lower()) if word in most_common_words])
        for pair in combinations(sorted(tokens), 2):
            co_occurrences[pair] += 1

    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    net.add_nodes(most_common_words, value=[Counter(words)[w] for w in most_common_words])

    for pair, count in co_occurrences.items():
        if count > 5: # Co-occurrence threshold
            net.add_edge(pair[0], pair[1], value=count)
    
    net.save_graph("keyword_network.html")


# --- MAIN APPLICATION ---

# Load models and data
search_model, summarizer = load_models()
df, embeddings_tensor = load_data()

# Store data in session state to be accessible by callbacks
if 'df' not in st.session_state:
    st.session_state.df = df

st.title("🚀 NASA Bioscience Publication Explorer")
st.markdown("An AI-powered dashboard to search and summarize NASA's bioscience research.")

if df is None or embeddings_tensor is None:
    st.error("Critical files not found! Please make sure `publications.csv` and `embeddings.npy` are in the same directory as `app.py`.")
else:
    # --- SIDEBAR ---
    st.sidebar.header("Search & Options")
    query = st.sidebar.text_area("Enter your research query:", "effects of microgravity on bone density")
    search_button = st.sidebar.button("Search Publications")
    st.sidebar.markdown("---")
    st.sidebar.info(f"Loaded {len(df)} publications.")

    # --- MAIN CONTENT TABS ---
    tab1, tab2 = st.tabs(["🔎 Search Results", "🕸️ Keyword Network"])

    with tab1:
        st.header("Search Results")
        if search_button:
            with st.spinner("Searching through publications..."):
                hits = search(query, search_model, embeddings_tensor)
                st.session_state.hits = hits # Save hits to session state
        
        if 'hits' in st.session_state and st.session_state.hits:
            st.success(f"Found {len(st.session_state.hits)} relevant results.")
            for hit in st.session_state.hits:
                paper_index = hit['corpus_id']
                paper_score = hit['score']
                paper_data = df.iloc[paper_index]
                
                with st.expander(f"**{paper_data['Title']}** (Score: {paper_score:.2f})"):
                    st.markdown(f"**Authors:** {paper_data.get('Authors', 'N/A')}")
                    st.markdown(f"**Year:** {paper_data.get('Year', 'N/A')}")
                    
                    st.subheader("Abstract")
                    st.write(paper_data['Abstract'])

                    # Unique key for each button is essential
                    summary_button_key = f"summary_{paper_index}"
                    if st.button("Generate AI Summary", key=summary_button_key):
                        with st.spinner("Generating summary..."):
                            summary = get_summary(paper_data['Abstract'])
                            st.info(summary)
        else:
            st.info("Enter a query and click 'Search' to see results.")

    with tab2:
        st.header("Keyword Co-occurrence Network")
        st.markdown("This network shows the most common keywords from all abstracts and the connections between them. Thicker lines mean they appear together more often.")
        
        # Check if the graph HTML exists
        if not os.path.exists("keyword_network.html"):
            st.warning("Keyword graph not found.")
            if st.button("Generate Graph Now (takes ~1-2 minutes)"):
                generate_keyword_network_graph()
                st.success("Graph generated! Refreshing...")
                st.experimental_rerun() # Rerun the script to load the HTML file
        else:
            try:
                with open("keyword_network.html", 'r', encoding='utf-8') as f:
                    html_source = f.read()
                    components.html(html_source, height=800)
            except Exception as e:
                st.error(f"Could not load the network graph. Error: {e}")