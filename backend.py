"""Backend for the NASA BioQuery app, handling summarization and search."""
from transformers import pipeline
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
search_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_summary(text):
    """Generates a summary for a given text using a pre-loaded model."""
    if not text or len(text.split()) < 50: # Don't summarize very short texts
        return "Abstract too short to summarize."
    result = summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)
    return result[0]['summary_text']

def search(query, embeddings_tensor, top_k=10):
    """Performs semantic search on the given embeddings."""
    if not query:
        return []
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    hits = semantic_search(query_embedding, embeddings_tensor, top_k=top_k)
    # hits is a list of lists, get the first one
    return hits[0]
