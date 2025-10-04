# In backend.py
from transformers import pipeline
# In backend.py
from sentence_transformers.util import semantic_search
import pytorch
# Use a smaller model for speed
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

def get_summary(text):
    if not text or len(text.split()) < 50: # Don't summarize very short texts
        return "Abstract too short to summarize."
    # Max length of input is 1024 for this model
    result = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']
# This function assumes model and embeddings are loaded elsewhere
def search(query, model, embeddings_tensor, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = semantic_search(query_embedding, embeddings_tensor, top_k=top_k)
    # hits is a list of lists, get the first one
    return hits[0]