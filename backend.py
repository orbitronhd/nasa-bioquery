# In backend.py
from transformers import pipeline
# Use a smaller model for speed
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

def get_summary(text):
    if not text or len(text.split()) < 50: # Don't summarize very short texts
        return "Abstract too short to summarize."
    # Max length of input is 1024 for this model
    result = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']