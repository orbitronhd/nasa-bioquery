"""Pre-processes publication data to generate and save sentence embeddings."""
from sentence_transformers import SentenceTransformer  # type: ignore
import pandas as pd
import numpy as np

df = pd.read_csv("publications.csv", encoding="latin-1")
abstracts = df["Abstract"].fillna("").tolist()
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings for all abstracts...")
embeddings = model.encode(abstracts, show_progress_bar=True)
np.save("embeddings.npy", embeddings)
print("Embeddings saved!")
