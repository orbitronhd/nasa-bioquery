# In graph.py
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import combinations
from pyvis.network import Network

def generate_keyword_network_graph(df, top_n=30, co_occurrence_threshold=5):
    stop_words = set(stopwords.words('english'))
    # Add custom stopwords
    stop_words.update(['study', 'results', 'showed', 'effects', 'effect', 'also', 'using', 'space', 'flight'])

    corpus = " ".join(df['abstract'].dropna().tolist()).lower()
    words = [word for word in word_tokenize(corpus) if word.isalpha() and word not in stop_words]
    most_common_words = [word for word, freq in Counter(words).most_common(top_n)]

    co_occurrences = Counter()
    for abstract in df['abstract'].dropna():
        tokens = set([word for word in word_tokenize(abstract.lower()) if word in most_common_words])
        for pair in combinations(sorted(tokens), 2):
            co_occurrences[pair] += 1

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    net.add_nodes(most_common_words, value=[Counter(words)[w] for w in most_common_words])

    for pair, count in co_occurrences.items():
        if count > co_occurrence_threshold:
            net.add_edge(pair[0], pair[1], value=count)

    net.save_graph("keyword_network.html")