import os
import numpy as np
import json

def load_geneformer_embeddings(path, reordered=True, brain=True):
    if brain:
        path = os.path.join(path, "geneformer", "gene_embeddings_ordered_brain.json")
    elif reordered:
        path = os.path.join(path, "geneformer", "gene_embeddings_ordered.json")
    else:
        path = os.path.join(path, "geneformer", "gene_embeddings_big.json")
    embeddings_dict = json.load(open(path))
    all_genes = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[gene] for gene in all_genes])
    return np.array(embeddings), list(all_genes)

def load_word2vec_embeddings(path):
    emb_path = os.path.join(path, "word2vec", "cleaned_google_news_embeddings.npy")
    embeddings= np.load(emb_path)
    words_path = os.path.join(path, "word2vec", "cleaned_google_news_words.npy")
    words = np.load(words_path)
    return np.array(embeddings), list(words)