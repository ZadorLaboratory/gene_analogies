# %% [markdown]
# # Vector analogies in gene embeddings
# 
# It is quite possible that we can learn a lot about genes by looking at their embeddings.
# 
# ## 

# %%

import os
from transformers import BertForMaskedLM
import pickle
import json
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

import requests, sys
import h5py
import time 
import anndata as ad



# %%
if os.path.exists("gene_embeddings_big.json"):
    embeddings_dict = json.load(open("gene_embeddings_big.json"))
    all_genes = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[gene] for gene in all_genes])
else:
    ## Get token dict
    token_path = "/nfs/scratch-1/benjami/barseq-transformer/geneformer/token_dictionary.pkl"
    print(f"Loading {token_path}")
    with open(token_path, "rb") as fp:
        token_dictionary = pickle.load(fp)

    ## Load the model
    model =  BertForMaskedLM.from_pretrained("/mnt/storage/benjami/geneformer/geneformer-12L-30M/models")

    # Get all gene tokens from the tokenizer
    # all_genes = [token for token in token_dictionary.keys() if token.startswith("E")]
    all_genes = token_dictionary.keys()
    embeddings = model.bert.embeddings.word_embeddings.weight.detach().numpy()
    embeddings_dict = {gene:emb for gene,emb in zip(all_genes, embeddings[len(embeddings)-len(all_genes):])}
    with open("gene_embeddings_big.json", "w") as fp:
        serializable_dict = {gene:emb.tolist() for gene,emb in embeddings_dict.items()}
        json.dump(serializable_dict, fp)

from sklearn.metrics.pairwise import cosine_similarity

def try_and_wait(n,headers):
    success = False
    while not success:
        r = requests.get(n,headers=headers)
        if r.status_code == 429:
            seconds_to_wait = int(r.headers["Retry-After"])
            time.sleep(seconds_to_wait)
        else:
            success = True
    return r

def get_gene_description(ensembl_id):
    server = "https://rest.ensembl.org"
    ext = f"/lookup/id/{ensembl_id}?content-type=application/json"
    r = try_and_wait(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        print(r)
        return None
    decoded = r.json()
    if 'description' not in decoded:
        print(decoded)
        return None
    return decoded['description']
    
if os.path.exists("gene_descriptions.json"):
    gene_descriptions = json.load(open("gene_descriptions.json"))
    def get_gene_description(ensembl_id):
        return gene_descriptions[ensembl_id]


def get_most_similar_genes(query_gene, embeddings_dict, embeddings, metric='cosine',n=10):
    query_embedding = embeddings_dict[query_gene]
    if metric=='cosine':
        similarities = cosine_similarity([query_embedding], embeddings)
        most_similar_genes = np.argsort(similarities[0])[::-1][:n]
    elif metric=='euclidean':
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)
        most_similar_genes = np.argsort(distances)[:n]
    return most_similar_genes


# %%
if not os.path.exists("gene_descriptions.json"):
    gene_descriptions = {}
    for gene in all_genes:
        if gene.startswith("ENSG"):
            gene_descriptions[gene] = get_gene_description(gene)
            print(gene, gene_descriptions[gene])
    with open("gene_descriptions.json", "w") as fp:
        json.dump(gene_descriptions, fp)
