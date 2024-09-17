

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


# Create an AnnData object
adata = sc.AnnData(X=embeddings)
adata.obs_names = list(all_genes)

# Perform UMAP
sc.pp.neighbors(adata, use_rep='X')
sc.tl.umap(adata)

# Perform Leiden clustering
sc.tl.leiden(adata)

# Plot the results
sc.pl.umap(adata, color='leiden', save='_gene_embeddings_leiden.png')

print("UMAP projection and Leiden clustering completed. Results saved as umap_gene_embeddings_leiden.png")

# Save the AnnData object
adata.write("gene_embeddings.h5ad")
print("Gene embeddings saved as gene_embeddings.h5ad")
