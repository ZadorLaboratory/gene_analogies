"""Load a geneformer model and get gene embeddings for all available genes.
First, we'll load the token dictionary. Then, we'll load the model and get the embeddings for all genes.
Finally, we'll create a new dictionary with the gene names and their embeddings and save it to a file."""

import torch
from transformers import AutoTokenizer, AutoModel
import json
from geneformer.util.parameters import (big_models_set, foundation_models_path,
                                        gene_panel_mapping_path_map,
                                        models_path_map, pretrained_models_set,
                                        token_dict_pkl_name, token_path_map)
import argparse

parent_directory = os.environ.get('ROOT_DATA_PATH') # /home/benjami/mnt/zador_nlsas_norepl_data/Ari/transcriptomics

args = argparse.ArgumentParser()
args.add_argument('--base-model-name', type=str, default="base_human_geneformer", help="name of the base model")
args = args.parse_args()

## Get token dict
token_path = "/home/benjami/barseq-transformer/geneformer/token_dictionary.pkl"
print(f"Loading {token_path}")
with open(token_path, "rb") as fp:
    token_dictionary = pickle.load(fp)

## Load the model
base_model_path = models_path_map[args.base_model_name]
model =  BertForMaskedLM.from_pretrained(base_model_path)
print(f"Loaded model from {base_model_path}.")

# Get all gene tokens from the tokenizer
all_genes = [token for token in token_dictionary.keys() if token.startswith("E")]

# Create a dictionary to store gene embeddings
gene_embeddings = {}

# Get embeddings for all genes by in
model.eval()
with torch.no_grad():
    for gene in all_genes:
        inputs = tokenizer(gene, return_tensors="pt")
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[0, 0, :].numpy()  # Get the first token's embedding
        gene_embeddings[gene] = embedding.tolist()  # Convert to list for JSON serialization

# Save the gene embeddings to a file
with open("gene_embeddings.json", "w") as f:
    json.dump(gene_embeddings, f)

print(f"Gene embeddings for {len(gene_embeddings)} genes have been saved to gene_embeddings.json")