import numpy as np
from scipy.spatial.distance import pdist, squareform
import argparse
import sys
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
sys.path.append("../")  # Add parent directory to Python path
from data.dataloaders import load_geneformer_embeddings, load_word2vec_embeddings
from tqdm import tqdm

def calculate_similarities(M, use_cosine=True):
    print("Calculating pairwise similarities...")
    if use_cosine:
        # Normalize the vectors for cosine similarity
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        normalized_embeddings = M / norms
        # Cosine similarity is dot product of normalized vectors
        similarities = normalized_embeddings @ normalized_embeddings.T
    else:
        # Use negative Euclidean distance for similarity
        distances = pdist(M, metric='euclidean')
        similarities = -squareform(distances)
    
    return similarities

def find_top_n_pairs(similarities, n):
    print(f"Finding top {n} pairs...")
    # Get the indices of the upper triangle of the matrix (excluding diagonal)
    i_upper, j_upper = np.triu_indices(similarities.shape[0], k=1)
    
    # Get the similarities of the upper triangle
    sim_upper = similarities[i_upper, j_upper]
    
    # Sort the similarities in descending order
    top_n_indices = np.argsort(sim_upper)[-n:][::-1]
    
    # Get the corresponding i, j indices and similarities
    top_pairs = [(i_upper[idx], j_upper[idx], sim_upper[idx]) for idx in top_n_indices]
    
    return top_pairs

def save_results(pairs, words, output_file):
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        for i, j, similarity in pairs:
            f.write(f"{words[i]} {words[j]} {similarity:.6f}\n")

def load_gene_descriptions():
    with open("data/geneformer/gene_descriptions.json", "r") as f:
        return json.load(f)

def create_histogram(similarities):
    print("Creating histogram...")
    # Subsample to 100k points
    subsample = np.random.choice(similarities.flatten(), size=100000, replace=False)
    
    plt.figure(figsize=(6, 3))
    plt.hist(subsample, bins=50, edgecolor='black')
    plt.title('Distribution of Similarities')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # Save plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64


def write_markdown_results(pairs, words, topk, output_file, use_cosine, gene_descriptions, similarities, embeddings_type):
    similarity_type = "Cosine similarity" if use_cosine else "Negative Euclidean distance"
    
    # Calculate summary statistics
    sim_mean = np.mean(similarities)
    sim_std = np.std(similarities)
    sim_min = np.min(similarities)
    sim_max = np.max(similarities)
    
    # Create histogram
    histogram_base64 = create_histogram(similarities)
    
    with open(output_file, 'w') as f:
        f.write(f"# Similarity Analysis of {embeddings_type.capitalize()} Embeddings\n\n")
        
        f.write("This analysis explores the pairwise similarities between gene embeddings generated using the "
                f"{embeddings_type} model. Gene embeddings are vector representations of genes that capture their "
                "functional and relational properties in a high-dimensional space.\n\n")
        
        f.write(f"We use {similarity_type} as our measure of similarity between gene embeddings. "
                f"For each pair of genes, we calculate their similarity score, resulting in a "
                f"similarity matrix of size {len(words)} x {len(words)}.\n\n")
        
        f.write(f"## Distribution Summary\n\n")
        f.write(f"Similarity measure: {similarity_type}\n\n")
        f.write(f"- Mean: {sim_mean:.6f}\n")
        f.write(f"- Standard Deviation: {sim_std:.6f}\n")
        f.write(f"- Minimum: {sim_min:.6f}\n")
        f.write(f"- Maximum: {sim_max:.6f}\n\n")
        
        f.write("## Similarity Distribution Histogram\n\n")
        f.write(f"![Similarity Distribution](data:image/png;base64,{histogram_base64})\n\n")
        
        f.write(f"## Top {topk} Most Similar Pairs\n\n")
        f.write("| Gene 1 | Gene 2 | Similarity |\n")
        f.write("|--------|--------|------------|\n")
        
        for i, j, similarity in pairs:
            w1, w2 = words[i], words[j]
            f.write(f"| [{w1}](https://www.proteinatlas.org/{w1}) | [{w2}](https://www.proteinatlas.org/{w2}) | {similarity:.6f} |\n")
            f.write(f"| {gene_descriptions.get(w1, 'N/A')} | {gene_descriptions.get(w2, 'N/A')} | |\n")


def main(args):
    print("-"*50)
    print("|         Pairwise Similarity Calculator         |")
    print("-"*50)
    
    # Load embeddings
    if args.embeddings == "geneformer":
        M, words = load_geneformer_embeddings(args.path)
    elif args.embeddings == "word2vec":
        M, words = load_word2vec_embeddings(args.path)
    else:
        raise ValueError(f"Unknown embeddings: {args.embeddings}")

    # Truncate embeddings and words if specified
    if args.truncate > 0:
        M = M[:args.truncate]
        words = words[:args.truncate]

    print(f" Loaded {args.embeddings} embeddings from {args.path}")
    print(f" Using {len(words)} embeddings of dimension {M.shape[1]}")
    print(" Using cosine similarity" if args.use_cosine else " Using Euclidean distance")

    # Calculate similarities
    similarities = calculate_similarities(M, args.use_cosine)

    # Find top N pairs
    top_pairs = find_top_n_pairs(similarities, args.topk)

    # Save results
    save_results(top_pairs, words, args.output_file)
    print(f"\nResults saved to {args.output_file}")

    # If using geneformer embeddings, also save markdown results
    if args.embeddings == "geneformer":
        gene_descriptions = load_gene_descriptions()
        write_markdown_results(top_pairs, words, args.topk, "outputs.md", args.use_cosine, gene_descriptions, similarities, args.embeddings)
        print("\nDetailed results saved to outputs.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top N similar pairs of embeddings")
    parser.add_argument("--embeddings", type=str, default="word2vec", help="Embeddings to use (geneformer or word2vec)")
    parser.add_argument("--path", type=str, required=True, help="Path to embeddings")
    parser.add_argument("--topk", type=int, default=1000, help="Number of top pairs to return")
    parser.add_argument("--output-file", type=str, default="top_pairs.txt", help="File to save the results")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity (default is Euclidean distance)")
    parser.add_argument("--truncate", type=int, default=0, help="Truncate to first N words (0 means no truncation)")

    args = parser.parse_args()
    main(args)