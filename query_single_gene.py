import numpy as np
import faiss
from typing import List, Tuple
import pickle
import os
import argparse
import sys
sys.path.append("../")
from data.dataloaders import load_geneformer_embeddings, load_word2vec_embeddings
import heapq
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_index(index_file: str, cpu_only: bool = False):
    
    # Load the index
    print(f"Loading index from {index_file}...")
    index = faiss.read_index(index_file)
    
    # Set search parameters
    index.nprobe = 64  # Adjust this value as needed
    
    if not cpu_only and faiss.get_num_gpus() > 0:
        print(f"Number of GPUs: {faiss.get_num_gpus()}")
        gpu_index = faiss.index_cpu_to_all_gpus(index)
    else:
        print("Using CPU-only mode")
        gpu_index = index  # Use the CPU index directly
    
    return gpu_index

def vectorized_upper_triangle(i, N):
    """
    Generate a boolean vector indicating whether (i, j) is in the upper triangle
    for all possible j in an N x N matrix.
    
    Args:
    i (int): Row index (0-based)
    N (int): Size of the square matrix
    
    Returns:
    np.array: Vector of length N, where +1 indicates (i, j) is in the upper triangle and -1 is lower triangle
    """
    j_values = np.arange(N)
    return ((i < j_values) & (i < N) & (j_values < N)).astype(int)*2-1

def triu_index(n, idx):
    # Calculate the row (k) and column (l) for a given index
    k = int((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * idx)) / 2)
    l = idx + k + 1 - k * (2 * n - k - 1) // 2
    return k, l

def find_analogies(query_word: str, M: np.ndarray, words: List[str], index: faiss.Index, topk: int = 10,
                   use_cosine: bool = False, vector_sum: bool = False
                   ) -> List[Tuple[str, str, float]]:
    query_index = words.index(query_word)
    query_vector = M[query_index].astype(np.float32)
    
    # Compute all differences vectorized
    if vector_sum:
        query_vector = query_vector + M
        # triu_or_tril = np.ones(len(words))
    else:
        diff_vectors = M - query_vector

        # If [query_index, i] is in the tril part of the pairwise matrix,
        # Then we need to reverse the order of the words for the search
        # and then reverse the order of the results at the end
        # triu_or_tril = vectorized_upper_triangle(query_index, len(M))
        # diff_vectors = triu_or_tril.reshape(-1,1) * diff_vectors
    triu_or_tril = np.ones(len(words))
    diff_vectors = diff_vectors.astype(np.float32)

    # Normalize the difference vectors for cosine similarity
    if use_cosine:
        faiss.normalize_L2(diff_vectors)
    
    # Perform the search
    D, I = index.search(diff_vectors, topk + 2)  # + 2 to account for query_word and self 

    results = []

    for query_B_idx, (distances, indices) in enumerate(zip(D, I)):
        for dist, idx in zip(distances[1:], indices[1:]):  # Skip the first result (self-match)
            k, l = triu_index(len(M), idx)
            if len({query_index, query_B_idx, k, l}) == 4:  # Ensure all indices are different
                similarity = dist if use_cosine else -dist  # For L2, smaller distance means more similar
                if triu_or_tril[query_B_idx] < 0: # switch order if switched earlier
                    results.append((similarity, words[query_index], words[query_B_idx], words[l], words[k]))
                else:
                    results.append((similarity, words[query_B_idx], words[query_index], words[k], words[l]))

    return sorted(results, key=lambda x: x[0], reverse=True)[:topk]

def load_gene_descriptions():
    with open("data/geneformer/gene_descriptions.json", "r") as f:
        return json.load(f)

def cluster_and_visualize(analogies: List[Tuple], M: np.ndarray, words: List[str], n_clusters: int = 5):
    # Extract A-B vectors
    ab_vectors = []
    for _, a, b, _, _ in analogies:
        a_idx, b_idx = words.index(a), words.index(b)
        ab_vector = M[a_idx] - M[b_idx]
        ab_vectors.append(ab_vector)
    
    ab_vectors = np.array(ab_vectors)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(ab_vectors)
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(ab_vectors)
    
    # Sort by cluster labels
    sorted_indices = np.argsort(cluster_labels)
    sorted_similarities = similarities[sorted_indices][:, sorted_indices]
    sorted_labels = cluster_labels[sorted_indices]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sorted_similarities, cmap='viridis', cbar_kws={'label': 'Cosine Similarity'})
    
    # Add cluster separators and labels
    current_tick = 0
    for cluster in range(n_clusters):
        cluster_size = np.sum(sorted_labels == cluster)
        if cluster_size > 0:
            # Add a line to separate clusters
            ax.axhline(y=current_tick, color='red', linewidth=2)
            ax.axvline(x=current_tick, color='red', linewidth=2)
            
            # Add cluster label
            plt.text(-0.05, current_tick + cluster_size/2, f'Cluster {cluster}', 
                     verticalalignment='center', horizontalalignment='right',
                     rotation=90, fontsize=10, fontweight='bold')
            
            current_tick += cluster_size
    
    # Add a final line at the bottom/right
    ax.axhline(y=len(sorted_labels), color='red', linewidth=2)
    ax.axvline(x=len(sorted_labels), color='red', linewidth=2)
    plt.yticks([])
    plt.title('Pairwise Similarities of A-B Vectors (Clustered and Annotated)')
    plt.xlabel('Analogy Index')
    plt.ylabel('Analogy Index')
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save the plot
    heatmap_file = os.path.join("outputs", "ab_vector_similarities_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cluster_labels, heatmap_file

def interactive_mode(M: np.ndarray, words: List[str], index: faiss.Index, gene_descriptions: dict, embeddings_type: str, topk: int = 10,
                     use_cosine: bool = False, vector_sum: bool = False): 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join("outputs", f"interactive_analogies_{timestamp}.md")
    
    with open(output_file, 'w') as f:
        f.write(f"# Interactive Analogy Analysis\n\n")
        f.write(f"Using {embeddings_type} embeddings\n")

    print(f"Interactive session results will be saved to: {output_file}")

    op = "+" if vector_sum else "-"

    while True:
        query_word = input("Enter a word to find analogies (or 'quit' to exit): ")
        if query_word.lower() == 'quit':
            break
        
        try:
            analogies = find_analogies(query_word, M, words, index, topk, use_cosine, vector_sum)
            
            # Perform clustering and visualization
            cluster_labels, heatmap_file = cluster_and_visualize(analogies, M, words)
            
        except ValueError:
            print(f"Word '{query_word}' not found in the vocabulary.")
            continue
        
        with open(output_file, 'a') as f:
            if not analogies:
                print(f"No analogies found for '{query_word}'.")
                f.write(f"\n## Query: {query_word}\n\nNo analogies found.\n")
            else:
                query_description = gene_descriptions.get(query_word, 'N/A').split(" [Source")[0]
                print(f"\n\nTop {len(analogies)} analogies for {query_word} ({query_description}):\n")
                f.write(f"## Top {len(analogies)} analogies for [{query_word}](https://www.proteinatlas.org/{query_word}) ({query_description}):\n\n")
                f.write(f"![A-B Vector Similarities Heatmap]({heatmap_file})\n\n")
                f.write("| Analogy | Score | Cluster | Description A | Description B | Description C | Description D |\n")
                f.write("|---------|-------|---------|---------------|---------------|---------------|---------------|\n")

                print("| Analogy | Score | Cluster | Description A | Description B | Description C | Description D |")
                print("|---------|-------|---------|---------------|---------------|---------------|---------------|")
                for (score, A, B, C, D), cluster in zip(analogies, cluster_labels):
                    desc1 = gene_descriptions.get(A, 'N/A').split(" [Source")[0]
                    desc2 = gene_descriptions.get(B, 'N/A').split(" [Source")[0]
                    desc3 = gene_descriptions.get(C, 'N/A').split(" [Source")[0]
                    desc4 = gene_descriptions.get(D, 'N/A').split(" [Source")[0]

                    print(f"| {A} {op} {B} ~ {C} {op} {D} | {score:.6f} | {cluster} | {desc1} | {desc2} | {desc3} | {desc4} |")
                    f.write(f"| [{A}](https://www.proteinatlas.org/{A}) {op} [{B}](https://www.proteinatlas.org/{B}) ~ [{C}](https://www.proteinatlas.org/{C}) {op} [{D}](https://www.proteinatlas.org/{D}) | {score:.6f} | {cluster} | {desc1} | {desc2} | {desc3} | {desc4} |\n")

        print("\n" + "-"*50 + "\n")

    print(f"\nInteractive session results have been saved to: {output_file}")

def load_embeddings(embeddings_type, path, truncate=0):
    if embeddings_type == "geneformer":
        M, words = load_geneformer_embeddings(path)
    elif embeddings_type == "word2vec":
        M, words = load_word2vec_embeddings(path)
    else:
        raise ValueError(f"Unknown embeddings: {embeddings_type}")
    
    if truncate > 0:
        M = M[:truncate]
        words = words[:truncate]
    
    return M, words

def main(args):
    print("-"*50)
    print("|            Interactive Analogy Finder           |")
    print("-"*50)
    
    # Load embeddings
    M, words = load_embeddings(args.embeddings, args.path, args.truncate)

    print(f" Loaded {args.embeddings} embeddings from {args.path}")
    print(f" Using {len(words)} embeddings of dimension {M.shape[1]}")

    # Load index and prepare for search
    index = load_index(args.index_file, args.cpu_only)

    os.makedirs("outputs", exist_ok=True)

    gene_descriptions = load_gene_descriptions() if args.embeddings == "geneformer" else {}
    interactive_mode(M, words, index, gene_descriptions, args.embeddings, args.topk, args.use_cosine, args.sum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find analogies interactively using a pre-trained FAISS index")
    parser.add_argument("--embeddings", type=str, default="word2vec", help="Embeddings to use (geneformer or word2vec)")
    parser.add_argument("--path", type=str, required=True, help="Path to embeddings")
    parser.add_argument("--index-file", type=str, required=True, help="Path to pre-trained FAISS index")
    parser.add_argument("--topk", type=int, default=10, help="Number of top analogies to return")
    parser.add_argument("--truncate", type=int, default=0, help="Truncate to first N words (0 means no truncation)")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity (default is L2 distance)")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only for search (no GPU)")
    parser.add_argument("--sum", action="store_true", help="Use sum of vectors instead of difference")

    args = parser.parse_args()
    main(args)