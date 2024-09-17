import numpy as np
import faiss
from typing import List, Tuple
import argparse
import sys
import json
sys.path.append("../")  # Add parent directory to Python path
from data.dataloaders import load_geneformer_embeddings, load_word2vec_embeddings
from tqdm import tqdm

def create_summed_vector(M: np.ndarray, word1: str, word2: str, words: List[str], take_difference: bool = False) -> Tuple[np.ndarray, int, int]:
    try:
        idx1 = words.index(word1)
    except ValueError:
        raise ValueError(f"'{word1}' is not in the vocabulary.")
    try:
        idx2 = words.index(word2)
    except ValueError:
        raise ValueError(f"'{word2}' is not in the vocabulary.")
    
    if take_difference:
        return M[idx1] - M[idx2], idx1, idx2
    else:
        return M[idx1] + M[idx2], idx1, idx2

def load_index_and_prepare_search(index_file: str, use_cosine: bool = True, cpu_only: bool = False):
    print(f" Loading index from {index_file}...")
    index = faiss.read_index(index_file)
    
    # Set search parameters
    index.nprobe = 512  # Adjust this value as needed

    print(" Preparing for search...")
    if cpu_only:
        print(" Using CPU-only mode")
        return index, use_cosine
    else:
        # Get the number of GPUs
        ngpus = faiss.get_num_gpus()
        print(f" Number of GPUs: {ngpus}")
        
        # Convert to GPU index
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        
        return gpu_index, use_cosine

def triu_index(n, idx):
    # Calculate the row (k) and column (l) for a given index
    k = int((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * idx)) / 2)
    l = idx + k + 1 - k * (2 * n - k - 1) // 2
    return k, l

def search_similar_pairs(index, M: np.ndarray, words: List[str], query_pair: Tuple[str, str],
                         topk: int = 10, use_cosine: bool = True, take_difference: bool = False) -> List[Tuple[float, str, str, str, str]]:
    
    # Create query vector
    query_vector, idx1, idx2 = create_summed_vector(M, query_pair[0], query_pair[1], words, take_difference)
    query_vector = query_vector.reshape(1, -1).astype(np.float32)

    if use_cosine:
        # Normalize the query vector for cosine similarity
        faiss.normalize_L2(query_vector)
    
    # Perform the search
    n = len(words)
    D, I = index.search(query_vector, topk * n + 1)  # more to account for self-match

    reversed_flags = np.zeros(topk * n + 1, dtype=bool)
    if take_difference:
        # perform the search again with the reversed query
        query_vector = -query_vector
        D2, I2 = index.search(query_vector, topk * n + 1)

        # check if the similarity is higher with the reversed query
        for i in range(1, topk * n + 1):
            if D2[0][i] > D[0][i]:
                D[0][i] = D2[0][i]
                I[0][i] = I2[0][i]
                reversed_flags[i] = True

    results = []
    
    z = 0
    for i, (dist, idx) in enumerate(zip(D[0], I[0])): 
        k, l = triu_index(n, idx)
        if len({idx1, idx2, k, l}) == 4:  # Ensure all indices are different
            similarity = dist if use_cosine else -dist  # For L2, smaller distance means more similar
            if reversed_flags[i]:
                k, l = l, k
            results.append((similarity, words[k], words[l], words[idx1], words[idx2]))
            z += 1
        if z>topk:
            break
        
    return results[:topk]  # Return top k results


def load_gene_descriptions():
    with open("data/geneformer/gene_descriptions.json", "r") as f:
        return json.load(f)

def write_markdown_results(similar_pairs, word1, word2, topk, output_file, take_difference, gene_descriptions):
    operation = "-" if take_difference else "+"
    
    with open(output_file, 'w') as f:
        f.write(f"# Query Results\n\n")
        f.write(f"Query pair: {word1} {operation} {word2}\n\n")
        f.write(f"Top {topk} most similar pairs:\n\n")
        
        f.write("| Gene 1 | Gene 2 | ≈ | Gene 3 | Gene 4 | Similarity |\n")
        f.write("|--------|--------|---|--------|--------|------------|\n")
        
        for similarity, w1, w2, q1, q2 in similar_pairs:
            f.write(f"| [{w1}](https://www.proteinatlas.org/{w1}) | [{w2}](https://www.proteinatlas.org/{w2}) | ≈ | [{q1}](https://www.proteinatlas.org/{q1}) | [{q2}](https://www.proteinatlas.org/{q2}) | {similarity:.4f} |\n")
            f.write(f"| {gene_descriptions.get(w1, 'N/A')} | {gene_descriptions.get(w2, 'N/A')} | | {gene_descriptions.get(q1, 'N/A')} | {gene_descriptions.get(q2, 'N/A')} | |\n")

def main(args):
    print("-"*50)
    print("|              Automated analogizer              |")
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

    print(" Using cosine similarity" if args.use_cosine else " Using L2 distance")  
    print(" Using difference of vectors" if args.difference else " Using sum of vectors")
    print(" Please ensure that the index was trained with the same settings.")

    # Load index and prepare for search
    index, use_cosine = load_index_and_prepare_search(args.index_file, args.use_cosine, args.cpu_only)

    # Load gene descriptions
    gene_descriptions = load_gene_descriptions()

    if args.query_file:
        # Single query mode
        with open(args.query_file, 'r') as f:
            word1, word2 = f.read().strip().split()
        try:
            similar_pairs = search_similar_pairs(index, M, words, (word1, word2),
                                                 args.topk, use_cosine, args.difference)
            print_and_save_results(similar_pairs, word1, word2, args.topk, args.output_file, args.difference)
            if args.embeddings == "geneformer":
                write_markdown_results(similar_pairs, word1, word2, args.topk, "outputs.md", args.difference, gene_descriptions)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        while True:
            word1 = input("Enter the first word of the pair (or 'quit' to exit): ")
            if word1.lower() == 'quit':
                break
            word2 = input("Enter the second word of the pair: ")
            
            try:
                similar_pairs = search_similar_pairs(index, M, words, (word1, word2),
                                                     args.topk, use_cosine, args.difference)
                print_results(similar_pairs, word1, word2, args.topk, args.difference)
                if args.embeddings == "geneformer":
                    write_markdown_results(similar_pairs, word1, word2, args.topk, "outputs.md", args.difference, gene_descriptions)
                print("\nResults also saved to outputs.md")
            except ValueError as e:
                print(f"Error: {e}")
                print("Please try again with different words.")
            
            print("\n" + "-"*50 + "\n")

def print_results(similar_pairs, word1, word2, topk, take_difference):
    operation = "-" if take_difference else "+"
    print(f"\nTop {topk} pairs most similar to '{word1} {operation} {word2}':")
    for similarity, w1, w2, q1, q2 in similar_pairs:
        print(f"{w1} {operation} {w2} ≈ {q1} {operation} {q2} (similarity: {similarity:.4f})")

def print_and_save_results(similar_pairs, word1, word2, topk, output_file, take_difference):
    print_results(similar_pairs, word1, word2, topk, take_difference)
    
    operation = "-" if take_difference else "+"
    with open(output_file, 'w') as f:
        f.write(f"Query pair: {word1} {operation} {word2}\n\n")
        f.write(f"Top {topk} most similar pairs:\n\n")
        for similarity, w1, w2, q1, q2 in similar_pairs:
            f.write(f"{w1} {operation} {w2} ≈ {q1} {operation} {q2} (similarity: {similarity:.4f})\n")
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query similar pairs using FAISS index")
    parser.add_argument("--index-file", type=str, required=True, help="File containing the trained index")
    parser.add_argument("--embeddings", type=str, default="word2vec", help="Embeddings to use (geneformer or word2vec)")
    parser.add_argument("--path", type=str, required=True, help="Path to embeddings")
    parser.add_argument("--topk", type=int, default=10, help="Number of top pairs to return")
    parser.add_argument("--output-file", type=str, default="query_results.txt", help="File to save the results (only used with --query-file)")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity (default is L2 distance)")
    parser.add_argument("--difference", action="store_true", help="Use difference of vectors instead of sum")
    parser.add_argument("--query-file", type=str, help="File containing the query pair (optional)")
    parser.add_argument("--truncate", type=int, default=0, help="Truncate to first N words (0 means no truncation)")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only for search (no GPU)")

    args = parser.parse_args()
    
    main(args)