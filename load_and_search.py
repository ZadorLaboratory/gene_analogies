import numpy as np
import faiss
from typing import List, Tuple
import pickle
import os
import argparse
import sys
sys.path.append("../")  # Add parent directory to Python path
from data.dataloaders import load_geneformer_embeddings, load_word2vec_embeddings  # Import custom data loading functions
import heapq
from tqdm import tqdm

def create_summed_vectors(M, take_difference=True, start_idx=0, end_idx=None, triu_indices=None):
    n, d = M.shape
    if triu_indices is None:
        triu_indices = np.triu_indices(n, 1)
    else:
        assert len(triu_indices) == 2, "triu_indices must be a tuple of length 2"
        assert triu_indices[0].shape[0] == n * (n - 1) // 2, "triu_indices must have the correct number of elements"

    if end_idx is None:
        end_idx = len(triu_indices[0])
    
    i, j = triu_indices
    i = i[start_idx:end_idx]
    j = j[start_idx:end_idx]
    
    if take_difference:
        result = M[i] - M[j]
    else:
        result = M[i] + M[j]
    print(f"Created {len(result)} summed vectors")

    return np.ascontiguousarray(result, dtype=np.float32), (i, j)

def load_index_and_search(M: np.ndarray, words: List[str], index_file: str, batch_size=10000,
                          topk: int = 5000, output_dir: str = "output", use_cosine: bool = True, 
                          take_difference: bool = False, cpu_only: bool = False) -> List[Tuple[str, str, str, str, float]]:
    
    # Prepare summed vectors for search
    print("Preparing summed vectors for search...")
    n, d = M.shape
    triu_indices = np.triu_indices(n, 1)

    # Load the index
    print(f"Loading index from {index_file}...")
    index = faiss.read_index(index_file)
    
    # Set search parameters
    index.nprobe = 64  # Adjust this value as needed
    
    print("Searching for similar pairs...")
    if not cpu_only:
        # Get the number of GPUs
        ngpus = faiss.get_num_gpus()
        print(f"Number of GPUs: {ngpus}")
        
        # Convert to GPU index
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        
        # Set search parameters
        gpu_index.nprobe = 64  # Adjust this value as needed
    else:
        print("Using CPU-only mode")
        gpu_index = index  # Use the CPU index directly
    
    print("Searching for similar pairs...")
    total_vectors = len(triu_indices[0])

    n_batches = (total_vectors + batch_size - 1) // batch_size
    
    similarity_heap = []
    for ii, start_idx in enumerate(range(0, total_vectors, batch_size)):
        end_idx = min(start_idx + batch_size, total_vectors)
        print(f"Batch {ii + 1}/{n_batches} ({start_idx}-{end_idx} = {end_idx - start_idx} vectors)")
        batch_vectors, _ = create_summed_vectors(M, take_difference, start_idx, end_idx, triu_indices)

        if use_cosine:
        # Normalize the summed vectors for cosine similarity
            faiss.normalize_L2(batch_vectors)
        
        D, I = gpu_index.search(batch_vectors, 501)  # Search for top 51 (including self)
        
        for query_idx, (distances, indices) in enumerate(zip(D, I)):
            i, j = triu_indices[0][start_idx + query_idx], triu_indices[1][start_idx + query_idx]
            for dist, idx in zip(distances[1:], indices[1:]):  # Skip the first result (self-match)
                k, l = triu_indices[0][idx], triu_indices[1][idx]
                if i > j or k > l or i > k:  # Ensure lexicographic order
                    continue
                if len({i, j, k, l}) == 4:  # Ensure all indices are different
                    similarity = dist if use_cosine else -dist  # For L2, smaller distance means more similar
                    if len(similarity_heap) < topk:
                        heapq.heappush(similarity_heap, (similarity, i, j, k, l, start_idx + query_idx))
                    elif similarity > similarity_heap[0][0]:
                        heapq.heapreplace(similarity_heap, (similarity, i, j, k, l, start_idx + query_idx))
    
    # Sort the results (heapq is a min-heap, so we need to reverse it)
    results = sorted(similarity_heap, reverse=True)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the sorted top-k list to a separate file
    topk_file = os.path.join(output_dir, "_top_k_pairs.txt")
    print(f"Saving top {topk} pairs to {topk_file}")
    with open(topk_file, 'w') as f:
        f.write(f"Top {topk} most similar pairs:\n\n")
        for rank, (sim, i, j, k, l, _) in enumerate(results, 1):
            if take_difference:
                f.write(f"{rank}. {words[i]} - {words[j]} ≈ {words[k]} - {words[l]} (similarity: {sim:.4f})\n")
            else:
                f.write(f"{rank}. {words[i]} + {words[j]} ≈ {words[k]} + {words[l]} (similarity: {sim:.4f})\n")
    
    return [(sim, words[i], words[j], words[k], words[l]) for sim, i, j, k, l, _ in results]

def main(args):
    # Load embeddings
    if args.embeddings == "geneformer":
        M, words = load_geneformer_embeddings(args.path, brain=not args.body_order)  # Load Geneformer embeddings
    elif args.embeddings == "word2vec":
        M, words = load_word2vec_embeddings(args.path)  # Load Word2Vec embeddings
    else:
        raise ValueError(f"Unknown embeddings: {args.embeddings}")  # Raise error for unknown embedding type
    print(f"Loaded {len(words)} embeddings of dimension {M.shape[1]}")  # Log successful loading

    # Truncate the number of embeddings
    if args.truncate > 0:
        M = M[:args.truncate]
        words = words[:args.truncate]
        print(f"Truncated to {args.truncate} embeddings")    

    similar_pairs = load_index_and_search(M, words, args.index_file, args.N,
                                          args.topk, args.output_dir, args.use_cosine, 
                                          args.difference, args.cpu_only)
    
    print(f"Top {min(5, args.topk)} most similar pairs:")
    for similarity, word1, word2, word3, word4 in similar_pairs[:5]:
        print(f"{word1} + {word2} ≈ {word3} + {word4} (similarity: {similarity:.4f})")
    print(f"\nFull list of top {args.topk} pairs saved to {os.path.join(args.output_dir, 'top_k_pairs.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load FAISS index and search for similar pairs")
    parser.add_argument("--index-file", type=str, default="faiss_index_word2vec/trained_index.faiss", help="File containing the trained index")
    parser.add_argument("--topk", type=int, default=500, help="Number of top pairs to return")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save detailed output files")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity (default is L2 distance)")
    parser.add_argument("--N", type=int, default=2**17, help="Batch size for query")
    parser.add_argument("--truncate", type=int, default=10000, help="Truncate the number of embeddings to use")
    parser.add_argument("--embeddings", type=str, default="word2vec", help="Embeddings to use (geneformer or word2vec)")
    parser.add_argument("--path", type=str, default="/home/benjami/barseq-transformer/gene_embeddings/data", help="Path to embeddings")
    parser.add_argument("--difference", action="store_true", help="Cluster and search on difference of vectors instead of sum")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only for search (no GPU)")
    parser.add_argument("--body-order", action="store_true", help="Select the top [truncate] genes in the body instead of brain")

    args = parser.parse_args()
    
    main(args)