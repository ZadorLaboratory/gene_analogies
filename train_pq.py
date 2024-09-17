import numpy as np
import faiss
from faiss.contrib import clustering
from typing import List, Tuple
from tqdm import tqdm
import pickle
import os
import sys
sys.path.append("../")  # Add parent directory to Python path
from data.dataloaders import load_geneformer_embeddings, load_word2vec_embeddings  # Import custom data loading functions
import argparse

def create_summed_vectors(M, take_difference=True, blocks:int = 1, i:int=0):
    n, d = M.shape
    triu_indices = np.triu_indices(n, 1)

    block_size = len(triu_indices[0]) // blocks
    start = i * block_size
    end = (i + 1) * block_size
    
    i, j = triu_indices
    i = i[start:end]
    j = j[start:end]
    
    if take_difference:
        result = M[i] - M[j]
    else:
        result = M[i] + M[j]
    print(f"Created {len(result)} summed vectors")

    return np.ascontiguousarray(result, dtype=np.float32)

def create_and_train_index(M: np.ndarray, index_file: str, 
                           use_cosine: bool = False, difference: bool = False,
                           cpu_only: bool = False, blocks: int = 1):
    
    n, d = M.shape

    n_index = n * (n - 1) // 2

    # Initalize index
    if n_index < 1e5:
        options = "HNSW32"
    else:
        k = 8 * int(np.sqrt(n_index))
        options = f"IVF{k},Flat"
    
    # Initialize FAISS index
    print("Initializing FAISS index...")
    metric = faiss.METRIC_INNER_PRODUCT if use_cosine else faiss.METRIC_L2

    cpu_index = faiss.index_factory(d, options, metric)
    
    M = np.array(M.astype(np.float32)) # Convert to float32 as required by FAISS

    for i in range(blocks):
        additions = create_summed_vectors(M, difference, blocks, i)

        if use_cosine:
        # Normalize the vectors
            faiss.normalize_L2(additions)

        if cpu_only:
            print("Using CPU-only mode for training and adding vectors...")
            # Train the index on CPU
            if i == 0:
                print("Training index on CPU...")
                cpu_index.train(additions)
            
            # Add vectors to the index
            print("Adding vectors to index...")
            cpu_index.add(additions)
        else:
            if i == 0:
                # Get the number of GPUs
                ngpus = faiss.get_num_gpus()
                print(f"Number of GPUs: {ngpus}")
                
                # Create a GPU resource object for each GPU
                gpu_resources = [faiss.StandardGpuResources() for i in range(ngpus)]
                
                # Configure GPU options
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True  # Use float16 to reduce memory usage
                
                # Make a GPU index
                gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
                
                # Train the index
                print("Training index on all GPUs...")
                gpu_index.train(additions)
                print("Copying index back to CPU...")
                cpu_index = faiss.index_gpu_to_cpu(gpu_index)

            # Add vectors to the index
            print("Adding vectors to index...")
            cpu_index.add(additions)

    # Save the index
    print(f"Saving index to {index_file}...")
    faiss.write_index(cpu_index, index_file)
    
    print("Index training and saving completed.")

def main(args):

    # Set GPU
    if args.gpu != 'all' and not args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
    
    index_file = os.path.join(args.save_path, "trained_index.faiss")
    
    create_and_train_index(M, index_file, args.use_cosine, args.difference, args.cpu_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the most similar pairs of summed embeddings.")
    parser.add_argument("--embeddings", type=str, default="word2vec", help="Embeddings to use (geneformer or word2vec)")
    parser.add_argument("--truncate", type=int, default=0, help="Truncate the number of embeddings to use")
    parser.add_argument("--path", type=str, default="/home/benjami/barseq-transformer/gene_embeddings/data", help="Path to embeddings")
    parser.add_argument("--save-path", type=str, default="./faiss_index_word2vec", help="Path to save checkpoints")
    parser.add_argument("--gpu", type=str, default="all", help="GPU to use")
    parser.add_argument("--use-cosine", action="store_true", help="Use cosine similarity (default is L2 distance)")
    parser.add_argument("--difference", action="store_true", help="Cluster and search on difference of vectors instead of sum")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU only for index creation and training")
    parser.add_argument("--blocks", type=int, default=1, help="Number of blocks to divide data over.")
    parser.add_argument("--body-order", action="store_true", help="Select the top [truncate] genes in the body instead of brain")

    args = parser.parse_args()  # Parse command line arguments
    main(args)