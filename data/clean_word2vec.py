import os
import numpy as np
import jax.numpy as jnp
import re
from tqdm import tqdm

def load_word2vec_embeddings(path="/home/benjami/barseq-transformer/gene_embeddings/data"):
    emb_path = os.path.join(path, "word2vec", "google_news_embeddings.npy")
    embeddings = np.load(emb_path)
    words_path = os.path.join(path, "word2vec", "google_news_words.npy")
    words = np.load(words_path)
    return jnp.array(embeddings), list(words)

def is_valid_word(word):
    return bool(re.match(r'^[a-z]+$', word))

def clean_embeddings(embeddings, words):
    word_to_index = {}
    cleaned_words = []
    cleaned_embeddings = []

    for idx, word in tqdm(enumerate(words)):
        word = word.lower()
        if is_valid_word(word) and word not in word_to_index:
            word_to_index[word] = len(cleaned_words)
            cleaned_words.append(word)
            cleaned_embeddings.append(embeddings[idx])

    return np.array(cleaned_embeddings), cleaned_words

def save_cleaned_embeddings(embeddings, words, path="/home/benjami/barseq-transformer/gene_embeddings/data"):
    cleaned_emb_path = os.path.join(path, "word2vec", "cleaned_google_news_embeddings.npy")
    cleaned_words_path = os.path.join(path, "word2vec", "cleaned_google_news_words.npy")
    
    np.save(cleaned_emb_path, embeddings)
    np.save(cleaned_words_path, words)
    
    print(f"Cleaned embeddings saved to: {cleaned_emb_path}")
    print(f"Cleaned words saved to: {cleaned_words_path}")

    #also save words to a text file
    words_text_path = os.path.join(path, "word2vec", "cleaned_google_news_words.txt")
    with open(words_text_path, 'w') as f:
        for word in words:
            f.write(f"{word}\n")

def main():
    # Load original embeddings and words
    embeddings, words = load_word2vec_embeddings()
    
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"Original words count: {len(words)}")
    
    # Clean embeddings and words
    cleaned_embeddings, cleaned_words = clean_embeddings(embeddings, words)
    
    print(f"Cleaned embeddings shape: {cleaned_embeddings.shape}")
    print(f"Cleaned words count: {len(cleaned_words)}")
    
    # Save cleaned embeddings and words
    save_cleaned_embeddings(cleaned_embeddings, cleaned_words)

if __name__ == "__main__":
    main()