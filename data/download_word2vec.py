import gensim
import gensim.downloader as api
import numpy as np
import json

# Download and load the pre-trained model
model = api.load('word2vec-google-news-300')

# Extract words and embeddings
words = list(model.key_to_index.keys())
embeddings = [model[word] for word in words]

# Convert embeddings to NumPy array
embeddings_array = np.array(embeddings)

# Save embeddings as .npy file
np.save('google_news_embeddings.npy', embeddings_array)
np.save('google_news_words.npy', words)


print("Embeddings saved as 'google_news_embeddings.npy'")
