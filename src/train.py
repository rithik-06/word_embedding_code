from gensim.models import Word2Vec
from .utils import read_corpus
from .config import DATA_PATH, EMBEDDINGS_PATH

def train_word2vec():
    sentences = list(read_corpus(DATA_PATH))
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2, epochs=20)
    model.save(EMBEDDINGS_PATH)
    print(f"Model saved to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    train_word2vec() 