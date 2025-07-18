from src.train import train_word2vec
from src.visualize import visualize_embeddings

if __name__ == "__main__":
    print("Training word embeddings...")
    train_word2vec()
    print("Visualizing embeddings...")
    visualize_embeddings() 