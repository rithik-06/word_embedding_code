from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from .config import EMBEDDINGS_PATH

def visualize_embeddings():
    model = Word2Vec.load(EMBEDDINGS_PATH)
    words = list(model.wv.index_to_key)
    vectors = model.wv[words]

    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(Y[:, 0], Y[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(Y[i, 0], Y[i, 1]))
    plt.title("Word Embeddings Visualization (t-SNE)")
    plt.show()

if __name__ == "__main__":
    visualize_embeddings() 