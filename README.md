🌍 Word Embedding from Scratch


This repository implements popular word embedding techniques from scratch using pure Python and NumPy, without relying on deep learning frameworks like PyTorch or TensorFlow.

Word embeddings are crucial in Natural Language Processing (NLP) for representing words in dense vector space where semantic similarity is preserved.

📌 Features
✨ Word2Vec

Continuous Bag of Words (CBOW)

Skip-Gram model

📉 Negative Sampling and Softmax Loss

🔡 Vocabulary building & one-hot encoding

🔍 Cosine similarity for word similarity comparison

📊 Optional: PCA/t-SNE for 2D visualization of embeddings

🧠 Algorithms Covered
CBOW (Continuous Bag of Words)
Predict the center word from surrounding context words.

Skip-Gram
Predict the context words given a center word.

Negative Sampling
Efficient training method by sampling negative examples.

🗂️ Project Structure
bash
Copy
Edit
word_embedding/
├── data/
│   └── corpus.txt                # Training corpus
├── models/
│   ├── cbow.py                   # CBOW implementation
│   └── skipgram.py               # Skip-Gram implementation
├── utils/
│   └── preprocessing.py          # Text cleaning, tokenization, vocab building
├── train.py                      # Training loop and vector updates
├── visualize.py                  # Optional: Plotting word vectors
└── README.md
