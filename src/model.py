from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class RetrievalModel:
    def __init__(self):
        # Используем TF-IDF векторайзер и алгоритм ближайших соседей
        self.vectorizer = TfidfVectorizer()
        self.model = NearestNeighbors(n_neighbors=1, metric='cosine')


    def fit(self, X):
        vectors = self.vectorizer.fit_transform(X)
        self.model.fit(vectors)


    def predict(self, X):
        vectors = self.vectorizer.transform(X)
        distances, indices = self.model.kneighbors(vectors)
        return indices.flatten()
    

# # Обучение модели
# model = RetrievalModel()
# model.fit(train_df['other_speaker'])
