from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class RetrievalModel:
    def __init__(self, n_neighbors=1):
        self.vectorizer = TfidfVectorizer()
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')

    def fit(self, X):
        vectors = self.vectorizer.fit_transform(X)
        self.model.fit(vectors)

    def predict(self, X):
        vectors = self.vectorizer.transform(X)
        distances, indices = self.model.kneighbors(vectors)
        return indices.flatten()

    def save_model(self, model_path='retrieval_model.pkl'):
        joblib.dump((self.vectorizer, self.model), model_path)

    def load_model(self, model_path='retrieval_model.pkl'):
        self.vectorizer, self.model = joblib.load(model_path)


from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def evaluate_bleu_rouge(y_true, y_pred):
    # Преобразуем y_true и y_pred в списки строк
    y_true = list(y_true)
    y_pred = list(y_pred)

    # Проверка на совпадение длин
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred.")

    # Используем сглаживание для BLEU
    smoothing = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu([true.split()], pred.split(), smoothing_function=smoothing)
        for true, pred in zip(y_true, y_pred)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    # ROUGE Score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(y_pred, y_true, avg=True)

    return avg_bleu, rouge_scores


def plot_bleu_rouge_scores(avg_bleu, rouge_scores):
    print(f"Average BLEU Score: {avg_bleu}")
    print(f"ROUGE Scores: {rouge_scores}")

    # График для BLEU
    plt.figure(figsize=(10, 5))
    plt.bar(['BLEU'], [avg_bleu])
    plt.title("BLEU Score")
    plt.ylim(0, 1)
    plt.show()

    # График для ROUGE
    rouge_metrics = rouge_scores['rouge-1']
    sns.barplot(x=list(rouge_metrics.keys()), y=list(rouge_metrics.values()))
    plt.title("ROUGE-1 Scores")
    plt.ylim(0, 1)
    plt.show()