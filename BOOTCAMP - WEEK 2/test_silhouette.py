import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load data (mimic notebook preprocessing)
df = pd.read_csv(r'c:\laragon\www\LKS\AI - 2026\dataset\penguins.csv')
df.dropna(inplace=True)
df['sex'] = df['sex'].map({'MALE': 0, 'FEMALE': 1})

# Clean data (drop the row with sex='.' that became NaN after mapping)
df_clean = df.dropna().copy()
X_clean = df_clean[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm',
                    'body_mass_g', 'sex']].values.astype(float)

min_vals = X_clean.min(axis=0)
max_vals = X_clean.max(axis=0)
rentang = max_vals - min_vals
rentang = np.where(rentang == 0, 1.0, rentang)
X_s = (X_clean - min_vals) / rentang

# Custom KMeans
class KMeans:
    def __init__(self, k=3, max_iter=100, random_state=89):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state

    def hitung_jarak(self, X):
        return np.sqrt(((X - self.centroid[:, np.newaxis]) ** 2).sum(axis=2))

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        np.random.seed(self.random_state)
        n_sampel = len(X)
        index = np.random.choice(n_sampel, self.k, replace=False)
        self.centroid = X[index]
        for _ in range(self.max_iter):
            self.labels = np.argmin(self.hitung_jarak(X), axis=0)
            centroid_baru = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroid, centroid_baru):
                break
            self.centroid = centroid_baru
        self.wcss = sum(np.sum((X[self.labels == i] - self.centroid[i]) ** 2) for i in range(self.k))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.argmin(self.hitung_jarak(X), axis=0)

# Silhouette evaluation
ks_eval = range(2, 6)
silhouette_scores = []

for k in ks_eval:
    model_k = KMeans(k=k, random_state=89)
    model_k.fit(X_s)
    score = silhouette_score(X_s, model_k.labels)
    silhouette_scores.append(score)
    print(f'K = {k} -> Silhouette Score: {score:.4f}')

best_k = list(ks_eval)[np.argmax(silhouette_scores)]
print(f'\nJumlah cluster terbaik berdasarkan Silhouette Score: K = {best_k}')

# Visualization
plt.figure(figsize=(8, 5))
bars = plt.bar(list(ks_eval), silhouette_scores, color='#87CEEB', edgecolor='black')

avg_score = np.mean(silhouette_scores)
plt.axhline(y=avg_score, color='red', linestyle='--', linewidth=2, label=f'Rata-rata: {avg_score:.3f}')

for bar, score in zip(bars, silhouette_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{score:.3f}', ha='center', va='bottom', fontsize=11)

plt.title('Evaluasi Silhouette Score', fontsize=18)
plt.xlabel('Jumlah Cluster (K)', fontsize=14)
plt.ylabel('Silhouette Coefficient', fontsize=14)
plt.xticks(list(ks_eval))
plt.ylim(0, max(silhouette_scores) + 0.05)
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig(r'c:\laragon\www\LKS\AI - 2026\BOOTCAMP - WEEK 2\silhouette_test.png', dpi=100)
print('\nVisualization saved to silhouette_test.png')
