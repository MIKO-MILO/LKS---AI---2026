import json

nb_path = r"c:\laragon\www\LKS\AI - 2026\BOOTCAMP - WEEK 2\KMeans(1).ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove the previously added silhouette cell (if exists) to avoid duplicates
nb["cells"] = [c for c in nb["cells"] if c.get("id") != "silhouette_eval_cell"]

# 2) Build the silhouette evaluation code cell
silhouette_source = [
    "# Evaluasi Silhouette Score untuk menentukan jumlah cluster (K) terbaik\n",
    "from sklearn.metrics import silhouette_score\n",
    "\n",
    "# Pastikan data bersih (hapus baris dengan nilai NaN akibat mapping sex)\n",
    "df_clean = df.dropna().copy()\n",
    "X_clean = df_clean[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm',\n",
    "                    'body_mass_g', 'sex']].values.astype(float)\n",
    "\n",
    "min_v = X_clean.min(axis=0)\n",
    "max_v = X_clean.max(axis=0)\n",
    "rentang_v = max_v - min_v\n",
    "rentang_v = np.where(rentang_v == 0, 1.0, rentang_v)\n",
    "X_s_clean = (X_clean - min_v) / rentang_v\n",
    "\n",
    "ks_eval = range(2, 6)  # K = 2, 3, 4, 5\n",
    "silhouette_scores = []\n",
    "\n",
    "for k in ks_eval:\n",
    "    model_k = KMeans(k=k, random_state=89)\n",
    "    model_k.fit(X_s_clean)\n",
    "    score = silhouette_score(X_s_clean, model_k.labels)\n",
    "    silhouette_scores.append(score)\n",
    "    print(f'K = {k} -> Silhouette Score: {score:.4f}')\n",
    "\n",
    "best_k = list(ks_eval)[np.argmax(silhouette_scores)]\n",
    "print(f'\\nJumlah cluster terbaik berdasarkan Silhouette Score: K = {best_k}')\n",
    "\n",
    "# Visualisasi Silhouette Score\n",
    "plt.figure(figsize=(8, 5))\n",
    "bars = plt.bar(list(ks_eval), silhouette_scores, color='#87CEEB', edgecolor='black')\n",
    "\n",
    "# Garis putus-putus merah (rata-rata silhouette score)\n",
    "avg_score = np.mean(silhouette_scores)\n",
    "plt.axhline(y=avg_score, color='red', linestyle='--', linewidth=2,\n",
    "            label=f'Rata-rata: {avg_score:.3f}')\n",
    "\n",
    "# Anotasi nilai di atas setiap bar\n",
    "for bar, score in zip(bars, silhouette_scores):\n",
    "    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,\n",
    "             f'{score:.3f}', ha='center', va='bottom', fontsize=11)\n",
    "\n",
    "plt.title('Evaluasi Silhouette Score', fontsize=18)\n",
    "plt.xlabel('Jumlah Cluster (K)', fontsize=14)\n",
    "plt.ylabel('Silhouette Coefficient', fontsize=14)\n",
    "plt.xticks(list(ks_eval))\n",
    "plt.ylim(0, max(silhouette_scores) + 0.05)\n",
    "plt.grid(True, axis='y')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "silhouette_eval_cell",
    "metadata": {},
    "outputs": [],
    "source": silhouette_source
}

nb["cells"].append(new_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Silhouette evaluation cell updated successfully.")
