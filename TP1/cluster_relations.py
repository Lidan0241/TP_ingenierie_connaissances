import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter
from sentence_transformers import SentenceTransformer

# 1. Charger les phrases annotées
df = pd.read_csv("./labeled_triplets.csv")

# 2. Encoder les phrases
model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(df["sentence"].tolist(), show_progress_bar=True)

# 3. Clustering
n_clusters = len(df["label"].unique())
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X)

# 4. Étiquetage par contagion (label majoritaire du cluster)
cluster2label = {}
for cluster_id in sorted(df["cluster"].unique()):
    labels = df[df["cluster"] == cluster_id]["label"]
    if not labels.empty:
        most_common = labels.mode().iloc[0]
        cluster2label[cluster_id] = most_common

# 5. Étiquette propagée
df["contagion_label"] = df["cluster"].map(cluster2label)

# 6. Évaluation
accuracy = accuracy_score(df["label"], df["contagion_label"])
print(f"Précision par contagion de cluster : {accuracy:.3f} ({accuracy*100:.1f}%)")

# Optionnel : sauvegarde
df.to_csv("relations_clusterisees.csv", index=False)
print(" Résultats enregistrés dans : relations_clusterisees.csv")
