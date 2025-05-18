import os
import spacy
import csv
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from typing import List, Tuple
from concept_detection import load_corpus_from_folder, extract_ngrams, filter_extracted_ngrams

nlp = spacy.load("fr_core_news_sm")

def get_subjects(sentence: str) -> list[str]:
    doc = nlp(sentence)
    return [tok.text.lower() for tok in doc if tok.dep_ in {"nsubj", "nsubj:pass"}]

def find_concepts_in_sentence(sentence: str, concepts: List[str]) -> List[str]:
    sent_lc = sentence.lower()
    return [c for c in concepts if c.lower() in sent_lc]

def extract_relations_from_sentence(sentence: str, concepts: List[str]) -> List[Tuple[str, str]]:
    present = find_concepts_in_sentence(sentence, concepts)
    if len(present) < 2:
        return []
    subjects = get_subjects(sentence)
    relations = []
    for subj in present:
        if subj in subjects:
            for target in present:
                if target != subj:
                    relations.append((subj, target))
    return relations

def extract_all_relations(corpus: List[str], concepts: List[str]) -> Tuple[List[Tuple[str, str, str]], dict]:
    all_relations = []
    weighted_graph = defaultdict(int)
    for text in corpus:
        sentences = [sent.text for sent in nlp(text).sents]
        for sent in sentences:
            relations = extract_relations_from_sentence(sent, concepts)
            for subj, obj in relations:
                all_relations.append((subj, obj, sent))
                weighted_graph[(subj, obj)] += 1
    return all_relations, dict(weighted_graph)

def plot_relation_graph(weighted_graph: dict, min_weight: int = 2):
    G = nx.DiGraph()
    for (src, tgt), w in weighted_graph.items():
        if w >= min_weight:
            G.add_edge(src, tgt, weight=w)

    pos = nx.spring_layout(G, k=0.7)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=1200, font_size=10,
            width=[0.5 + w for w in weights], edge_color=weights, edge_cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})
    plt.title("Relations entre concepts (graphe pondéré)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Chargement du corpus...")
    texts = load_corpus_from_folder("./lotr_corpus")

    print("Extraction des unigrammes...")
    raw_ngrams = extract_ngrams(texts, n=1, remove_stopwords=False, lowercase=True, language="french")
    concepts = list(set(filter_extracted_ngrams(raw_ngrams)))
    print(f"{len(concepts)} concepts extraits.")

    print("Extraction des relations sujet → autre concept...")
    all_rels, graph = extract_all_relations(texts, concepts)
    print(f"{len(all_rels)} relations extraites à partir des phrases.")

    print("\nRelations les plus fréquentes :")
    for (src, tgt), count in sorted(graph.items(), key=lambda x: -x[1])[:10]:
        print(f"  {src} → {tgt} : {count}")

    print("\nSauvegarde CSV : relations_triplets.csv")
    df = pd.DataFrame(all_rels, columns=["entity1", "entity2", "sentence"])
    df.to_csv("relations_triplets.csv", index=False)

    print("\nAffichage du graphe (relations ≥ 2 occurrences)...")
    plot_relation_graph(graph, min_weight=2)
