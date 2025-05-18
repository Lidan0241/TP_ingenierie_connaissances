import streamlit as st
import os
import re
import tempfile
import time
import pandas as pd
import spacy
import pandas as pd
from nltk.util import ngrams
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os

from collections import Counter
from concept_detection import (
    load_corpus_from_folder, clean_text, get_stopword_set, extract_ngrams,
    get_tfidf_keywords, get_pmi_keywords, get_cvalue_keywords,
    get_rake_keywords, filter_extracted_ngrams
)

nlp = spacy.load("fr_core_news_sm")

@st.cache_data
def load_texts():
    return load_corpus_from_folder("./lotr_corpus")

def highlight_concepts(text, concepts):
    """Ajoute du HTML <mark> pour surligner les concepts dans le texte brut."""
    for c in sorted(concepts, key=len, reverse=True):
        escaped = re.escape(c)
        text = re.sub(rf"(?i)\b({escaped})\b", r'<mark>\1</mark>', text)
    return text


# --- Configuration ---
st.set_page_config(page_title="Concept Explorer", layout="wide")
st.title("Détection et annotation de concepts – LOTR")
st.markdown(
    "Cette application permet d'extraire et d'annoter des concepts à partir de textes du corpus LOTR."
)

# --- Chargement du corpus ---
texts = load_texts()

# --- Sélections alignées ---
col1, col2, col3 = st.columns(3)

with col1:
    doc_index = st.selectbox("📄 Choisir un document :", range(len(texts)))

with col2:
    method = st.radio("🔍 Méthode de détection :", ["TF-IDF", "PMI", "C-value", "RAKE"])

with col3:
    ngram_size = st.radio(
        "📏 Taille des n-grammes :",
        options=[1, 2, 3],
        index=1,
        format_func=lambda x: f"{x}-gramme{'s' if x > 1 else ''}"
    )

text = texts[doc_index]

if "selected_concepts" not in st.session_state:
    st.session_state.selected_concepts = []

selected_concepts = st.session_state.selected_concepts

# --- Bouton de lancement ---
if st.button("Rechercher"):
    with st.spinner("Extraction en cours..."):
        # Extraction selon méthode
        if method == "TF-IDF":
            candidates = get_tfidf_keywords(
                [text],
                ngram_range=(ngram_size, ngram_size),
                min_df=1,
                max_df=1.0
            )
            raw_ngrams = [term for term, _ in candidates]

        elif method == "PMI":
            raw_ngrams = [ng for ng, _ in get_pmi_keywords([text], n=ngram_size)]

        elif method == "C-value":
            raw_ngrams = [ng for ng, _ in get_cvalue_keywords([text], n=ngram_size)]

        elif method == "RAKE":
            raw_ngrams = [ng for ng, _ in get_rake_keywords([text], language="french")]

        filtered = filter_extracted_ngrams(raw_ngrams, language_model="fr_core_news_sm")
        st.session_state["concepts_detected"] = list(dict.fromkeys(filtered))


        # Filtrage syntaxique
        filtered = filter_extracted_ngrams(raw_ngrams, language_model="fr_core_news_sm")
        filtered = list(dict.fromkeys(filtered))  # dédoublonnage

if "concepts_detected" in st.session_state:
    st.markdown("## Concepts détectés (cliquez pour supprimer)")
    filtered = st.session_state["concepts_detected"]
    to_keep = []
    cols = st.columns(4)
    for i, concept in enumerate(filtered):
        col = cols[i % 4]
        if col.checkbox(concept, value=True, key=f"cpt_{i}"):
            to_keep.append(concept)

    st.markdown("## Texte avec concepts surlignés")
    with st.expander("Opérations avancées sur les concepts et relations"):

        if st.button("Trier les concepts par fréquence"):
            concept_counts = Counter(selected_concepts)
            sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
            st.write("Concepts triés par fréquence :")
            for c, freq in sorted_concepts:
                st.write(f"- {c} ({freq})")

        if st.button("Grouper concepts similaires (clustering sémantique)"):
            if not selected_concepts:
                st.warning("⚠️ Aucun concept sélectionné pour le clustering.")
            else:
                from sentence_transformers import SentenceTransformer, util
                from scipy.cluster.hierarchy import linkage, fcluster
                from collections import defaultdict

                start = time.time()
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode(list(selected_concepts))
                Z = linkage(embeddings, method="average", metric="cosine")
                clusters = fcluster(Z, t=0.4, criterion="distance")
                duration = time.time() - start

                clustered = defaultdict(list)
                for concept, cid in zip(selected_concepts, clusters):
                    clustered[cid].append(concept)

                st.write("Groupes de concepts similaires :")
                st.success(f"✅ Clustering terminé en {duration:.2f} sec.")
                for cid, group in clustered.items():
                    if len(group) > 1:
                        st.markdown(f"- Cluster {cid} : {', '.join(group)}")

        if st.button("Exporter concepts sélectionnés"):
            df_export = pd.DataFrame({"concept": selected_concepts})
            df_export.to_csv("concepts_selectionnes.csv", index=False)
            st.success("Exporté sous concepts_selectionnes.csv")
        

        if st.button("Exporter l'ontologie au format OWL"):
            df_relations = pd.read_csv("relations_extraites.csv")  # <-- ajoute ça

            g = Graph()
            ns = Namespace("http://lotr-ontology.org#")

            for _, row in df_relations.iterrows():
                s = URIRef(ns[row["entity1"].replace(" ", "_")])
                p = URIRef(ns[row["label"]])
                o = URIRef(ns[row["entity2"].replace(" ", "_")])
                g.add((s, p, o))

            g.serialize(destination="ontologie.owl", format="xml")
            st.success("Ontologie exportée dans ontologie.owl")


        if st.button("Exporter en version RTF (lisible humain)"):
            rtf_lines = []
            for _, row in df_relations.iterrows():
                rtf_lines.append(f"{row['sujet']} —[{row['relation']}]→ {row['objet']}")

            with open("ontologie.rtf", "w", encoding="utf-8") as f:
                f.write("{\\rtf1\\ansi\n")
                for line in rtf_lines:
                    f.write(f"{line}\\line\n")
                f.write("}")

            st.success("Ontologie exportée dans ontologie.rtf")
        
        if st.button("Visualiser l'ontologie (graphe interactif)"):
            df_relations = pd.read_csv("relations_clusterisees.csv")

            # Construire le graphe
            net = Network(height="700px", width="100%", directed=True, notebook=False)
            net.force_atlas_2based()

            for _, row in df_relations.iterrows():
                e1 = row["entity1"]
                e2 = row["entity2"]
                label = row["label"]
                net.add_node(e1, label=e1, title=e1)
                net.add_node(e2, label=e2, title=e2)
                net.add_edge(e1, e2, label=label)

            # Sauvegarde dans fichier HTML temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                net.save_graph(tmp_file.name)
                tmp_path = tmp_file.name

            # Affichage dans Streamlit
            st.success("Graphe généré.")
            components.iframe(tmp_path, height=750, scrolling=True)



    st.markdown(
        f"<div style='background-color:#f8f8f8;padding:1em;border-radius:8px;'>{highlight_concepts(text, to_keep)}</div>",
        unsafe_allow_html=True
    )

    st.download_button("Exporter les concepts gardés", "\n".join(to_keep), file_name="concepts_valides.txt")

