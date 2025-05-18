import streamlit as st
import json
from pyvis.network import Network
import tempfile
import os

from orql_core import OneRingDB, ORQueryExecutor

# --- Initialisation ---
if "db" not in st.session_state:
    st.session_state.db = OneRingDB()
    st.session_state.executor = ORQueryExecutor(st.session_state.db)
    st.session_state.last_path = None  # pour LINK

db = st.session_state.db
executor = st.session_state.executor

st.set_page_config(page_title="OneRingDB", layout="wide")

# --- Sidebar ---
st.sidebar.title("OneRing Query")

# 1. Entrée ORQL
user_query = st.sidebar.text_area("✍️ Requête ORQL :", height=150)

if st.sidebar.button("▶ Exécuter requête"):
    try:
        result = executor.execute(user_query)
        st.sidebar.success("Requête exécutée!")

        # mémorise le chemin s’il y en a un
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            st.session_state.last_path = result
        else:
            st.session_state.last_path = None

        if isinstance(result, (dict, list)):
            st.sidebar.json(result)
        else:
            st.sidebar.code(str(result))
    except Exception as e:
        st.sidebar.error(f"Erreur : {e}")

# 2. Chargement JSON
st.sidebar.markdown("---")
json_file = st.sidebar.file_uploader("📂 Charger JSON de base", type="json")

if json_file is not None:
    try:
        data = json.load(json_file)
        for name, cls in data["nodes"].items():
            executor.execute(f'CREATE (:{cls} {{name: "{name}"}})')
        for src, tgt, rel, props in data["relationships"]:
            props_str = (
                "{" + ", ".join(f'{k}: {repr(v)}' for k, v in props.items()) + "}"
                if props else ""
            )
            executor.execute(f'CREATE [:{rel} {props_str}] FROM "{src}" TO "{tgt}"')
        st.sidebar.success("Base chargée depuis JSON.")
    except Exception as e:
        st.sidebar.error(f"Erreur dans le fichier JSON : {e}")

# --- Zone principale ---
st.title("Base OneRing - Visualisation")

# 1. Infos
st.subheader("Résumé")
st.markdown(f"- Nœuds : {len(db.nodes)}")
st.markdown(f"- Relations : {len(db.edges)}")

# 2. Graphe interactif
st.subheader("🕸️ Graphe interactif")

net = Network(height="600px", width="100%", notebook=False)
color_map = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
]

if st.sidebar.button("🔄 Réinitialiser la base"):
    st.session_state.db = OneRingDB()
    st.session_state.executor = ORQueryExecutor(st.session_state.db)
    st.session_state.last_path = None
    st.sidebar.warning("La base a été réinitialisée.")

# Construire le graphe avec couleurs + surbrillance du chemin LINK
highlight_path = set()
if st.session_state.last_path:
    try:
        path_nodes = [n for n in st.session_state.last_path]
        path_ids = [node.id for name in path_nodes for node in db.find_nodes_by_name(name)]
        highlight_path = set(zip(path_ids, path_ids[1:]))
    except:
        pass

for node in db.nodes.values():
    label = node.name
    title = "<br>".join(f"{k}: {v}" for k, v in node.properties.items())
    color = color_map[node.color % len(color_map)] if getattr(node, "color", None) is not None else None
    net.add_node(node.id, label=label, title=title, group=node.node_class, color=color)

for edge in db.edges.values():
    highlight = (edge.source_id, edge.target_id) in highlight_path or (edge.target_id, edge.source_id) in highlight_path
    net.add_edge(
        edge.source_id,
        edge.target_id,
        title=", ".join(f"{k}: {v}" for k, v in edge.properties.items()),
        color="red" if highlight else None,
        width=3 if highlight else 1
    )

# Rendu HTML dans Streamlit
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
    net.save_graph(tmp.name)
    html_path = tmp.name

with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()
    st.components.v1.html(html, height=650, scrolling=True)

os.remove(html_path)
