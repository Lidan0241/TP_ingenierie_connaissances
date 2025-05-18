from lotr_dataset_large import nodes, relationships
from orql_core import OneRingDB, ORQueryExecutor

# 1. Initialisation de la base
db = OneRingDB()
executor = ORQueryExecutor(db)

# 2. Chargement des nœuds
for name, cls in nodes.items():
    orql = f'CREATE (:{cls} {{name: "{name}"}})'
    executor.execute(orql)

# 3. Chargement des relations
for src, tgt, rel, props in relationships:
    props_str = (
        "{" + ", ".join(f'{k}: {repr(v)}' for k, v in props.items()) + "}"
        if props else ""
    )
    orql = f'CREATE [:{rel} {props_str}] FROM "{src}" TO "{tgt}"'
    executor.execute(orql)

# 4. Résumé
print(f"Chargement terminé.")
print(f"Nœuds : {len(db.nodes)}")
print(f"Relations : {len(db.edges)}")
