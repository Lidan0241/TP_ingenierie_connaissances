import time
import random
import json
from lotr_dataset_large import nodes as full_nodes, generate_semantic_relationships
from orql_core import OneRingDB, ORQueryExecutor

def build_json_subset(nb_nodes, nb_edges):
    selected_nodes = dict(random.sample(sorted(full_nodes.items()), nb_nodes))
    rels = generate_semantic_relationships(selected_nodes, nb_edges)
    return {
        "nodes": selected_nodes,
        "relationships": rels
    }

def measure_create(db, executor, dataset):
    t0 = time.time()
    for name, cls in dataset["nodes"].items():
        # Format manuel ORQL {name: "xxx"}
        safe_name = name.replace('"', '\\"')
        props = f'{{name: "{safe_name}"}}'
        executor.execute(f'CREATE (:{cls} {props})')

    for src, tgt, rel, props in dataset["relationships"]:
        prop_str = (
            "{" + ", ".join(f'{k}: {json.dumps(v)}' for k, v in props.items()) + "}"
            if props else ""
        )
        executor.execute(f'CREATE [:{rel} {prop_str}] FROM "{src}" TO "{tgt}"')

    return (time.time() - t0) * 1000  # ms

def measure_read(executor, names):
    t0 = time.time()
    for name in names:
        executor.execute(f'READ (:Any) WHERE name = "{name}"')
    return (time.time() - t0) * 1000 / len(names)

def measure_link(executor, pairs):
    t0 = time.time()
    for src, tgt in pairs:
        try:
            executor.execute(f'LINK FROM "{src}" TO "{tgt}"')
        except:
            pass
    return (time.time() - t0) * 1000 / len(pairs)

def measure_cluster(executor):
    t0 = time.time()
    executor.execute("CLUSTER")
    return (time.time() - t0) * 1000

# --- MAIN BENCH ---
sizes = [(10, 20), (100, 200), (123, 242)]
results = []

for nb_nodes, nb_rels in sizes:
    db = OneRingDB()
    executor = ORQueryExecutor(db)

    print(f"\n=== TEST : {nb_nodes} nœuds, {nb_rels} relations ===")
    data = build_json_subset(nb_nodes, nb_rels)

    # CREATE
    create_time = measure_create(db, executor, data)

    # READ
    names = list(data["nodes"].keys())
    read_time = measure_read(executor, names)

    # LINK
    pairs = [random.sample(names, 2) for _ in range(min(len(names) // 2, 50))]
    link_time = measure_link(executor, pairs)

    # CLUSTER
    cluster_time = measure_cluster(executor)

    results.append({
        "N": nb_nodes,
        "create (ms)": round(create_time, 2),
        "read avg (ms)": round(read_time, 2),
        "link avg (ms)": round(link_time, 2),
        "cluster (ms)": round(cluster_time, 2)
    })

# --- Display table ---
print("\nRésumé des performances")
print(f"{'N':>6} | {'CREATE':>10} | {'READ':>10} | {'LINK':>10} | {'CLUSTER':>10}")
print("-" * 58)
for row in results:
    print(f"{row['N']:6} | {row['create (ms)']:10} | {row['read avg (ms)']:10} | {row['link avg (ms)']:10} | {row['cluster (ms)']:10}")
