from lotr_dataset_small import nodes, relationships
import json

output = {
    "nodes": nodes,
    "relationships": relationships
}

with open("lotr_dataset_small.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Fichier JSON exporté : lotr_dataset_small.json")
