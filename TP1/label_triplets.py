import pandas as pd
import ollama
import time

# Liste de méta-relations
META_RELATIONS = [
    "est_parent_de", "est_enfant_de", "combat_contre", "est_allie_de", "porte_objet",
    "appartient_a_race", "a_titre", "regne_sur", "fait_partie_de", "est_ami_de",
    "traque", "est_ennemi_de", "se_marie_avec", "sacrifie_pour", "est_mentor_de"
]

def build_prompt(sentence: str, e1: str, e2: str) -> str:
    rel_list = '\n'.join(f"{i+1}. {rel}" for i, rel in enumerate(META_RELATIONS))
    prompt = f"""
Tu es un expert du Seigneur des Anneaux et tu aides à construire une ontologie.

Voici une phrase :
"{sentence}"

Deux entités y apparaissent : "{e1}" et "{e2}".

Classe la relation exprimée entre elles parmi cette liste :

{rel_list}

Réponds uniquement par un seul du numéro de labels ci-dessus (par exemple 7), sans phrase ni explication, ne choisit qu'un seul label!
"""
    return prompt.strip()

def clean_llm_label(response: str) -> str:
    response = response.strip().lower()

    # Cas 1 : réponse est un index pur ("7")
    if response.isdigit():
        idx = int(response) - 1
        if 0 <= idx < len(META_RELATIONS):
            return META_RELATIONS[idx]

    # Cas 2 : "7. label" → on récupère "label"
    if "." in response:
        parts = response.split(".")
        # Si c’est un numéro + un label valide
        if parts[0].strip().isdigit():
            label_candidate = parts[1].strip()
            if label_candidate in META_RELATIONS:
                return label_candidate
            else:
                idx = int(parts[0].strip()) - 1
                if 0 <= idx < len(META_RELATIONS):
                    return META_RELATIONS[idx]

    # Cas 3 : label direct
    if response in META_RELATIONS:
        return response

    return "NA"

def label_relation_with_llm(ent1: str, ent2: str, sentence: str, model="mistral") -> str:
    prompt = build_prompt(sentence, ent1, ent2)
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"]
    return clean_llm_label(raw)

def annotate_triplets(triplet_csv_path: str, output_path: str):
    df = pd.read_csv(triplet_csv_path)
    labels = []

    for i, row in df.iterrows():
        e1, e2, sentence = row["entity1"], row["entity2"], row["sentence"]
        try:
            label = label_relation_with_llm(e1, e2, sentence)
        except Exception as e:
            print(f"Erreur à la ligne {i}: {e}")
            label = "NA"

        print(f"[{i+1}/{len(df)}] {e1} – {e2} → {label}")
        labels.append(label)

    df["label"] = labels
    df = df[df["label"] != "NA"]
    df.to_csv(output_path, index=False)
    print(f"\nFichier final sauvegardé dans : {output_path}")

if __name__ == "__main__":
    annotate_triplets("relations_triplets.csv", "relation_dataset.csv")
