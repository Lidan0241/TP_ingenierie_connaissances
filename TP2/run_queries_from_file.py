def run_orql_file(filepath, executor):
    print(f"📂 Lecture des requêtes depuis : {filepath}\n")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            query = line.strip()
            if not query or query.startswith("#"):
                continue  # ignorer commentaires et lignes vides
            print(f"▶️  L{line_num} : {query}")
            try:
                result = executor.execute(query)
                if isinstance(result, list):
                    for r in result:
                        print("  ✅", r)
                else:
                    print("  ✅", result)
            except Exception as e:
                print("  ❌ ERREUR :", e)

# Exemple d'appel :
# run_orql_file("requetes_CRUD.txt", executor)
