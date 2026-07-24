import pandas as pd

tr = pd.read_csv("data/raw_abstracts/train.csv")
te = pd.read_csv("data/raw_abstracts/test.csv")

nomes = {1: "neoplasms", 2: "digestive", 3: "nervous",
         4: "cardiovascular", 5: "general pathological"}

print("=== SHAPE ===")
print("train:", tr.shape, "| test:", te.shape)

print("\n=== UNICIDADE (train) ===")
print(f"linhas: {len(tr)} | abstracts unicos: {tr['medical_abstract'].nunique()}")

print("\n=== DISTRIBUICAO (train) ===")
vc = tr["condition_label"].value_counts().sort_index()
for k, v in vc.items():
    print(f"{k} ({nomes.get(k,'?')}): {v} ({v/len(tr):.1%})")

print("\n=== COMPRIMENTO (chars) ===")
print(tr["medical_abstract"].str.len().describe().round(1))

print("\n=== COMPRIMENTO (palavras) ===")
print(tr["medical_abstract"].str.split().str.len().describe().round(1))

print("\n=== NULOS ===")
print(tr.isna().sum())

print("\n=== AMOSTRA POR CLASSE (300 chars) ===")
for lvl in sorted(tr["condition_label"].unique()):
    txt = tr[tr["condition_label"] == lvl]["medical_abstract"].iloc[0]
    print(f"\n--- {lvl} = {nomes.get(lvl,'?')} ---")
    print(txt[:300])

print("\n=== MARCADORES DE URGENCIA (para avaliar weak supervision) ===")
marcadores = ["acute", "emergency", "urgent", "severe", "critical", "mortality",
              "ICU", "intensive care", "fatal", "chronic", "elective", "routine",
              "follow-up", "mild", "asymptomatic"]
low = tr["medical_abstract"].str.lower()
for m in marcadores:
    n = low.str.contains(m.lower(), regex=False).sum()
    print(f"{m:16s}: {n:6d} ({n/len(tr):.1%})")
