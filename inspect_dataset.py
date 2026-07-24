import pandas as pd

df = pd.read_csv("data/raw/fedmml_ed_triage_dataset.csv")

print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUNAS ===")
print(list(df.columns))

print("\n=== DISTRIBUICAO ESI ===")
print(df["esi_level"].value_counts().sort_index())
print(df["esi_level"].value_counts(normalize=True).sort_index().round(3))

print("\n=== COMPRIMENTO DAS NOTAS ===")
print(df["clinical_notes"].str.len().describe())

print("\n=== NULOS ===")
print(df[["clinical_notes", "esi_level"]].isna().sum())

print("\n=== AMOSTRAS POR CLASSE ===")
for lvl in sorted(df["esi_level"].dropna().unique()):
    nota = df[df["esi_level"] == lvl]["clinical_notes"].dropna().iloc[0]
    print(f"\n--- ESI {lvl} ---")
    print(nota[:300])

print("\n=== NOTAS UNICAS ===")
print(f"total: {len(df)} | unicas: {df['clinical_notes'].nunique()}")
