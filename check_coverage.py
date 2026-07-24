import pandas as pd

tr = pd.read_csv("data/raw_abstracts/train.csv").drop_duplicates("medical_abstract")
low = tr["medical_abstract"].str.lower()

alto = ["acute", "emergency", "mortality", "fatal", "intensive care", "life-threatening"]
baixo = ["chronic", "elective", "routine", "asymptomatic", "mild", "follow-up"]

tem_alto = low.apply(lambda t: any(m in t for m in alto))
tem_baixo = low.apply(lambda t: any(m in t for m in baixo))

print(f"base unica: {len(tr)}")
print(f"so marcador ALTO:   {(tem_alto & ~tem_baixo).sum()}")
print(f"so marcador BAIXO:  {(~tem_alto & tem_baixo).sum()}")
print(f"AMBOS (conflito):   {(tem_alto & tem_baixo).sum()}")
print(f"NENHUM (descarte):  {(~tem_alto & ~tem_baixo).sum()}")
rot = (tem_alto ^ tem_baixo).sum()
print(f"\nrotulaveis sem conflito: {rot} ({rot/len(tr):.1%})")
