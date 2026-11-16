"""ARI entre os perfis do K-Means e a classificação ANBIMA, nas classes completas e no primeiro nível
(primeira palavra da classe: Renda Fixa, Ações, Multimercados, Previdência, Cambial etc.).
Fundos sem classe ANBIMA no cadastro ficam de fora."""
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score as ari
OUT = Path(__file__).resolve().parent / "reports_real"
f = pd.read_csv(OUT / "fundos_clusterizados.csv")
c = pd.read_csv(OUT / "cadastro_classificado.csv", usecols=["CD_CVM", "CLASSE_ANBIMA"])
d = f.merge(c, on="CD_CVM", how="left").dropna(subset=["CLASSE_ANBIMA"])
d["nivel1"] = d.CLASSE_ANBIMA.str.split().str[0]
print(f"fundos com classe: {len(d)} (sem classe: {len(f) - len(d)})")
print(f"ARI nas {d.CLASSE_ANBIMA.nunique()} classes: {ari(d.perfil, d.CLASSE_ANBIMA):.2f}")
print(f"ARI no primeiro nível: {ari(d.perfil, d.nivel1):.2f}")
print(pd.crosstab(d.nivel1, d.perfil))
