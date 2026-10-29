"""Compara o K-Means com um corte por volatilidade nos mesmos tamanhos de grupo (1.721/2.677/2.289)."""
import pandas as pd
from pathlib import Path
f = pd.read_csv(Path(__file__).parent / "reports_real" / "fundos_clusterizados.csv")
n = f.perfil.value_counts()
f = f.sort_values("volatilidade_anual").reset_index(drop=True)
c, m = n["Conservador"], n["Moderado"]
f["corte_vol"] = ["Conservador"] * c + ["Moderado"] * m + ["Arrojado"] * (len(f) - c - m)
d = f[f.corte_vol != f.perfil]
print(f"mesmo perfil: {1 - len(d) / len(f):.1%} | trocas: {len(d)}")
print(pd.crosstab(d.corte_vol, d.perfil))
