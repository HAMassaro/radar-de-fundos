"""Sensibilidade a janela: recalcula as features com 504 pregoes (cerca de dois anos), mesmos
fundos elegiveis e mesmo corte de producao, e compara os grupos com os da janela de 756."""
import sqlite3
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
import pipeline_real as p

con = sqlite3.connect(p.DB_PATH)
cad = p.classify_eligibility(con)
elig = cad[cad.CLASSIFICACAO == "ELEGIVEL"].CD_CVM.tolist()
p.WINDOW = 504
f = p.build_feature_table(con, p.DATA_CORTE_PROD, cd_cvms=elig)
f, k, _, X = p.run_kmeans(f)
f["CD_CVM"] = f.CD_CVM.astype(int)
base = pd.read_csv(p.OUT / "fundos_clusterizados.csv")[["CD_CVM", "perfil"]].rename(columns={"perfil": "p756"})
m = f.merge(base, on="CD_CVM")
ct = pd.crosstab(m.p756, m.perfil)
igual = sum(ct.loc[q, q] for q in ct.index if q in ct.columns)
L = [f"fundos com 504 pregoes: {len(f)} | em comum com a janela de 756: {len(m)}",
     "tamanhos 504: " + f.perfil.value_counts().to_string().replace("\n", "; "),
     f"Silhouette 504: {silhouette_score(X, f.cluster):.3f} | ARI 504 x 756: {ari(m.p756, m.perfil):.3f} | mesmo perfil: {igual} ({igual / len(m):.1%})",
     ct.to_string()]
out = "\n".join(L); (p.OUT / "analise_janela_504.txt").write_text(out); print(out)
