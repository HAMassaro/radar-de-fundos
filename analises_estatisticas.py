"""Analises complementares sobre o agrupamento de producao (le reports/fundos_clusterizados.csv).
Gera reports_real/analises_estatisticas.txt: medianas por grupo, delta de Cliff, correlacoes de
Spearman, comparacao com tercis de volatilidade, cargas da PCA e estabilidade do K-Means."""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
import pipeline_real as p

PERFIS = ["Conservador", "Moderado", "Arrojado"]
d = pd.read_csv(p.OUT / "fundos_clusterizados.csv")
X = p.normalize(d)
cad = pd.read_csv(p.OUT / "cadastro_classificado.csv")[["CD_CVM", "CLASSE_ANBIMA"]]
L = []
L.append(f"fundos agrupados: {len(d)}")
L.append("tamanhos: " + ", ".join(f"{q} {int((d.perfil == q).sum())}" for q in PERFIS))
L.append("\nmedianas por grupo:\n" + d.groupby("perfil")[p.FEATURE_COLS].median().loc[PERFIS].round(4).to_string())
L.append("\ndelta de Cliff (Mann-Whitney):")
for f in ["assimetria", "consistencia"]:
    for a, b in itertools.combinations(PERFIS, 2):
        x, y = d.loc[d.perfil == a, f], d.loc[d.perfil == b, f]
        u, _ = mannwhitneyu(x, y)
        L.append(f"  {f} {a} x {b}: {2 * u / (len(x) * len(y)) - 1:.2f}")
L.append("\ncorrelacao de Spearman:\n" + d[p.FEATURE_COLS].corr("spearman").round(2).to_string())
t = pd.qcut(d.volatilidade_anual, 3, labels=["T1", "T2", "T3"])
ct = pd.crosstab(d.perfil, t).loc[PERFIS]
dif = len(d) - ct.loc["Conservador", "T1"] - ct.loc["Moderado", "T2"] - ct.loc["Arrojado", "T3"]
L.append("\nK-Means x tercis de volatilidade:\n" + ct.to_string())
L.append(f"ARI K-Means x tercis: {ari(d.perfil, t):.3f} | fundos em grupo diferente: {dif} ({dif / len(d):.1%})")
m = (d.perfil == "Moderado") & (t == "T1"); c = (d.perfil == "Conservador") & (t == "T1")
L.append(f"Moderado no tercil de menor volatilidade: {int(m.sum())}")
L.append(pd.DataFrame({"moderado_T1": d[m][p.FEATURE_COLS].median(), "conservador_T1": d[c][p.FEATURE_COLS].median()}).round(4).to_string())
L.append(d[m].merge(cad, on="CD_CVM").CLASSE_ANBIMA.value_counts().head(4).to_string())
pca = PCA(2, random_state=42).fit(X)
L.append(f"\nPCA: variancia {pca.explained_variance_ratio_.round(4).tolist()} total {pca.explained_variance_ratio_.sum():.4f}")
L.append("cargas:\n" + pd.DataFrame(pca.components_, columns=p.FEATURE_COLS, index=["PC1", "PC2"]).round(2).to_string())
s = [ari(d.cluster_raw, KMeans(3, init="k-means++", n_init=10, random_state=r).fit_predict(X)) for r in range(20)]
rng = np.random.default_rng(0); b = []
for _ in range(30):
    i = rng.choice(len(X), int(.8 * len(X)), replace=False)
    b.append(ari(d.cluster_raw, KMeans(3, init="k-means++", n_init=10, random_state=42).fit(X[i]).predict(X)))
L.append(f"\nestabilidade: 20 sementes ARI {min(s):.3f} a {max(s):.3f} | 30 subamostras de 80% mediana {np.median(b):.3f} minimo {min(b):.3f}")
out = "\n".join(L); (p.OUT / "analises_estatisticas.txt").write_text(out); print(out)
