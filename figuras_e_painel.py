"""Gera as figuras da monografia (escolha de k, distribuicao das features, PCA) e o arquivo
painel_dados.json usado pela interface web."""
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pipeline_real as p

FIG = p.BASE / "figuras"; FIG.mkdir(exist_ok=True)
PERFIS, CORES = ["Conservador", "Moderado", "Arrojado"], ["#2a6f97", "#e09f3e", "#9e2a2b"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
d = pd.read_csv(p.OUT / "fundos_clusterizados.csv")
X = p.normalize(d)

_, res = p.select_k(X)
res.to_csv(p.OUT / "escolha_k.csv", index=False)
fig, a1 = plt.subplots(figsize=(6.5, 3.6), dpi=200)
a1.plot(res.k, res.silhouette, "o-", color=CORES[0]); a1.set_xlabel("Número de grupos (k)"); a1.set_ylabel("Silhouette", color=CORES[0])
a1.axhline(p.SILHOUETTE_MIN, ls=":", color="gray")
a2 = a1.twinx(); a2.plot(res.k, res.calinski_harabasz, "s--", color=CORES[2]); a2.set_ylabel("Calinski-Harabasz", color=CORES[2]); a2.spines["right"].set_visible(True)
a1.axvline(3, color="black", alpha=.15, lw=8); fig.tight_layout(); fig.savefig(FIG / "fig_escolha_k.png"); plt.close(fig)

cols = [("retorno_anual", "Retorno anualizado (%)", 100), ("volatilidade_anual", "Volatilidade anualizada (%)", 100),
        ("mdd_abs", "Drawdown máximo (%)", 100), ("assimetria", "Assimetria", 1), ("consistencia", "Consistência (% dias positivos)", 100)]
fig, ax = plt.subplots(1, 5, figsize=(13, 3.6), dpi=200)
for a, (c, t, m) in zip(ax, cols):
    b = a.boxplot([d.loc[d.perfil == q, c] * m for q in PERFIS], tick_labels=["Cons.", "Mod.", "Arr."], showfliers=False, patch_artist=True, widths=.6)
    for bx, cc in zip(b["boxes"], CORES): bx.set_facecolor(cc); bx.set_alpha(.75)
    for md in b["medians"]: md.set_color("black")
    a.set_title(t, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig_boxplots_features.png"); plt.close(fig)

pca = PCA(2, random_state=42); Z = pca.fit_transform(X); v = pca.explained_variance_ratio_ * 100
fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
for q, c in zip(PERFIS, CORES):
    m = (d.perfil == q).values; ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=.45, c=c, label=f"{q} (n={m.sum()})", linewidths=0)
ax.set_xlabel(f"Componente 1 ({v[0]:.1f}%)".replace(".", ",")); ax.set_ylabel(f"Componente 2 ({v[1]:.1f}%)".replace(".", ","))
ax.legend(markerscale=3, frameon=False); fig.tight_layout(); fig.savefig(FIG / "fig_pca_grupos.png"); plt.close(fig)

# fracao de dias sem variacao de cota na janela (aviso na interface)
con = sqlite3.connect(p.DB_PATH)
q = "SELECT CD_CVM, DT_COMPTC, VL_QUOTA FROM inf_diario_unificado WHERE CD_CVM IN (%s) AND DT_COMPTC > '2022-11-01'" % ",".join(map(str, d.CD_CVM))
dd = pd.read_sql(q, con).sort_values(["CD_CVM", "DT_COMPTC"]).drop_duplicates(["CD_CVM", "DT_COMPTC"])
zero = {}
for cd, g in dd.groupby("CD_CVM"):
    v_ = g.VL_QUOTA.values[-(p.WINDOW + 1):]; r = v_[1:] / v_[:-1] - 1
    zero[int(cd)] = float((np.abs(r) < 1e-12).mean())
cad = pd.read_csv(p.OUT / "cadastro_classificado.csv")[["CD_CVM", "DENOM_SOCIAL", "CLASSE_ANBIMA"]]
t = d.merge(cad, on="CD_CVM", how="left")
t["DENOM_SOCIAL"] = t.DENOM_SOCIAL.fillna("Sem nome no cadastro"); t["CLASSE_ANBIMA"] = t.CLASSE_ANBIMA.fillna("Sem classe")
out = {"grupos": {}, "fundos": {}}
for q, g in t.groupby("perfil"):
    out["grupos"][q] = {"n": len(g), **g[p.FEATURE_COLS].median().round(4).to_dict()}
    out["fundos"][q] = [[int(r.CD_CVM), r.DENOM_SOCIAL.title() if r.DENOM_SOCIAL.isupper() else r.DENOM_SOCIAL, r.CLASSE_ANBIMA,
                         round(r.retorno_anual, 4), round(r.volatilidade_anual, 4), round(r.mdd_abs, 4), round(r.consistencia, 4),
                         round(r.assimetria, 3), round(zero.get(int(r.CD_CVM), 0), 3)] for r in g.itertuples()]
json.dump(out, open(p.BASE / "painel_dados.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(res.round(3).to_string()); print("pca", v.round(1), "| fundos com >20% dias sem variacao:", sum(x > .2 for x in zero.values()))
