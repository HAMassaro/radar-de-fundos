"""Analise complementar sugerida na revisao: dentro de cada grupo, carteiras de 10 fundos
escolhidos por (a) menor volatilidade, (b) maior Sharpe no treino e (c) maior retorno no treino,
comparadas com a carteira do sistema, com 200 sorteios de 10 fundos do grupo e com o CDI.
Uso: python top10_por_perfil.py 2024|2025. Gera reports_real/top10_por_perfil_<ano>.txt."""
import sqlite3, sys
import numpy as np, pandas as pd
import pipeline_real as p
ANO = sys.argv[1] if len(sys.argv) > 1 else "2024"
if ANO == "2025":
    p.CORTE_TREINO_BT = p.pd.Timestamp("2024-12-31")
    p.AVAL_INICIO, p.AVAL_FIM = p.pd.Timestamp("2025-01-01"), p.pd.Timestamp("2025-12-31")
con = sqlite3.connect(p.DB_PATH)
cad = p.classify_eligibility(con, p.CORTE_TREINO_BT)
elig = cad[cad.CLASSIFICACAO == "ELEGIVEL"][["CD_CVM", "CLASSE_ANBIMA"]]
fb = p.build_feature_table(con, p.CORTE_TREINO_BT, cd_cvms=elig.CD_CVM.tolist())
fb, k, _, _ = p.run_kmeans(fb)
fb = fb.merge(elig, on="CD_CVM", how="left")
q = "SELECT CD_CVM, DT_COMPTC, VL_QUOTA FROM inf_diario_unificado WHERE CD_CVM IN (%s) AND DT_COMPTC BETWEEN ? AND ?" % ",".join("?" * len(fb))
pr = pd.read_sql(q, con, params=fb.CD_CVM.tolist() + [(p.CORTE_TREINO_BT - p.pd.Timedelta(days=11)).strftime("%Y-%m-%d"), p.AVAL_FIM.strftime("%Y-%m-%d")])
pr["DT_COMPTC"] = pd.to_datetime(pr.DT_COMPTC)
pv = pr.groupby(["DT_COMPTC", "CD_CVM"]).VL_QUOTA.first().unstack().sort_index()
qf = pv.ffill()
erro = ((qf / qf.shift(1) - 1).abs() > p.MAX_SALTO_DIARIO) & ((qf.shift(-1) / qf.shift(1) - 1).abs() < 0.05)
pv = pv.mask(erro)
rets = pv.ffill().pct_change(fill_method=None)
rets = rets[rets.index >= p.AVAL_INICIO]
cdi = p.cdi_diario(rets.index)
# CDI anualizado da janela de treino (tres anos antes do corte), a partir do CDI anual
# divulgado pelo BCB: 2021 4,42%; 2022 12,37%; 2023 13,04%; 2024 10,88%.
CDI_ANUAL = {2021: .0442, 2022: .1237, 2023: .1304, 2024: .1088}
a0 = p.CORTE_TREINO_BT.year
cdi_ano_treino = np.prod([1 + CDI_ANUAL[a] for a in (a0 - 2, a0 - 1, a0)]) ** (1 / 3) - 1
fb["sharpe_treino"] = (fb.retorno_anual - cdi_ano_treino) / fb.volatilidade_anual.clip(lower=1e-4)

def serie(ids):
    ids = [i for i in ids if i in rets.columns]
    return rets[ids].fillna(0).mean(axis=1), len(ids)
def stats(r):
    e = r - cdi.reindex(r.index)
    return (1 + r).prod() - 1, e.mean() / e.std() * np.sqrt(252)
L = [f"CDI {ANO}: {(1 + cdi).prod() - 1:.2%} | CDI anualizado na janela de treino: {cdi_ano_treino:.2%}"]
rng = np.random.default_rng(42)
regras = {"10 de menor volatilidade": ("volatilidade_anual", True),
          "10 de maior Sharpe no treino": ("sharpe_treino", False),
          "10 de maior retorno no treino": ("retorno_anual", False)}
for perfil in ["Conservador", "Moderado", "Arrojado"]:
    g = fb[fb.perfil == perfil]
    L.append(f"\n{perfil} (grupo com {len(g)} fundos)")
    r, n = serie(g.sort_values("retorno_anual", ascending=False).groupby("CLASSE_ANBIMA").head(3).CD_CVM)
    ret, sh = stats(r); L.append(f"  carteira do sistema (3 por classe) N={n}: ret={ret:.2%} sharpe={sh:.2f}")
    for nome, (col, asc) in regras.items():
        r, n = serie(g.sort_values(col, ascending=asc).CD_CVM.head(10))
        ret, sh = stats(r); L.append(f"  {nome} N={n}: ret={ret:.2%} sharpe={sh:.2f}")
    ids_g = [i for i in g.CD_CVM if i in rets.columns]
    sims = [stats(serie(rng.choice(ids_g, 10, replace=False))[0])[0] for _ in range(200)]
    L.append(f"  sorteios de 10 fundos do grupo (200): mediana={np.median(sims):.2%} p5={np.percentile(sims, 5):.2%} p95={np.percentile(sims, 95):.2%}")
out = "\n".join(L); (p.OUT / f"top10_por_perfil_{ANO}.txt").write_text(out); print(out)
