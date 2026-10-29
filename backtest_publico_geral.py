"""Verificacao: mesmo backtest, mas a carteira, os sorteios e as referencias usam so fundos
de publico geral (o modelo continua treinado com todos os elegiveis).
Backtest de 2024 com comparacoes: carteira do sistema (3 fundos de maior retorno por classe
ANBIMA em cada grupo), 200 sorteios do mesmo grupo e mesmo tamanho, carteira de menor
volatilidade de mesmo tamanho, carteira ingenua de 10 fundos, regras alternativas e CDI.
Elegibilidade calculada so com dados ate o corte de treino. Gera reports_real/analises_backtest.txt
e figuras/fig_backtest_2024.png."""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
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
pub = pd.read_sql("SELECT CD_CVM, PUBLICO_ALVO FROM cad_unificado", con)
pub["CD_CVM"] = pd.to_numeric(pub.CD_CVM, errors="coerce")
pub = pub.dropna(subset=["CD_CVM"]).drop_duplicates("CD_CVM"); pub["CD_CVM"] = pub.CD_CVM.astype(int).astype(str); fb["CD_CVM"] = fb.CD_CVM.astype(str)
fb = fb.merge(pub, on="CD_CVM", how="left")
fb = fb[fb.PUBLICO_ALVO == "Público Geral"].copy()
q = "SELECT CD_CVM, DT_COMPTC, VL_QUOTA FROM inf_diario_unificado WHERE CD_CVM IN (%s) AND DT_COMPTC BETWEEN ? AND ?" % ",".join("?" * len(fb))
pr = pd.read_sql(q, con, params=fb.CD_CVM.tolist() + [(p.CORTE_TREINO_BT - p.pd.Timedelta(days=11)).strftime("%Y-%m-%d"), p.AVAL_FIM.strftime("%Y-%m-%d")])
pr["DT_COMPTC"] = pd.to_datetime(pr.DT_COMPTC)
pv = pr.groupby(["DT_COMPTC", "CD_CVM"]).VL_QUOTA.first().unstack().sort_index()
# erro de digitacao: salto > 50% que se desfaz no pregao seguinte (cota volta a menos de 5% do dia anterior)
qf = pv.ffill()
salto = (qf / qf.shift(1) - 1).abs() > p.MAX_SALTO_DIARIO
volta = (qf.shift(-1) / qf.shift(1) - 1).abs() < 0.05
erro = salto & volta
N_ERROS = int(erro.values.sum())
pv = pv.mask(erro)
rets = pv.ffill().pct_change(fill_method=None)
rets = rets[rets.index >= p.AVAL_INICIO]
cdi = p.cdi_diario(rets.index)

def serie(ids):
    ids = [i for i in ids if i in rets.columns]
    return rets[ids].fillna(0).mean(axis=1), len(ids)

def stats(r):
    e = r - cdi.reindex(r.index)
    return (1 + r).prod() - 1, e.mean() / e.std() * np.sqrt(252), e.mean() * 252, e.std() * np.sqrt(252)

fb["razao"] = fb.retorno_anual / fb.volatilidade_anual.clip(lower=1e-4)
regras = {"A retorno passado": ("retorno_anual", False), "B retorno/volatilidade": ("razao", False),
          "C menor drawdown": ("mdd_abs", True), "D menor volatilidade": ("volatilidade_anual", True)}
low = fb.sort_values("volatilidade_anual")
L = [f"CDI {ANO}: {(1 + cdi).prod() - 1:.2%} | fundos elegiveis no corte: {len(elig)} | com features: {len(fb)} | com cota no ano: {rets.shape[1]}"]
rng = np.random.default_rng(42)
curvas = {}
for perfil in ["Conservador", "Moderado", "Arrojado"]:
    g = fb[fb.perfil == perfil]
    L.append(f"\n{perfil} (grupo com {len(g)} fundos)")
    for nome, (col, asc) in regras.items():
        r, n = serie(g.sort_values(col, ascending=asc).groupby("CLASSE_ANBIMA").head(3).CD_CVM)
        ret, sh, em, ed = stats(r)
        L.append(f"  regra {nome}: N={n} ret={ret:.2%} sharpe={sh:.2f} excesso_med={em:.2%} desv_exc={ed:.2%}")
        if nome.startswith("A"):
            nA = n; curvas[perfil] = r
    ids_g = [i for i in g.CD_CVM if i in rets.columns]
    sims = [stats(serie(rng.choice(ids_g, nA, replace=False))[0])[0] for _ in range(200)]
    L.append(f"  sorteios (200, N={nA}): mediana={np.median(sims):.2%} p5={np.percentile(sims, 5):.2%} p95={np.percentile(sims, 95):.2%}")
    r, n = serie(low.CD_CVM.head(nA)); ret, sh, em, ed = stats(r)
    L.append(f"  menor volatilidade do universo N={n}: ret={ret:.2%} sharpe={sh:.2f}")
ing, n = serie(fb.nsmallest(10, "volatilidade_anual").CD_CVM)
ret, sh, em, ed = stats(ing)
L.append(f"\ningenua 10 fundos: ret={ret:.2%} sharpe={sh:.2f} excesso_med={em:.2%} desv_exc={ed:.2%} pior dia {ing.idxmin():%Y-%m-%d} {ing.min():.2%}")
mx = rets.abs().max(); L.append(f"cotas corrigidas como erro de digitacao: {N_ERROS} | fundos com variacao diaria >50% restante: {int((mx > 0.5).sum())}")
out = "\n".join(L); (p.OUT / f"backtest_publico_geral_{ANO}.txt").write_text(out); print(out)
