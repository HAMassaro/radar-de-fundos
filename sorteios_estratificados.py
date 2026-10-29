"""Sorteios estratificados por classe ANBIMA: cada sorteio tem o mesmo numero de fundos por
classe que a carteira do sistema, para separar o efeito da regra de selecao do efeito da
composicao por classe. Uso: python sorteios_estratificados.py 2024|2025"""
import sys
import numpy as np, pandas as pd
ANO = sys.argv[1] if len(sys.argv) > 1 else "2024"
sys.argv = ["x", ANO]
src = open("analises_backtest.py").read().split('fb["razao"]')[0]
exec(src)
rng = np.random.default_rng(42)
L = [f"Ano {ANO}"]
for perfil in ["Conservador", "Moderado", "Arrojado"]:
    g = fb[fb.perfil == perfil]
    g = g[g.CD_CVM.isin(rets.columns)]
    sel = g.sort_values("retorno_anual", ascending=False).groupby("CLASSE_ANBIMA").head(3)
    ret_sel = (1 + serie(sel.CD_CVM)[0]).prod() - 1
    cont = sel.CLASSE_ANBIMA.value_counts()
    sims = []
    for _ in range(200):
        ids = []
        for cl, n in cont.items():
            pool = g[g.CLASSE_ANBIMA == cl].CD_CVM.values
            ids += list(rng.choice(pool, n, replace=False))
        sims.append((1 + serie(ids)[0]).prod() - 1)
    sims = np.array(sims)
    L.append(f"  {perfil}: carteira {ret_sel:.2%} | sorteios estratificados mediana {np.median(sims):.2%} p5 {np.percentile(sims,5):.2%} p95 {np.percentile(sims,95):.2%} | fracao de sorteios abaixo da carteira {np.mean(sims < ret_sel):.0%}")
out = "\n".join(L); open(f"reports_real/sorteios_estratificados_{ANO}.txt", "w").write(out); print(out)
