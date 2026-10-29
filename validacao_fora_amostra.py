"""Validacao dos grupos com dados fora da amostra: os grupos formados com dados ate o corte
sao comparados pelo comportamento realizado no ano seguinte (volatilidade, queda maxima e
fracao de dias positivos). Inclui o caso dos fundos do terco de menor volatilidade que o
K-Means colocou no Moderado. Uso: python validacao_fora_amostra.py 2024|2025"""
import sys
import numpy as np, pandas as pd
ANO = sys.argv[1] if len(sys.argv) > 1 else "2024"
sys.argv = ["x", ANO]
src = open("analises_backtest.py").read().split('fb["razao"]')[0]
exec(src)
r = rets.copy()
def real(s):
    s = s.dropna()
    if len(s) < 120: return None
    q = (1 + s).cumprod(); dd = (1 - q / q.cummax()).max()
    return pd.Series({"vol": s.std() * np.sqrt(252), "mdd": dd, "cons": (s > 0).mean(), "ret": q.iloc[-1] - 1})
R = pd.DataFrame({c: real(r[c]) for c in r.columns if real(r[c]) is not None}).T
R.index = R.index.astype(str)
fb["CD_CVM"] = fb.CD_CVM.astype(str)
d = fb.set_index("CD_CVM").join(R, how="inner")
L = [f"Ano {ANO}: fundos com cota suficiente no ano: {len(d)}"]
med = d.groupby("perfil")[["vol", "mdd", "cons", "ret"]].median()
L.append("Medianas realizadas no ano (vol, queda maxima, dias positivos, retorno):")
for p in ["Conservador", "Moderado", "Arrojado"]:
    m = med.loc[p]; L.append(f"  {p}: n={int((d.perfil==p).sum())} vol={m.vol:.2%} mdd={m.mdd:.2%} cons={m.cons:.1%} ret={m.ret:.2%}")
# ordem de risco preservada por fundo: fracao de pares
from scipy.stats import spearmanr
rk = d.perfil.map({"Conservador": 0, "Moderado": 1, "Arrojado": 2})
L.append(f"Spearman entre perfil e vol realizada: {spearmanr(rk, d.vol).correlation:.2f}; com queda realizada: {spearmanr(rk, d.mdd).correlation:.2f}")
ter = pd.qcut(d.volatilidade_anual, 3, labels=False)
a = d[(ter == 0) & (d.perfil == "Moderado")]; b = d[(ter == 0) & (d.perfil == "Conservador")]
from scipy.stats import mannwhitneyu
L.append(f"Terco de menor volatilidade no treino: no Moderado n={len(a)}, no Conservador n={len(b)}")
for c, nome in [("vol", "volatilidade"), ("mdd", "queda maxima"), ("cons", "dias positivos"), ("ret", "retorno")]:
    p = mannwhitneyu(a[c], b[c]).pvalue
    L.append(f"  {nome}: Moderado {a[c].median():.2%} x Conservador {b[c].median():.2%} (p={p:.1e})")
out = "\n".join(L); (p_out := p.OUT / f"validacao_fora_amostra_{ANO}.txt") if False else None
open(f"reports_real/validacao_fora_amostra_{ANO}.txt", "w").write(out); print(out)
