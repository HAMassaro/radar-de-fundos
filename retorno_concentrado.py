"""Marca fundos cujo retorno na janela de producao depende de saltos isolados:
os 3 maiores retornos diarios respondem por mais da metade do retorno total (em log)
e o maior deles passa de 5% e de 20 desvios padrao dos demais dias.
Gera reports_real/retorno_concentrado.csv."""
import sqlite3, numpy as np, pandas as pd
import pipeline_real as p
con = sqlite3.connect(p.DB_PATH)
fc = pd.read_csv(p.OUT / "fundos_clusterizados.csv", dtype={"CD_CVM": str})
ids = set(fc.CD_CVM)
out = []
for ch in pd.read_sql("SELECT CD_CVM, DT_COMPTC, VL_QUOTA FROM inf_diario_unificado WHERE DT_COMPTC >= '2022-09-01' ORDER BY CD_CVM, DT_COMPTC", con, chunksize=2_000_000):
    out.append(ch[ch.CD_CVM.isin(ids)])
q = pd.concat(out)
res = []
for cd, g in q.groupby("CD_CVM"):
    v = g.VL_QUOTA.values[-757:]
    lr = np.diff(np.log(v))
    tot = lr.sum()
    srt = np.sort(lr); top3 = srt[-3:].sum(); dp = srt[:-3].std()
    salto = srt[-1] > 0.05 and srt[-1] > 20 * dp
    res.append((cd, tot, top3, bool(tot > 0 and top3 / tot > 0.5 and salto)))
r = pd.DataFrame(res, columns=["CD_CVM", "log_ret_total", "log_ret_top3", "concentrado"])
r.to_csv(p.OUT / "retorno_concentrado.csv", index=False)
m = r.merge(fc[["CD_CVM", "perfil"]], on="CD_CVM")
print(m.groupby("perfil").concentrado.agg(["sum", "count"]))
print(m[m.CD_CVM.isin(["833", "56545"])])
