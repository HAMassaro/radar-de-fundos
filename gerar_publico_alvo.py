"""Gera reports_real/publico_alvo.json (código CVM -> público-alvo da classe), usado pela API e pela interface."""
import json, sqlite3
import pandas as pd
import pipeline_real as p

con = sqlite3.connect(p.DB_PATH)
cad = pd.read_sql("SELECT CD_CVM, PUBLICO_ALVO FROM cad_unificado", con)
cad["CD_CVM"] = pd.to_numeric(cad.CD_CVM, errors="coerce")
cad = cad.dropna(subset=["CD_CVM"]).drop_duplicates("CD_CVM")
mapa = {int(c): a for c, a in zip(cad.CD_CVM, cad.PUBLICO_ALVO) if a}
(p.OUT / "publico_alvo.json").write_text(json.dumps(mapa, ensure_ascii=False))
print(f"{len(mapa)} códigos com público-alvo informado")
