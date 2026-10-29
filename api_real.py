"""API do sistema: questionário de perfil e fundos de cada grupo.

Uso: uvicorn api_real:app --reload
O questionario de triagem tem 6 perguntas de 0 a 3 pontos (total 0-18):
ate 6 = Conservador, de 7 a 12 = Moderado, acima de 12 = Arrojado.
"""
from pathlib import Path
from typing import List, Optional
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OUT = Path(__file__).resolve().parent / "reports_real"
PERFIS = ("Conservador", "Moderado", "Arrojado")
# publico-alvo da classe (cadastro CVM) permitido para cada tipo de investidor
ACESSO = {"comum": {"Público Geral"}, "qualificado": {"Público Geral", "Qualificado"},
          "profissional": None}  # None = todos, inclusive sem informacao
AVISO = ("Ferramenta educativa de triagem baseada no comportamento passado dos fundos. "
         "Nao e recomendacao de investimento e nao ha garantia de retorno futuro.")

app = FastAPI(title="Radar de Fundos", version="0.2.0")
_cache = {}


def base():
    if "df" not in _cache:
        feats = OUT / "fundos_clusterizados.csv"
        cad = OUT / "cadastro_classificado.csv"
        if not feats.exists() or not cad.exists():
            raise HTTPException(503, "Rode pipeline_real.py antes de subir a API.")
        f = pd.read_csv(feats)
        c = pd.read_csv(cad)[["CD_CVM", "DENOM_SOCIAL", "CLASSE_ANBIMA", "vl_patrim_medio"]]
        df = f.merge(c, on="CD_CVM", how="left")
        pub = json.loads((OUT / "publico_alvo.json").read_text())
        df["PUBLICO_ALVO"] = df.CD_CVM.astype(int).astype(str).map(pub)
        _cache["df"] = df
        _cache["med"] = df.groupby("perfil")[
            ["retorno_anual", "volatilidade_anual", "mdd_abs", "consistencia"]].median()
    return _cache["df"], _cache["med"]


class Questionario(BaseModel):
    horizonte_investimento: int = Field(ge=0, le=3, description="0 = curtissimo prazo, 3 = longo prazo")
    tolerancia_perdas: int = Field(ge=0, le=3)
    experiencia_mercado: int = Field(ge=0, le=3)
    objetivo: int = Field(ge=0, le=3, description="0 = preservacao, 3 = crescimento agressivo")
    reserva_emergencia: int = Field(ge=0, le=3)
    reacao_queda_10pct: int = Field(ge=0, le=3, description="0 = venderia tudo, 3 = aportaria mais")


def pontuacao(q: Questionario) -> int:
    return (q.horizonte_investimento + q.tolerancia_perdas + q.experiencia_mercado
            + q.objetivo + q.reserva_emergencia + q.reacao_queda_10pct)


def perfil_de(total: int) -> str:
    return "Conservador" if total <= 6 else "Moderado" if total <= 12 else "Arrojado"


class Recomendacao(BaseModel):
    codigo_cvm: int
    nome: Optional[str] = None
    classe_anbima: Optional[str] = None
    perfil: str
    retorno_anual: float
    volatilidade_anual: float
    mdd_abs: float
    consistencia: float
    justificativa: str


@app.post("/triagem")
def triagem(q: Questionario):
    total = pontuacao(q)
    return {"pontuacao": total, "perfil": perfil_de(total)}


@app.get("/recomendacao/{perfil}", response_model=List[Recomendacao])
def recomendacao(perfil: str, investidor: str = "comum", limite: int = 10):
    perfil = perfil.capitalize()
    if perfil not in PERFIS:
        raise HTTPException(400, "Perfil deve ser Conservador, Moderado ou Arrojado")
    if investidor not in ACESSO:
        raise HTTPException(400, "investidor deve ser comum, qualificado ou profissional")
    df, med = base()
    sub = df[df["perfil"] == perfil]
    if ACESSO[investidor] is not None:
        sub = sub[sub.PUBLICO_ALVO.isin(ACESSO[investidor])]
    sub = sub.sort_values("retorno_anual", ascending=False).head(limite)
    m = med.loc[perfil]
    out = []
    for _, r in sub.iterrows():
        just = (f"Pertence ao grupo {perfil} ({int((df.perfil == perfil).sum())} fundos). "
                f"Nos ultimos 756 pregoes: retorno anualizado {r['retorno_anual']:.1%} "
                f"(mediana do grupo {m['retorno_anual']:.1%}), volatilidade {r['volatilidade_anual']:.1%} "
                f"(grupo {m['volatilidade_anual']:.1%}), maior queda {r['mdd_abs']:.1%} "
                f"(grupo {m['mdd_abs']:.1%}), {r['consistencia']:.0%} dos dias positivos. "
                + AVISO)
        out.append(Recomendacao(
            codigo_cvm=int(r["CD_CVM"]), nome=r.get("DENOM_SOCIAL") if pd.notna(r.get("DENOM_SOCIAL")) else None,
            classe_anbima=r.get("CLASSE_ANBIMA") if pd.notna(r.get("CLASSE_ANBIMA")) else None,
            perfil=perfil, retorno_anual=float(r["retorno_anual"]),
            volatilidade_anual=float(r["volatilidade_anual"]), mdd_abs=float(r["mdd_abs"]),
            consistencia=float(r["consistencia"]), justificativa=just))
    return out


@app.get("/grupos")
def grupos():
    df, med = base()
    res = med.round(4).to_dict("index")
    for p in PERFIS:
        res[p]["fundos"] = int((df.perfil == p).sum())
    return {"grupos": res, "aviso": AVISO}


@app.get("/health")
def health():
    return {"status": "ok"}
