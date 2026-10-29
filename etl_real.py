"""
Extração e unificação dos informes diários da CVM.

Os informes até novembro de 2023 identificam o fundo por CNPJ_FUNDO; a partir de
dezembro de 2023, após a Resolução CVM 175, por CNPJ_FUNDO_CLASSE. O Código CVM,
obtido nos arquivos registro_fundo.csv e registro_classe.csv, é usado como
identificador único ao longo da mudança.

O arquivo cad_fi.csv não é usado para a situação dos fundos: ele mostra o estado
atual, e fundos que migraram para o modelo de classes aparecem nele como cancelados.
"""
import os
import re
import sqlite3
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent  # roda a partir de data_real/
RAW = BASE / "raw"
SCRATCH = Path(os.path.expanduser("~/tcc_scratch"))
SCRATCH.mkdir(parents=True, exist_ok=True)
DB_PATH = SCRATCH / "fundos_real.db"  # SQLite não escreve bem em pastas montadas via FUSE
CUTOFF_NEW_SCHEMA = "202312"  # primeiro mês no schema novo


def only_digits(s):
    if pd.isna(s):
        return None
    return re.sub(r"\D", "", str(s))


def build_cdcvm_maps():
    rf = pd.read_csv(BASE / "registro_fundo_classe" / "registro_fundo.csv", sep=";", encoding="latin1", dtype=str)
    rf["CNPJ_DIGITS"] = rf["CNPJ_Fundo"].map(only_digits)
    # um CNPJ_Fundo pode aparecer mais de uma vez no registro (histórico de
    # reaproveitamento de código); fica com o registro de Codigo_CVM mais recente
    rf_dedup = rf.dropna(subset=["CNPJ_DIGITS"]).drop_duplicates(subset=["CNPJ_DIGITS"], keep="last")
    cnpj_to_cdcvm_old = dict(zip(rf_dedup["CNPJ_DIGITS"], rf_dedup["Codigo_CVM"]))
    idfundo_to_cdcvm = dict(zip(rf["ID_Registro_Fundo"], rf["Codigo_CVM"]))
    fundo_attrs = rf.set_index("Codigo_CVM")[["Denominacao_Social", "Situacao", "Data_Cancelamento",
                                               "Data_Adaptacao_RCVM175"]]
    fundo_attrs = fundo_attrs[~fundo_attrs.index.duplicated(keep="last")]

    rc = pd.read_csv(BASE / "registro_fundo_classe" / "registro_classe.csv", sep=";", encoding="latin1", dtype=str)
    rc["CNPJ_DIGITS"] = rc["CNPJ_Classe"].map(only_digits)
    rc["CD_CVM"] = rc["ID_Registro_Fundo"].map(idfundo_to_cdcvm)
    cnpj_to_cdcvm_new = dict(zip(rc["CNPJ_DIGITS"], rc["CD_CVM"]))

    rc2 = rc.dropna(subset=["CD_CVM"]).copy()
    rc2["Patrimonio_Liquido"] = pd.to_numeric(rc2["Patrimonio_Liquido"].str.replace(",", "."), errors="coerce")
    rc2 = rc2.sort_values("Patrimonio_Liquido", ascending=False)
    rc_principal = rc2.drop_duplicates(subset=["CD_CVM"], keep="first").set_index("CD_CVM")[[
        "Denominacao_Social", "Situacao", "Forma_Condominio", "Exclusivo",
        "Classificacao_Anbima", "Publico_Alvo"
    ]]

    return cnpj_to_cdcvm_old, cnpj_to_cdcvm_new, fundo_attrs, rc_principal, rc2


def unify_cadastro(fundo_attrs: pd.DataFrame, rc_principal: pd.DataFrame) -> pd.DataFrame:
    """Cadastro unificado por Codigo_CVM. Prioriza o status da CLASSE (mais
    granular e atual); cai para o status do FUNDO pai quando a classe não
    existe (fundo nunca migrou / é anterior à RCVM 175 e ainda não tem
    classe registrada, ou já foi de fato extinto)."""
    all_cd = set(fundo_attrs.index) | set(rc_principal.index)
    rows = []
    for cd in all_cd:
        classe = rc_principal.loc[cd] if cd in rc_principal.index else None
        fundo = fundo_attrs.loc[cd] if cd in fundo_attrs.index else None
        if classe is not None:
            rows.append({
                "CD_CVM": cd,
                "DENOM_SOCIAL": classe["Denominacao_Social"],
                "SIT": classe["Situacao"],
                "CONDOM": classe["Forma_Condominio"],
                "EXCLUSIVO": classe["Exclusivo"],
                "CLASSE_ANBIMA": classe["Classificacao_Anbima"],
                "PUBLICO_ALVO": classe["Publico_Alvo"],
                "FONTE_CADASTRO": "classe",
            })
        else:
            rows.append({
                "CD_CVM": cd,
                "DENOM_SOCIAL": fundo["Denominacao_Social"] if fundo is not None else None,
                "SIT": fundo["Situacao"] if fundo is not None else None,
                "CONDOM": None,
                "EXCLUSIVO": None,
                "CLASSE_ANBIMA": None,
                "PUBLICO_ALVO": None,
                "FONTE_CADASTRO": "fundo_apenas",
            })
    return pd.DataFrame(rows).set_index("CD_CVM")


def process_month(fname: str, is_new_schema: bool, cnpj_to_cdcvm: dict, rc2: pd.DataFrame):
    df = pd.read_csv(RAW / fname, sep=";", encoding="latin1", dtype=str)
    if is_new_schema:
        # linhas de subclasse (ID_SUBCLASSE preenchido) têm cotas
        # próprias e duplicam fundo+data; mantém apenas o nível da classe.
        df = df[df["ID_SUBCLASSE"].isna()]
        df["CNPJ_DIGITS"] = df["CNPJ_FUNDO_CLASSE"].map(only_digits)
        principal_cnpjs = set(rc2.drop_duplicates(subset=["CD_CVM"], keep="first")["CNPJ_DIGITS"])
        df = df[df["CNPJ_DIGITS"].isin(principal_cnpjs)]
    else:
        df["CNPJ_DIGITS"] = df["CNPJ_FUNDO"].map(only_digits)

    df["CD_CVM"] = df["CNPJ_DIGITS"].map(cnpj_to_cdcvm)
    df = df.dropna(subset=["CD_CVM"])
    df["VL_QUOTA"] = pd.to_numeric(df["VL_QUOTA"], errors="coerce")
    df["VL_PATRIM_LIQ"] = pd.to_numeric(df["VL_PATRIM_LIQ"], errors="coerce")
    df["NR_COTST"] = pd.to_numeric(df["NR_COTST"], errors="coerce")
    out = df[["CD_CVM", "DT_COMPTC", "VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"]].dropna(subset=["VL_QUOTA"])
    out = out[out["VL_QUOTA"] > 0]
    # o Codigo_CVM não é único. Se mais de um CNPJ distinto
    # mapeia para o mesmo código no mesmo mês, as cotas de fundos diferentes
    # seriam misturadas numa só série (volatilidade falsa). Esses códigos são
    # marcados como ambíguos e removidos ao final.
    amb = df.groupby("CD_CVM")["CNPJ_DIGITS"].nunique()
    amb = set(amb[amb > 1].index)
    return out, amb


def run():
    print("[ETL-real] Construindo mapas Codigo_CVM (via registro_fundo.csv) ...")
    cnpj_old, cnpj_new, fundo_attrs, rc_principal, rc2 = build_cdcvm_maps()
    cad_unificado = unify_cadastro(fundo_attrs, rc_principal)
    print(f"[ETL-real] {len(cad_unificado)} fundos no cadastro unificado (Codigo_CVM)")
    print(cad_unificado["FONTE_CADASTRO"].value_counts())
    print(cad_unificado["SIT"].value_counts(dropna=False).head(10))

    files = sorted(RAW.glob("inf_diario_fi_*.csv"))
    con = sqlite3.connect(DB_PATH)
    cad_unificado.reset_index().to_sql("cad_unificado", con, if_exists="replace", index=False)

    first = True
    total_rows = 0
    ambiguos = set()
    for f in files:
        month = f.stem.replace("inf_diario_fi_", "")
        is_new = month >= CUTOFF_NEW_SCHEMA
        mapping = cnpj_new if is_new else cnpj_old
        out, amb = process_month(f.name, is_new, mapping, rc2)
        ambiguos |= amb
        out.to_sql("inf_diario_unificado", con, if_exists="replace" if first else "append", index=False)
        first = False
        total_rows += len(out)
        print(f"[ETL-real] {month} ({'novo' if is_new else 'antigo'} schema): {len(out)} linhas")

    pd.DataFrame({"CD_CVM": sorted(ambiguos)}).to_sql("codigos_ambiguos", con, if_exists="replace", index=False)
    con.execute("DELETE FROM inf_diario_unificado WHERE CD_CVM IN (SELECT CD_CVM FROM codigos_ambiguos)")
    print(f"[ETL-real] {len(ambiguos)} códigos CVM ambíguos removidos")
    # linhas repetidas para o mesmo fundo e data: mantém a primeira ocorrência
    n_dup = con.execute("""DELETE FROM inf_diario_unificado WHERE rowid NOT IN
                           (SELECT MIN(rowid) FROM inf_diario_unificado GROUP BY CD_CVM, DT_COMPTC)""").rowcount
    con.commit()
    print(f"[ETL-real] {n_dup} linhas repetidas de fundo e data removidas")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cd ON inf_diario_unificado(CD_CVM)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_dt ON inf_diario_unificado(DT_COMPTC)")

    # diagnóstico rápido de continuidade
    cont = pd.read_sql(
        "SELECT CD_CVM, COUNT(DISTINCT DT_COMPTC) as n_pregoes FROM inf_diario_unificado GROUP BY CD_CVM", con)
    print("\n[ETL-real] Distribuição de pregões por fundo após unificação:")
    print(cont["n_pregoes"].describe())
    print(f"Fundos com >= 756 pregões: {(cont['n_pregoes'] >= 756).sum()}")

    con.commit()
    con.close()
    print(f"\n[ETL-real] TOTAL: {total_rows} linhas diárias em {DB_PATH}")


if __name__ == "__main__":
    run()
