"""
Cálculo das features, agrupamento por K-Means, validação e backtest de 2024
sobre a base unificada gerada por etl_real.py (tabela inf_diario_unificado).
"""
import os
import sqlite3
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, mannwhitneyu
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

SCRATCH = Path(os.path.expanduser("~/tcc_scratch"))
DB_PATH = SCRATCH / "fundos_real.db"
BASE = Path(__file__).resolve().parent  # data_real/
OUT = BASE / "reports_real"
OUT.mkdir(exist_ok=True)

WINDOW = 756
WINSOR_PCT = 0.01
FEATURE_COLS = ["retorno_anual", "volatilidade_anual", "mdd_abs", "assimetria", "consistencia"]
SILHOUETTE_MIN = 0.30
K_RANGE = range(2, 9)
# k fixado em 3, um grupo por perfil do questionário. A regra Silhouette >= 0,30 com
# maior Calinski-Harabasz indica k = 2; os dois valores são gravados no relatório.
K_FIXO = 3
MAX_SALTO_DIARIO = 0.5  # variação diária acima de 50% indica erro de cota ou mudança de base
RANDOM_STATE = 42

DATA_CORTE_PROD = pd.Timestamp("2025-12-31")
CORTE_TREINO_BT = pd.Timestamp("2023-12-31")
AVAL_INICIO, AVAL_FIM = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")


# ---------- elegibilidade ----------

def classify_eligibility(con, data_corte=None):
    """Classifica cada fundo. Com data_corte (backtest), usa só dados até o corte e
    troca a situação cadastral atual por 'tinha cota no último mês antes do corte'."""
    cad = pd.read_sql("SELECT * FROM cad_unificado", con)
    filtro = f"WHERE DT_COMPTC <= '{data_corte:%Y-%m-%d}'" if data_corte is not None else ""
    daily_stats = pd.read_sql(
        f"""SELECT CD_CVM, MAX(DT_COMPTC) as ultima_data,
                  COUNT(DISTINCT DT_COMPTC) as n_pregoes,
                  AVG(CASE WHEN NR_COTST<=1 THEN 1.0 ELSE 0.0 END) as share_cotst_1,
                  AVG(VL_PATRIM_LIQ) as vl_patrim_medio
           FROM inf_diario_unificado {filtro} GROUP BY CD_CVM""", con)
    cad = cad.merge(daily_stats, on="CD_CVM", how="left")
    cad[["n_pregoes", "share_cotst_1", "vl_patrim_medio"]] = cad[["n_pregoes", "share_cotst_1", "vl_patrim_medio"]].fillna(0)

    try:
        amb = set(pd.read_sql("SELECT CD_CVM FROM codigos_ambiguos", con)["CD_CVM"].astype(str))
    except Exception:
        amb = set()

    def classify(row):
        if str(row["CD_CVM"]) in amb:
            return "CODIGO_AMBIGUO"
        if row["share_cotst_1"] > 0.90 or str(row["EXCLUSIVO"]).upper() == "S":
            return "EXCLUSIVO"
        if data_corte is None:
            sit = str(row["SIT"]).upper()
            if "NORMAL" not in sit and "FUNCIONAMENTO" not in sit:
                return "CANCELADO_OU_LIQUIDACAO"
        elif not isinstance(row["ultima_data"], str) or pd.Timestamp(row["ultima_data"]) < data_corte - pd.Timedelta(days=31):
            return "CANCELADO_OU_LIQUIDACAO"
        if str(row["CONDOM"]).upper() == "FECHADO":
            return "FECHADO"
        if row["n_pregoes"] < WINDOW:
            return "HISTORICO_INSUFICIENTE"
        if row["vl_patrim_medio"] < 1_000_000:
            return "PATRIMONIO_BAIXO"
        return "ELEGIVEL"

    cad["CLASSIFICACAO"] = cad.apply(classify, axis=1)
    return cad


# ---------- features ----------

def winsorize(x, pct=WINSOR_PCT):
    lo, hi = np.quantile(x, [pct, 1 - pct])
    return np.clip(x, lo, hi)


def compute_features_for_fund(quotas: pd.Series, data_corte: pd.Timestamp):
    s = quotas[quotas.index <= data_corte].dropna()
    if len(s) < WINDOW + 1:
        return None
    s = s.iloc[-(WINDOW + 1):]
    prices = s.values
    r_brutos = prices[1:] / prices[:-1] - 1
    if np.abs(r_brutos).max() > MAX_SALTO_DIARIO:
        return None  # série com descontinuidade
    r_wins = winsorize(r_brutos)  # usada na volatilidade e na assimetria
    n = len(r_brutos)
    gross = np.prod(1 + r_brutos)  # retorno efetivo da cota
    ret_anual = gross ** (252 / n) - 1 if gross > 0 else -1.0
    vol_anual = np.std(r_wins, ddof=1) * np.sqrt(252)
    roll_max = np.maximum.accumulate(prices)
    mdd = abs(((prices - roll_max) / roll_max).min())
    return {
        "retorno_anual": ret_anual, "volatilidade_anual": vol_anual, "mdd_abs": mdd,
        "assimetria": float(skew(r_wins)), "consistencia": float(np.mean(r_brutos > 0)),
    }


def build_feature_table(con, data_corte, cd_cvms=None):
    q = "SELECT CD_CVM, DT_COMPTC, VL_QUOTA FROM inf_diario_unificado"
    if cd_cvms is not None:
        placeholders = ",".join("?" for _ in cd_cvms)
        q += f" WHERE CD_CVM IN ({placeholders})"
        daily = pd.read_sql(q, con, params=list(cd_cvms))
    else:
        daily = pd.read_sql(q, con)
    daily["DT_COMPTC"] = pd.to_datetime(daily["DT_COMPTC"])
    rows = []
    for cd, grp in daily.groupby("CD_CVM"):
        grp = grp.sort_values("DT_COMPTC").drop_duplicates("DT_COMPTC", keep="first").set_index("DT_COMPTC")["VL_QUOTA"]
        feats = compute_features_for_fund(grp, data_corte)
        if feats:
            feats["CD_CVM"] = cd
            rows.append(feats)
    return pd.DataFrame(rows)


# ---------- clustering ----------

def normalize(feats):
    pipe = Pipeline([("standard", StandardScaler())])
    return pipe.fit_transform(feats[FEATURE_COLS].values)


def select_k(X):
    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        results.append({"k": k, "silhouette": silhouette_score(X, labels),
                         "calinski_harabasz": calinski_harabasz_score(X, labels)})
    res_df = pd.DataFrame(results)
    cand = res_df[res_df["silhouette"] >= SILHOUETTE_MIN]
    best = (cand.loc[cand["calinski_harabasz"].idxmax()] if len(cand) else
            res_df.loc[res_df["silhouette"].idxmax()])
    return int(best["k"]), res_df


def run_kmeans(feats):
    X = normalize(feats)
    k, res_df = select_k(X)
    if K_FIXO:
        k = K_FIXO
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    feats = feats.copy()
    feats["cluster_raw"] = labels
    vol_order = feats.groupby("cluster_raw")["volatilidade_anual"].mean().sort_values()
    order_map = {old: new for new, old in enumerate(vol_order.index)}
    feats["cluster"] = feats["cluster_raw"].map(order_map)
    n_clusters = feats["cluster"].nunique()
    perfil_labels = ["Conservador", "Moderado", "Arrojado"]
    perfil_map = {c: perfil_labels[min(int(c * len(perfil_labels) / n_clusters), 2)] for c in sorted(feats["cluster"].unique())}
    feats["perfil"] = feats["cluster"].map(perfil_map)
    return feats, k, res_df, X


# ---------- validação ----------

def mann_whitney_pairwise(df):
    clusters = sorted(df["cluster"].unique())
    raw = []
    for feature in ["assimetria", "consistencia"]:
        for a, b in itertools.combinations(clusters, 2):
            xa, xb = df.loc[df.cluster == a, feature], df.loc[df.cluster == b, feature]
            if len(xa) < 2 or len(xb) < 2:
                continue
            stat, p = mannwhitneyu(xa, xb, alternative="two-sided")
            raw.append({"feature": feature, "a": a, "b": b, "p_raw": p})
    res = pd.DataFrame(raw)
    if len(res) == 0:
        return res
    reject, p_holm, _, _ = multipletests(res["p_raw"], alpha=0.05, method="holm")
    res["p_holm"] = p_holm
    res["significativo"] = reject
    return res


def main():
    con = sqlite3.connect(DB_PATH)

    print("=== Classificando elegibilidade ===")
    cad = classify_eligibility(con)
    print(cad["CLASSIFICACAO"].value_counts())
    cad.to_csv(OUT / "cadastro_classificado.csv", index=False)

    elig = cad[cad["CLASSIFICACAO"] == "ELEGIVEL"]["CD_CVM"].tolist()
    print(f"\n{len(elig)} fundos elegíveis para o motor de recomendação")

    print("\n=== Calculando features (produção, corte 31/12/2025) ===")
    feats = build_feature_table(con, DATA_CORTE_PROD, cd_cvms=elig)
    print(f"{len(feats)} fundos com histórico suficiente (>= {WINDOW} pregões)")
    feats.to_csv(OUT / "features_producao.csv", index=False)

    if len(feats) < 10:
        print("Poucos fundos elegíveis com histórico suficiente — abortando clustering.")
        return

    print("\n=== K-Means ===")
    feats_labeled, k, res_df, X = run_kmeans(feats)
    print(res_df)
    print(f"k escolhido: {k}")
    print(feats_labeled.groupby(["cluster", "perfil"]).size())
    feats_labeled.to_csv(OUT / "fundos_clusterizados.csv", index=False)

    print("\n=== Validação estatística ===")
    mw = mann_whitney_pairwise(feats_labeled)
    if len(mw):
        print(f"{mw['significativo'].sum()} de {len(mw)} pares significativos (Holm, alpha=0.05)")

    merged = feats_labeled.merge(cad[["CD_CVM", "CLASSE_ANBIMA"]], on="CD_CVM", how="left")
    merged["CLASSE_ANBIMA"] = merged["CLASSE_ANBIMA"].fillna("DESCONHECIDA")
    ari = adjusted_rand_score(merged["CLASSE_ANBIMA"], merged["cluster"])
    print(f"ARI vs classificação ANBIMA: {ari:.4f}")

    sil_kmeans = silhouette_score(X, feats_labeled["cluster"])
    sil_terc = silhouette_score(X, pd.qcut(feats_labeled["volatilidade_anual"], q=3, labels=False))
    sil_quart = silhouette_score(X, pd.qcut(feats_labeled["volatilidade_anual"], q=4, labels=False))
    cand = res_df[res_df["silhouette"] >= SILHOUETTE_MIN]
    k_regra = int(cand.loc[cand["calinski_harabasz"].idxmax(), "k"]) if len(cand) else int(res_df.loc[res_df["silhouette"].idxmax(), "k"])
    print(f"Silhouette K-Means: {sil_kmeans:.4f} | tercis de volatilidade: {sil_terc:.4f} | quartis: {sil_quart:.4f}")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    ve = pca.explained_variance_ratio_.sum()
    print(f"Variância explicada PCA 2D: {ve:.2%}")

    with open(OUT / "validacao_resumo.txt", "w") as f:
        f.write(f"k pela regra (Silhouette>=0,30 e maior Calinski-Harabasz): {k_regra} | k adotado: {k}\n")
        f.write(f"ARI vs ANBIMA: {ari:.4f}\n")
        f.write(f"Silhouette K-Means 5D: {sil_kmeans:.4f}\n")
        f.write(f"Silhouette tercis de volatilidade: {sil_terc:.4f}\n")
        f.write(f"Silhouette quartis de volatilidade: {sil_quart:.4f}\n")
        f.write(f"Variância explicada PCA 2D: {ve:.2%}\n")
        if len(mw):
            f.write(f"Mann-Whitney significativos: {mw['significativo'].sum()}/{len(mw)}\n")

    print("\n=== Backtesting 2024 ===")
    run_backtest(con, cad)

    con.close()
    print(f"\nResultados salvos em {OUT}")


# CDI diário (% ao dia útil), Banco Central do Brasil, SGS série 12, consultado em 17/10/2026.
# A taxa muda só nas datas de decisão do Copom; cada valor vale até a próxima mudança.
CDI_2024 = [("2024-01-02", 0.043739), ("2024-02-01", 0.041957), ("2024-03-21", 0.040168),
            ("2024-05-09", 0.039270), ("2024-09-19", 0.040168), ("2024-11-07", 0.041957),
            ("2024-12-12", 0.045513), ("2025-01-30", 0.049037), ("2025-03-20", 0.052531),
            ("2025-05-08", 0.054266), ("2025-06-20", 0.055131)]
# CDI_2024 reproduz a serie 12 do SGS/BCB (taxa diaria, % a.d.) em 2024 e 2025: as taxas so mudam nas datas acima.


def cdi_diario(index):
    """CDI diário (fração) alinhado às datas dos retornos das carteiras."""
    s = pd.Series({pd.Timestamp(d): v / 100 for d, v in CDI_2024})
    return s.reindex(s.index.union(index)).sort_index().ffill().reindex(index)


def sharpe_anual(r, cdi):
    exc = r - cdi.reindex(r.index)
    return exc.mean() / exc.std(ddof=1) * np.sqrt(252)


def run_backtest(con, cad):
    cad = classify_eligibility(con, CORTE_TREINO_BT)  # elegibilidade só com dados até o corte
    elig = cad[cad["CLASSIFICACAO"].isin(["ELEGIVEL"])][["CD_CVM", "CLASSE_ANBIMA"]]
    elig_ids = elig["CD_CVM"].tolist()

    feats_bt = build_feature_table(con, CORTE_TREINO_BT, cd_cvms=elig_ids)
    if len(feats_bt) < 10:
        print("Fundos insuficientes para backtest com o corte de treino de 31/12/2023.")
        return

    ingenua = feats_bt.nsmallest(10, "volatilidade_anual")["CD_CVM"].tolist()

    feats_bt_labeled, k_bt, _, _ = run_kmeans(feats_bt)
    feats_bt_labeled = feats_bt_labeled.merge(elig, on="CD_CVM", how="left")
    moderado = feats_bt_labeled[feats_bt_labeled["perfil"] == "Moderado"]
    if moderado.empty:
        moderado = feats_bt_labeled
    kmeans_port = (moderado.sort_values("retorno_anual", ascending=False)
                   .groupby("CLASSE_ANBIMA").head(3)["CD_CVM"].tolist())

    all_ids = list(set(ingenua) | set(kmeans_port))
    placeholders = ",".join("?" for _ in all_ids)
    q = f"""SELECT CD_CVM, DT_COMPTC, VL_QUOTA FROM inf_diario_unificado
            WHERE CD_CVM IN ({placeholders}) AND DT_COMPTC BETWEEN ? AND ?"""
    prices = pd.read_sql(q, con, params=all_ids + [
        (AVAL_INICIO - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), AVAL_FIM.strftime("%Y-%m-%d")])
    prices["DT_COMPTC"] = pd.to_datetime(prices["DT_COMPTC"])
    prices = prices.groupby(["DT_COMPTC", "CD_CVM"], as_index=False)["VL_QUOTA"].first()
    pivot = prices.pivot(index="DT_COMPTC", columns="CD_CVM", values="VL_QUOTA").sort_index()

    def port_returns(cols):
        cols = [c for c in cols if c in pivot.columns]
        rets = pivot[cols].ffill().pct_change(fill_method=None).dropna(how="all").fillna(0)
        rets = rets[rets.index >= AVAL_INICIO]  # apenas retornos de 2024
        return rets.mean(axis=1)

    r_ing = port_returns(ingenua)
    r_km = port_returns(kmeans_port)

    cdi = cdi_diario(r_km.index)
    acc_cdi = (1 + cdi).prod() - 1
    sh_ing, sh_km = sharpe_anual(r_ing, cdi), sharpe_anual(r_km, cdi)
    acc_ing = (1 + r_ing).prod() - 1
    acc_km = (1 + r_km).prod() - 1
    print(f"k backtest: {k_bt} | Carteira ingênua: {len(ingenua)} fundos, retorno 2024 = {acc_ing:.2%}")
    print(f"Carteira K-Means (Moderado): {len(kmeans_port)} fundos, retorno 2024 = {acc_km:.2%}")

    print(f"CDI acumulado no período: {acc_cdi:.2%} | Sharpe ingênua: {sh_ing:.2f} | Sharpe K-Means: {sh_km:.2f}")
    with open(OUT / "backtest_resumo.txt", "w") as f:
        f.write(f"CDI acumulado no período: {acc_cdi:.2%}\n")
        f.write(f"Sharpe anualizado (excesso sobre CDI): ingênua {sh_ing:.2f} | K-Means Moderado {sh_km:.2f}\n")
        f.write(f"Carteira ingênua ({len(ingenua)} fundos): retorno 2024 = {acc_ing:.2%}\n")
        f.write(f"Carteira K-Means Moderado ({len(kmeans_port)} fundos): retorno 2024 = {acc_km:.2%}\n")


if __name__ == "__main__":
    main()
