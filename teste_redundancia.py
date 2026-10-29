import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
d=pd.read_csv('reports_real/fundos_clusterizados.csv')
F=['retorno_anual','volatilidade_anual','mdd_abs','assimetria','consistencia']
km=d.perfil.map({'Conservador':0,'Moderado':1,'Arrojado':2}).values
def run(cols=None,pca=None):
    X=StandardScaler().fit_transform(d[cols or F])
    if pca: X=PCA(pca,random_state=42).fit_transform(X)
    l=KMeans(3,n_init=10,random_state=42).fit_predict(X)
    v=pd.Series(d.volatilidade_anual.values).groupby(l).mean().sort_values(); m={c:i for i,c in enumerate(v.index)}; l=np.array([m[x] for x in l])
    return adjusted_rand_score(km,l),(l==km).mean(),np.bincount(l)
for nome,cols,p in [('sem drawdown',[c for c in F if c!='mdd_abs'],None),('sem consistencia',[c for c in F if c!='consistencia'],None),('sem drawdown e sem consistencia',['retorno_anual','volatilidade_anual','assimetria'],None),('PCA 3 componentes',None,3)]:
    a,s,b=run(cols,p); print(f"{nome:32s} ARI {a:.3f} mesmo perfil {s:.1%} tamanhos {b}")
