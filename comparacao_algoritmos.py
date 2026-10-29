import pandas as pd, numpy as np, time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
d=pd.read_csv('reports_real/fundos_clusterizados.csv')
F=['retorno_anual','volatilidade_anual','mdd_abs','assimetria','consistencia']
X=StandardScaler().fit_transform(d[F])
km=d.perfil.map({'Conservador':0,'Moderado':1,'Arrojado':2}).values
km2=KMeans(3,init='k-means++',n_init=10,random_state=42).fit_predict(X)
print('check kmeans reproduz (ARI com rotulos salvos):',round(adjusted_rand_score(km,km2),3))
def nomeia(lab):
    v=pd.Series(d.volatilidade_anual.values).groupby(lab).mean().sort_values()
    m={c:i for i,c in enumerate(v.index)}; return np.array([m[x] for x in lab])
ter=pd.qcut(d.volatilidade_anual,3,labels=False).values
res={}
t=time.time(); res['K-Means']=km
res['Hierárquico (Ward)']=nomeia(AgglomerativeClustering(3,linkage='ward').fit_predict(X))
res['Misturas gaussianas']=nomeia(GaussianMixture(3,covariance_type='full',n_init=5,random_state=42).fit_predict(X))
for n,l in res.items():
    sz=np.bincount(l,minlength=3)
    print(f"{n:22s} tamanhos {sz} sil {silhouette_score(X,l):.3f} CH {calinski_harabasz_score(X,l):.0f} ARI_km {adjusted_rand_score(km,l):.3f} ARI_tercis {adjusted_rand_score(ter,l):.3f}")
    med=d.assign(g=l).groupby('g')[F].median()
    print((med*[100,100,100,1,100]).round(2).to_string())
# Ward silhouette em outros k
for k in (2,4):
    l=AgglomerativeClustering(k,linkage='ward').fit_predict(X); print('ward k',k,round(silhouette_score(X,l),3))
w=res['Hierárquico (Ward)']; print('ward mesmo perfil %.1f%%'%(100*(w==km).mean())); print(pd.crosstab(km,w))
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
d=pd.read_csv('reports_real/fundos_clusterizados.csv')
X=StandardScaler().fit_transform(d[['retorno_anual','volatilidade_anual','mdd_abs','assimetria','consistencia']])
k=np.sort(NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)[0][:,-1]); print('dist 10o vizinho p50/p90/p95',np.percentile(k,[50,90,95]).round(2))
for eps in (0.3,0.5,0.8,1.2):
    l=DBSCAN(eps=eps,min_samples=10).fit_predict(X); c=pd.Series(l).value_counts()
    print(eps,'grupos',(c.index>=0).sum(),'ruido',int((l==-1).sum()),'maiores',c[c.index>=0].head(3).tolist())
