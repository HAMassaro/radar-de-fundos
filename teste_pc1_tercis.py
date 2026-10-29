import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
d=pd.read_csv('reports_real/fundos_clusterizados.csv')
F=['retorno_anual','volatilidade_anual','mdd_abs','assimetria','consistencia']
X=StandardScaler().fit_transform(d[F]); km=d.perfil.map({'Conservador':0,'Moderado':1,'Arrojado':2}).values
pc=PCA(2,random_state=42).fit(X); s=pc.transform(X)[:,0]
if np.corrcoef(s,d.volatilidade_anual)[0,1]<0: s=-s
ter=np.asarray(pd.qcut(s,3,labels=False))
print('PC1 tercis: sil %.3f  ARI_km %.3f  mesmo %.1f%%'%(silhouette_score(X,ter),adjusted_rand_score(km,ter),100*(ter==km).mean()))
print(pd.crosstab(km,ter))
med=d.assign(g=ter).groupby('g')[F].median(); print((med*[100,100,100,1,100]).round(2))
# fundos do Moderado do K-Means no tercil baixo da PC1
x=d[(km==1)&(ter==0)]; print('Mod KM em PC1 baixo',len(x))
