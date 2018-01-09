from Scripts.Tools.IO import IOTools
from sklearn.cluster.hierarchical import AgglomerativeClustering
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score, completeness_score, homogeneity_score, classification_report

iotools = IOTools()

dataset = iotools.LoadPickle('F:\My_Pro\Python\Jobs2\Scripts\Clustering\ProfDemands\ProfDemands&Classes.p')

model = AgglomerativeClustering(n_clusters=10, affinity='cosine', linkage='complete')#, affinity='cosine', linkage='average')
#model = AgglomerativeClustering(n_clusters=10, affinity='l2', linkage='average')#, affinity='cosine', linkage='average')

predsTfidf = model.fit_predict(dataset['demands'].tolist())
print(predsTfidf)

#print(predsTfidf)
print('v_measure '+str(v_measure_score(dataset['classes'],predsTfidf)))
print('completeness '+str(completeness_score(dataset['classes'],predsTfidf)))
print('homogenity '+str(homogeneity_score(dataset['classes'],predsTfidf)))
#print("")
#print(classification_report(dataset['classes'],predsTfidf))