import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ds = pd.read_csv('data_SS.csv')
#ds -> original dataset 
df = pd.DataFrame(columns = ["mean", "number of reviews", "number of no reviews"])
#df -> dataset for primary analysis
i=1

"""This while loop does the 1st analysis on averaging again the reviews and counting the amt of reviews"""
while i < 25:
    num_of_review = 0
    num_not_review = 0
    mean = 0

    num_of_review = ds[ds[f"Category {i}"] > 0].count(axis="index").loc['User']
    #before .count -> filters out rows with 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    num_not_review = ds[ds[f"Category {i}"] == 0].count(axis="index").loc['User']
    #before .count -> filters to only get rows with 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    mean = ds[f"Category {i}"].sum() / num_of_review
    # .sum -> adds up the row (don't have to filter out 0 cause adding 0 doesn't change anything)

    df.loc[len(df)] = {"mean" : mean, "number of reviews" : num_of_review, "number of no reviews" : num_not_review}
    #add to the new dataset the mean, number of users who review that type of attraction, number of users who don't review that type of attraction

    i +=1
    
#print('\nResult Dataset:\n', df)

#for clustering: find the optimal amount of f-means for clustering, staderdized data (unnecessary because already standerdized),
# do PCA becasue high dimensionality and avoid the curese of dimensionality, plot the data?/create processed csv data for Sher to plot
#add to the original dataset which cluster each ID belong to -> filter through each ID in another loop and do another cycle of primary analysis 

"""Clustering the Users"""

dn = ds
dn = dn.drop(axis = 1, columns = "User")
dn = dn.drop(index = 2714)
dn = dn.drop(axis = 1, columns = "Unnamed: 25")
dn.dropna(inplace = True)
##had to remove any data with empty (NaN/NaT) values to be able to do ".fit" and ".fit_transform"
#TODO see what other 2 indexes that got dropped (3 was dropped, index 2714 by me), find other 2 with "dn.isna" and remove all other rows

scaler = StandardScaler()
std_data = scaler.fit_transform(dn)
#standardzed the data, will test if changes anything

pca = PCA(n_components = 2) #first test standardized data after PCA to see how many clusters we want, after testing we plug in how many in ()
pca.fit(std_data)

#print(pca.explained_variance_ratio_) 
#tells me only 2 clusters

scores_pca = pca.transform(dn.values)
#added to our dataset later to graph the clusters

'''
wcss = []
for j in range(1,5):
    kmeans_pca = KMeans(n_clusters = i, init = "k-means++", random_state=1)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize = (10,8))
plt.plot(range(1,5), wcss, marker = "o", linestyle = "--")
plt.show()
'''
#graphing for elbow method to find right number of clusters doesn't work because only 2 clusters


kmeans_pca_labels = KMeans(n_clusters = 2, init = "k-means++", random_state=1).fit(scores_pca).labels_ #2 clusters is best confirmed again
print(calinski_harabasz_score(dn.values, kmeans_pca_labels)) #want as high as possible because higher is denser and more separated
print(davies_bouldin_score(dn.values, kmeans_pca_labels)) #low as possible to be more denser and more seperate
print(silhouette_score(dn.values, kmeans_pca_labels)) #as close to 1 as possible to be more seperate from other cluster
                                                        #if close to 0 (as it is), means there is bordering and even some overlapping clusters
                                                        #which makes sense in this context of reviews
#these are three different tests that tells how good the chosen amount of clusters are

dn = dn.join(pd.DataFrame(scores_pca), how = "right")
dn["Segment K-means PCA"] = kmeans_pca_labels

"""dn.to_csv("cluster.csv")"""
#DO NOT GRAB THIS IT WAS FOR A TEST
#convert new table to csv to check if formatting is correct
#column 0 is segment 1 which should be y-axis of graph
#column 1 is segment 2 which should be the x-axis of graph
#column "Segment K-means PCA" is the cluter type (0 or 1)
#TODO check if there is basically only one cluster when plotting *!*!**!*!*!*!*!*!*!* when in tableau with the 2 at the bottom
    #graph together with same color and see? if different enough then I am happy if not then we can talk about it in presentation!

"""Secondary Analysis"""

d0 = pd.DataFrame(columns = ["mean", "number of reviews", "number of no reviews"])
d1 = pd.DataFrame(columns = ["mean", "number of reviews", "number of no reviews"])
#where d0 is the primary analysis but for only cluster group 0 and d1 for cluster group 1

i = 1
#for d0
while i < 25:
    num_of_review = 0
    num_not_review = 0
    mean = 0

    num_of_review = dn[(dn[f"Category {i}"] > 0) & dn["Segment K-means PCA"] == 0].count(axis="index").loc['Category 1']
    #before .count -> filters out rows with 0 AND the cluster type has to equal 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    num_not_review = dn[(dn[f"Category {i}"] == 0) & dn["Segment K-means PCA"] == 0].count(axis="index").loc['Category 1']
    #before .count -> filters to only get rows with 0 AND the cluster type has to equal 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    if ((dn["Segment K-means PCA"]).eq(0)):
        mean = dn[f"Category {i}"].sum() / num_of_review
        #if -> check if in cluster group 0
        #  .sum -> adds up the row (don't have to filter out 0 cause adding 0 doesn't change anything)

    d0.loc[len(df)] = {"mean" : mean, "number of reviews" : num_of_review, "number of no reviews" : num_not_review}
    #add to the new dataset the mean, number of users who review that type of attraction, number of users who don't review that type of attraction

    i +=1

i = 1
#for d1
while i < 25:
    num_of_review = 0
    num_not_review = 0
    mean = 0

    num_of_review = dn[(dn[f"Category {i}"] > 0) & dn["Segment K-means PCA"] == 1].count(axis="index").loc['Category 1']
    #before .count -> filters out rows with 0 AND the cluster type has to equal 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    num_not_review = dn[(dn[f"Category {i}"] == 0) & dn["Segment K-means PCA"] == 1].count(axis="index").loc['Category 1']
    #before .count -> filters to only get rows with 0 AND the cluster type has to equal 0
    #after .count -> counts the number of rows **AS A SERIES**
    #.loc -> pulls a specific number from the series so fixes a major issue
    if ((dn["Segment K-means PCA"]).eq(1)):
        mean = dn[f"Category {i}"].sum() / num_of_review
        #if -> check if in cluster group 0
        #  .sum -> adds up the row (don't have to filter out 0 cause adding 0 doesn't change anything)
    
    d1.loc[len(df)] = {"mean" : mean, "number of reviews" : num_of_review, "number of no reviews" : num_not_review}
    #add to the new dataset the mean, number of users who review that type of attraction, number of users who don't review that type of attraction

    i +=1

d0.to_csv("cluster_0.csv")
d1.to_csv("cluster_1.csv")