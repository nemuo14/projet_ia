import plotly.express as px
from sklearn.cluster import BisectingKMeans as BKM
from sklearn.cluster import KMeans as KM
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
import pandas as pd




def carte_without_clusturing(data):
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_name="haut_tot", color='haut_tot', zoom=12, height=700)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='haut_tot',hover_name='haut_tot')
    fig.show()

def carte_BKM(data,n):
    if 11>n>0:
        list_name = [str(i) for i in range(n)]
    # if n==1:
    #     list_name = ["arbres"]
    # elif n==2:
    #     list_name = ["petit", "grand"]
    # elif n==3:
    #     list_name = ["petit", "moyen", "grand"]
    # elif n==4:
    #     list_name = ["petit", "moyen-","moyen+", "grand"]
    # elif n==5:
    #     list_name = ["minuscule","petit", "moyen", "grand","immense"]
    # elif 11>n>0:
    #     list_name = [str(i) for i in range(n)]
    else:
        print("Veuillez choisir un nombre de cluster compris entre 1 et 10 compris")
        return False
    bisect_means = BKM(n_clusters=n, random_state=0).fit(data)
    cluster_labels=bisect_means.labels_
    data_frame=pd.DataFrame(bisect_means.cluster_centers_, columns=['latitude', 'longitude', 'haut_tot'])
    data_frame=data_frame.sort_values(by='haut_tot')
    cluster_names={}
    for i in range(n):
        cluster_names[data_frame.index[i]]=list_name[i]
    print(cluster_names)
    data['cluster'] = cluster_labels
    data['cluster_name'] = data['cluster'].map(cluster_names)
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    # for i in range(n):
    #     cluster_names[data_frame.index[i]]=i
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    silhouette_scr = silhouette_score(data, cluster_labels)
    dbi_scr = davies_bouldin_score(data, cluster_labels)
    wcss_scr = bisect_means.inertia_

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé
    print("Le score Within-cluster Sum of Squares:",wcss_scr) # un score faible indique des clusters plus compacte, pas de conculsion à tirer de ce score

    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.show()

def carte_KM(data,n):
    if 11>n>0:
        list_name = [str(i) for i in range(n)]
    # if n==1:
    #     list_name = ["arbres"]
    # elif n==2:
    #     list_name = ["petit", "grand"]
    # elif n==3:
    #     list_name = ["petit", "moyen", "grand"]
    # elif n==4:
    #     list_name = ["petit", "moyen-","moyen+", "grand"]
    # elif n==5:
    #     list_name = ["minuscule","petit", "moyen", "grand","immense"]
    # elif 11>n>0:
    #     list_name = [str(i) for i in range(n)]
    else:
        print("Veuillez choisir un nombre de cluster compris entre 1 et 10 compris")
        return False
    k_means = KM(n_clusters=n, random_state=0).fit(data)
    cluster_labels=k_means.labels_
    data_frame=pd.DataFrame(k_means.cluster_centers_, columns=['latitude', 'longitude', 'haut_tot'])
    data_frame=data_frame.sort_values(by='haut_tot')
    cluster_names={}
    for i in range(n):
        cluster_names[data_frame.index[i]]=list_name[i]
    print(cluster_names)
    data['cluster'] = cluster_labels
    data['cluster_name'] = data['cluster'].map(cluster_names)
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    # for i in range(n):
    #     cluster_names[data_frame.index[i]]=i
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    silhouette_scr = silhouette_score(data, cluster_labels)
    dbi_scr = davies_bouldin_score(data, cluster_labels)
    wcss_scr = k_means.inertia_

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé
    print("Le score Within-cluster Sum of Squares:",wcss_scr) # un score faible indique des clusters plus compacte, pas de conculsion à tirer de ce score

    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.show()

def carte_HDB(data,size,samples):
    # if n==1:
    #     list_name = ["arbres"]
    # elif n==2:
    #     list_name = ["petit", "grand"]
    # elif n==3:
    #     list_name = ["petit", "moyen", "grand"]
    # elif n==4:
    #     list_name = ["petit", "moyen-","moyen+", "grand"]
    # elif n==5:
    #     list_name = ["minuscule","petit", "moyen", "grand","immense"]
    # elif 11>n>0:
    #     list_name = [str(i) for i in range(n)]
    hdb = HDBSCAN(min_cluster_size=size,min_samples=samples)
    cluster_labels=hdb.fit_predict(data)
    data['cluster'] = cluster_labels
    data['bruit'] = (cluster_labels == -1)
    print(cluster_labels)
    # data_frame=pd.DataFrame(cluster_labels, columns=['latitude', 'longitude', 'haut_tot'])
    # data_frame=data_frame.sort_values(by='haut_tot')
    cluster_centers = data[data['cluster'] != -1].groupby('cluster').mean().reset_index()
    print(cluster_centers)
    cluster_names={}
    list_name = [str(i) for i in range(len(cluster_centers))]
    for i in range(len(cluster_centers)):
        cluster_names[cluster_centers.index[i]]=list_name[i]
    print(cluster_names)

    data['cluster_name'] = data['cluster'].map(cluster_names)
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    # for i in range(n):
    #     cluster_names[data_frame.index[i]]=i
    valid_points = data[data['cluster'] != -1]
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    silhouette_scr = silhouette_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])
    dbi_scr = davies_bouldin_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé

    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    fig.add_trace(px.scatter_mapbox(data[data['bruit']], lat="latitude", lon="longitude",hover_name="haut_tot", color_discrete_sequence=['black']).data[0])
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.add_trace(px.scatter_3d(data[data['bruit']], x='longitude', y='latitude', z='haut_tot', hover_name="haut_tot",color_discrete_sequence=['black']).data[0])
    fig.show()

def carte_DB(data,r,min_samples):
    db = DBSCAN(eps=r,min_samples=min_samples)
    cluster_labels=db.fit_predict(data)
    data['cluster'] = cluster_labels
    data['bruit'] = (cluster_labels == -1)
    print("atttttttttttt",cluster_labels)
    valid_points = data[data['cluster'] != -1]
    print("les dataaaaa",data['cluster'])
    print("les valid points",valid_points)
    cluster_centers = valid_points.groupby('cluster').mean().reset_index()
    cluster_centers = cluster_centers.sort_values(by='haut_tot')
    print(cluster_centers)
    cluster_names={}
    n=len(cluster_centers)
    print("le n : ", n)
    list_name = [str(i) for i in range(n)]
    for i in range(n):
        cluster_names[cluster_centers.index[i]]=list_name[i]
    print(cluster_names)
    # data_frame=pd.DataFrame(cluster_labels, columns=['latitude', 'longitude', 'haut_tot'])
    # data_frame=data_frame.sort_values(by='haut_tot')

    data['cluster_name'] = data['cluster'].map(cluster_names)
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    # for i in range(n):
    #     cluster_names[data_frame.index[i]]=i
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    silhouette_scr = silhouette_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])
    dbi_scr = davies_bouldin_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé

    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    fig.add_trace(px.scatter_mapbox(data[data['bruit']], lat="latitude", lon="longitude",hover_name="haut_tot", color_discrete_sequence=['black']).data[0])
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    print("les légendes",data['cluster_name'])
    data=data.sort_values(by='cluster_name')
    print("les légendes", data['cluster_name'])
    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.add_trace(px.scatter_3d(data[data['bruit']], x='longitude', y='latitude', z='haut_tot', hover_name="haut_tot",color_discrete_sequence=['black']).data[0])
    fig.show()

def main():
    data = pd.read_csv('Data_Arbre.csv')
    cols=["longitude","latitude","haut_tot"]
    new_data=data[cols]
    #carte_DB(new_data, 0.2,20)  # rayon, min_samples , exemple : 0.2 et 20
    #carte_HDB(new_data,10,1) #cluster_size, min_samples , exemple : 10,1 #on ne retient pas ce model, base de donnée pas adaptée
    #carte_BKM(new_data,3)
    #carte_KM(new_data,10)
    #carte_without_clusturing(new_data)
    return 0


main()




