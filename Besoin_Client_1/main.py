import plotly.express as px
from sklearn.cluster import BisectingKMeans as BKM
from sklearn.cluster import KMeans as KM
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
from pyproj import Transformer, CRS


def carte_without_clusturing(data):
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_name="haut_tot", color='haut_tot', zoom=12, height=700) #créé une map rempli de points de data
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='haut_tot',hover_name='haut_tot') #visualisation en 3D
    fig.show()


def carte_KM(data,n):
    if 11>n>0:
        list_name = [str(i) for i in range(n)]
    else:
        print("Veuillez choisir un nombre de cluster compris entre 1 et 10 compris")
        return False
    k_means = KM(n_clusters=n, random_state=0).fit(data) #On applique le model KMeans puis on fit
    cluster_labels=k_means.labels_ #on récupère ce qu'il a fit, la valeur de chaque cluster correspondant à chaque arbre
    data_frame=pd.DataFrame(k_means.cluster_centers_, columns=['latitude', 'longitude', 'haut_tot'])#on en crée un dataFrame pour pouvoir l'utiliser
    data_frame=data_frame.sort_values(by='haut_tot') #on les trie, on aura des numéros de clusters dans le bon ordre
    cluster_names={}
    for i in range(n):
        cluster_names[data_frame.index[i]]=list_name[i]
    data['cluster'] = cluster_labels #on leur assigne leur cluster
    data['cluster_name'] = data['cluster'].map(cluster_names) #on leur donne un nom de cluster
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    silhouette_scr = silhouette_score(data, cluster_labels) #on calcul le score silhouette
    dbi_scr = davies_bouldin_score(data, cluster_labels) #on calcul le score dbi
    wcss_scr = k_means.inertia_ #on calcul la somme carré des dist entre les echantillons et le centre de son cluster

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé si proche de 0
    print("Le score Within-cluster Sum of Squares:",wcss_scr) # un score faible indique des clusters plus compacte, pas de conculsion à tirer de ce score

    data = calcul_label(data) #fonction qui va mettre dans 'cluster_name' le mode de chaque cluster, et les tries pour bien afficher les légendes sur la carte

    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.show()


def carte_BKM(data,n):
    if 11>n>0: #11 pour que chaque légende ait sa couleur, sinon on peut faire 39 clusters max
        list_name = [str(i) for i in range(n)]
    else:
        print("Veuillez choisir un nombre de cluster compris entre 1 et 10 compris") #la aussi, que pour la beauté de la légende, sinon 39 max
        return False
    bisect_means = BKM(n_clusters=n, random_state=0).fit(data) #On applique le model BisectingKMeans puis on fit
    cluster_labels=bisect_means.labels_ #on récupère ce qu'il a fit
    data_frame=pd.DataFrame(bisect_means.cluster_centers_, columns=['latitude', 'longitude', 'haut_tot']) #on en crée un dataFrame pour pouvoir l'utiliser
    data_frame=data_frame.sort_values(by='haut_tot') #on les tris, on aura des numéro de clusters dans le bon ordre
    cluster_names={}
    for i in range(n):
        cluster_names[data_frame.index[i]]=list_name[i]
    data['cluster'] = cluster_labels #on leur assigne leur cluster
    data['cluster_name'] = data['cluster'].map(cluster_names) #on leur donne un nom de cluster
    # Évaluation des clusters via métriques : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    silhouette_scr = silhouette_score(data, cluster_labels) #on calcul le score silhouette
    dbi_scr = davies_bouldin_score(data, cluster_labels) #on calcul le score dbi
    wcss_scr = bisect_means.inertia_ #on calcul la somme carré des dist entre les echantillons et le centre de son cluster

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé si proche de 0
    print("Le score Within-cluster Sum of Squares:",wcss_scr) # un score faible indique des clusters plus compacte, pas de conculsion à tirer de ce score

    data = calcul_label(data)  # fonction qui va mettre dans 'cluster_name' le mode de chaque cluster, et les tries pour bien afficher les légendes sur la carte

    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.show()


def carte_AG(data,n):
    ag = AgglomerativeClustering(n_clusters=n)
    cluster_labels = ag.fit_predict(data[['longitude', 'latitude', 'haut_tot']])#On récupère les clusters via fit_predict
    data['cluster'] = cluster_labels #on leur assigne leur cluster
    #data['bruit'] = (cluster_labels == -1) #Tous les labels qui sont assigné -1 sont des bruits, on en fait une nouvelle colonne pour les afficher plus tard
    valid_points = data[data['cluster'] != -1] #Tous les cluster sauf le bruit
    cluster_centers = valid_points.groupby('cluster')[['longitude', 'latitude', 'haut_tot']].mean().reset_index() #On calcule le centre de chaque cluster
    cluster_centers = cluster_centers.sort_values(by='haut_tot') #Grace à ça on a les clusters dans l'ordre
    cluster_names={}
    n=len(cluster_centers)
    list_name = [str(i) for i in range(n)]
    for i in range(n):
        cluster_names[cluster_centers.index[i]]=list_name[i]
    data['cluster_name'] = data['cluster'].map(cluster_names)
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    silhouette_scr = silhouette_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster']) #cette fois-ci on calcul uniquement sur les points qui ne sont pas jugés aberrants
    dbi_scr = davies_bouldin_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé

    data=calcul_label(data,valid_points) # fonction qui va mettre dans 'cluster_name' le mode de chaque cluster, et les tries pour bien afficher les légendes sur la carte
    fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    #fig.add_trace(px.scatter_mapbox(data[data['bruit']], lat="latitude", lon="longitude",hover_name="haut_tot", color_discrete_sequence=['black']).data[0]) #affiche les valeurs aberrantes en noir
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    #fig.add_trace(px.scatter_3d(data[data['bruit']], x='longitude', y='latitude', z='haut_tot', hover_name="haut_tot",color_discrete_sequence=['black']).data[0]) #affiche les valeurs aberrantes en noir
    fig.show()


def carte_DB(data,r,min_samples):
    db = DBSCAN(eps=r,min_samples=min_samples) #On applique le model DBScan
    cluster_labels=db.fit_predict(data) #On récupère les clusters via fit_predict
    data['cluster'] = cluster_labels #on leur assigne leur cluster
    data['bruit'] = (cluster_labels == -1) #Tous les labels qui sont assigné -1 sont des bruits, on en fait une nouvelle colonne pour les afficher plus tard
    valid_points = data[data['cluster'] != -1] #Tous les cluster sauf le bruit
    cluster_centers = valid_points.groupby('cluster').mean().reset_index() #On calcule le centre de chaque cluster
    cluster_centers = cluster_centers.sort_values(by='haut_tot') #Grace à ça on a les clusters dans l'ordre
    cluster_names={}
    n=len(cluster_centers)
    list_name = [str(i) for i in range(n)]
    for i in range(n):
        cluster_names[cluster_centers.index[i]]=list_name[i]
    data['cluster_name'] = data['cluster'].map(cluster_names)
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    silhouette_scr = silhouette_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster']) #cette fois-ci on calcul uniquement sur les points qui ne sont pas jugés aberrants
    dbi_scr = davies_bouldin_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé

    data=calcul_label(data,valid_points) # fonction qui va mettre dans 'cluster_name' le mode de chaque cluster, et les tries pour bien afficher les légendes sur la carte

    # fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    # fig.add_trace(px.scatter_mapbox(data[data['bruit']], lat="latitude", lon="longitude",hover_name="haut_tot", color_discrete_sequence=['black']).data[0]) #affiche les valeurs aberrantes en noir
    # fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.add_trace(px.scatter_3d(data[data['bruit']], x='longitude', y='latitude', z='haut_tot', hover_name="haut_tot",color_discrete_sequence=['black']).data[0]) #affiche les valeurs aberrantes en noir
    fig.show()


def carte_HDB(data,size,samples):
    hdb = HDBSCAN(min_cluster_size=size,min_samples=samples) #On applique le model HDBScan
    cluster_labels=hdb.fit_predict(data) #On récupère les clusters via fit_predict
    data['cluster'] = cluster_labels
    data['bruit'] = (cluster_labels == -1)
    cluster_centers = data[data['cluster'] != -1].groupby('cluster').mean().reset_index()
    cluster_names={}
    list_name = [str(i) for i in range(len(cluster_centers))]
    for i in range(len(cluster_centers)):
        cluster_names[cluster_centers.index[i]]=list_name[i]

    data['cluster_name'] = data['cluster'].map(cluster_names)
    # Évaluation des clusters : #######################Ne marche que si les cluster name sont des int, pas pour petit ect
    valid_points = data[data['cluster'] != -1]
    silhouette_scr = silhouette_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])
    dbi_scr = davies_bouldin_score(valid_points[['longitude', 'latitude', 'haut_tot']], valid_points['cluster'])

    print("Le Silhouette Score:",silhouette_scr) #entre -1 et 1, proche de 1 => les arbres sont bien regroupés au sein de leur cluster
    print("Le score Davies-Bouldin Index:", dbi_scr) #cluster bien formé si proche de 0

    data=calcul_label(data,valid_points) # fonction qui va mettre dans 'cluster_name' le mode de chaque cluster, et les tries pour bien afficher les légendes sur la carte

    # fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", hover_data="haut_tot",color='cluster_name', zoom=12, height=700)
    # fig.add_trace(px.scatter_mapbox(data[data['bruit']], lat="latitude", lon="longitude",hover_name="haut_tot", color_discrete_sequence=['black']).data[0])
    # fig.update_layout(mapbox_style="open-street-map")
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.show()

    fig = px.scatter_3d(data, x='longitude', y='latitude', z='haut_tot', color='cluster_name',hover_name='haut_tot')
    fig.add_trace(px.scatter_3d(data[data['bruit']], x='longitude', y='latitude', z='haut_tot', hover_name="haut_tot",color_discrete_sequence=['black']).data[0])
    fig.show()


def calcul_label(data,valid_points=pd.DataFrame()): #On va calculer le mode de chaque cluster
    cluster_list = []
    min_heights = []
    max_heights = []
    if (valid_points.empty): #cas où on ne fais pas de clustering Scan
        valid_points=data
    grouped = valid_points.groupby('cluster') #On les regoupes par cluster
    for cluster, group in grouped:
        cluster_list.append(cluster)
        min_heights.append(group['haut_tot'].min()) #On ajoute dans la liste des max, le max de chaque cluster
        max_heights.append(group['haut_tot'].max()) #pareil pour les min

    cluster_ranges = pd.DataFrame({
        'cluster': cluster_list,
        'min': min_heights,
        'max': max_heights
    })
    cluster_names = {}
    for i, row in cluster_ranges.iterrows(): #on va assigner dans le DataFrame les max et min
        cluster = row['cluster']
        min_height = row['min']
        max_height = row['max']
        if max_height==min_height:
            cluster_names[cluster] = f"{max_height}"  # le format du label
        else:
            cluster_names[cluster] = f"{min_height}-{max_height}" #le format du label

    data['cluster_name'] = data['cluster'].map(cluster_names) #on finit par mapper les mode correspondant

    # data['cluster_name_numeric'] = pd.to_numeric(data['cluster_name'], errors='coerce') #colone temporaire pour trier le data en fonction de son cluster
    # data = data.sort_values(by='cluster_name_numeric').drop(columns=['cluster_name_numeric']) #ça a pour but de mettre la légende de forme croissante, plus lisible
    data=data.sort_values(by='haut_tot')
    return data


def question(): #fonction qui demande à l'utilisateur ce qu'il veut faire
    clustering=3
    answer = 3
    precision = 3
    cols=3
    set_data = 3
    while (set_data!=0 and set_data!=1):
        set_data=int(input("Voulez-vous utiliser notre dataset (tapez 0), ou celle des professeurs (tapez 1)?"))
        if set_data != "":
            set_data = int(set_data)
    while (clustering!=0 and clustering!=1):
        clustering=input("Voulez-vous simplement visualiser les arbres (tapez 0), ou faire du clustering (tapez 1)?")
        if clustering!="":
            clustering=int(clustering)
    if clustering==0:
        return set_data,clustering,0,0
    else:
        while (answer!=0 and answer!=1):
            answer=input("Voulez-vous faire un clustering avec tous les arbres (tapez 0), ou du clustering en enlevant les valeurs aberrantes ? (tapez 1)")
            if answer != "":
                answer = int(answer)
        if answer==0:
            while (precision != 0 and precision != 1 and precision!=2):
                precision = input("Voulez-vous utiliser du Kmeans clustering (tapez 0), du Bisecting K-Means clustering ? (tapez 1) ou du Agglomerative Clustering (tapez 2)")
                if precision != "":
                    precision = int(precision)
        else:
            while (precision != 0 and precision != 1):
                precision = input("Voulez-vous utiliser du DBScan clustering (tapez 0), ou du HDBScan clustering ? (tapez 1)")
                if precision != "":
                    precision = int(precision)
            while (cols != 0 and cols != 1):
                cols = input("Voulez-vous utiliser 1 variable (tapez 0), ou 3 variables ? (tapez 1)")
                if cols != "":
                    cols = int(cols)
        return set_data,clustering,answer,precision,cols


def data_traitement(set_data):
    print(set_data)
    if set_data==1 or set_data==3:
        data = pd.read_csv('Besoin_Client_1/Data_Arbre.csv')
        if set_data==3:
            print("oui je passe (=3)")
            cols = ["longitude", "latitude", "haut_tot","tronc_diam","haut_tronc"]
        else:
            cols = ["longitude", "latitude", "haut_tot"]
        new_data = data[cols]
        return new_data
    else:
        data=pd.read_csv('Besoin_Client_1/data_exported.csv')
        if set_data==0:
            cols = ["X", "Y", "haut_tot"]
        else:
            print("oui je passe (=2)")
            cols = ["X", "Y", "haut_tot","tronc_diam","haut_tronc"]
        new_data = data[cols]
        crs_source = CRS.from_epsg(3949) #code source
        crs_target = CRS.from_epsg(4326)  #code recherché : EPSG:4326 pour WGS84
        transformer = Transformer.from_crs(crs_source,crs_target) #On applique la transformation
        new_data['latitude'], new_data['longitude'] = transformer.transform(new_data['X'].values, new_data['Y'].values) #on créé deux autres colones avec les vraies coordonnées
        new_data=new_data.drop(columns=['X'])
        new_data = new_data.drop(columns=['Y'])
        return new_data



def main():
    set_data,clus, ans, prec, cols= question() # on récupère les réponses aux questions
    if cols: #si on prend plusieurs variables
        set_data +=2
    data=data_traitement(set_data) #On prend la dataset demandée
    if clus:#clustering ou non
        if ans: #quel type de clustering
            if prec: #quel model de clustering
                if cols: #plusieurs variables ou non
                    carte_HDB(data,14,1) #cluster_size, min_samples ; exemple : 10,1 #on ne retient pas ce model, base de donnée pas adaptée : 14,1
                else:
                    carte_HDB(data, 10, 1)
            else:
                if cols:
                    carte_DB(data, 2,20)  # rayon, min_samples ; exemple : 2,10 ; 1.5,1 ; 1.8,5 ;  |2,20 avec data prof|
                else:

                    carte_DB(data, 0.6,20) # meilleur score : 0.2,20 ; joli clustering : 0.6,20
        else:
            if prec==1:
                carte_BKM(data,3) #nombre de cluster à choisir
            elif prec==0:
                carte_KM(data,3) #nombre de cluster à choisir
            else:
                carte_AG(data,3) #nombre de cluster à choisir
    else:
        carte_without_clusturing(data)
    return 0


main()




