import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as plo
import pickle
import json

def evaluation_modele(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    matrice = confusion_matrix(y_true, y_pred)
    print("confusion matrix:", matrice)
    print("accuracy:", accuracy)
    print("f1-score:", f1)


    matrice = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=np.unique(y_true))
    disp.plot()


def traitement_donnees(fichier_json):
    with open(fichier_json, 'r') as fichier:
        data_json = fichier.read()
    data = pd.read_json(data_json)
    data2 = data.copy()
    new_data = ["haut_tot", "haut_tronc", "tronc_diam", "age_estim",
                "fk_stadedev", "fk_port", "fk_pied", "fk_situation", "fk_revetement",
                "longitude", 'latitude']

    donnes_qualitatives = ["fk_stadedev", "fk_port", "fk_pied", "fk_situation", "fk_revetement"]
    donnes_quantitatives = ["haut_tot", "haut_tronc", "tronc_diam", "age_estim"]

    encoding = OrdinalEncoder()
    data2[donnes_qualitatives] = encoding.fit_transform(data[donnes_qualitatives])

    std = StandardScaler()
    data2[donnes_quantitatives] = std.fit_transform(data[donnes_quantitatives])

    X = pd.DataFrame(data[donnes_quantitatives])
    y = data["fk_arb_etat"]


    return data2[new_data],X,y


def predictions_risques(data):
    models = chargement_modele()
    data_traitees = traitement_donnees(data)
    predictions = models.predict(data_traitees)
    return predictions

def chargement_modele():
    with open(r"modèle.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def carte(data,pred):
    data2= data.copy()
    data2['predictions'] = pred
    figure = px.scatter_mapbox(data2,lon="longitude",
                               lat="latitude",
                               zoom=10,
                               mapbox_style="open-street-map",
                               color='fk_arb_etat'
    )
    figure.update_layout(mapbox_style="open-street-map")

    plo.plot(figure,filename="map.html")


def carte_prob(data, prob_pred):
    data2 = data.copy()
    data2['probabilite'] = prob_pred[:, 1]
    figure = px.scatter_mapbox(data2, lon="longitude", lat="latitude", zoom=10,
                               mapbox_style="open-street-map", color='probabilite',
                               color_continuous_scale=px.colors.sequential.Viridis)
    figure.update_layout(mapbox_style="open-street-map")
    plo.plot(figure, filename="map_prob.html")





def main(fichier_json):

    data,X,y = traitement_donnees(fichier_json)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=2)
    modele = chargement_modele()
    y_pred= modele.predict(X_test)
    evaluation_modele(y_test, y_pred)
    predictions = predictions_risques(data)
    carte(data, predictions)
    carte_prob(data,predictions)


main(r"data_json")
