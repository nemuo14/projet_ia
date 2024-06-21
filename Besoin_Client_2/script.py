import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pickle
import json


def prediction_de_age(json_input):
    """
    focntion qui va utiliser une dataframe JSON en entré, traiter les données et faire des prédictions en fonction de variables
    pour en déterminer l'âge
    :param json_input:
    :return: le dataframe des prédictions de l'âge en formoat JSON
    """
    data = pd.read_json(json_input)   #on ouvre la dataframe d'entrée
    print(data['age_estim'][:5])

    # on charge le modèle et les modèles de prétraitements
    with open('Besoin_Client_2/enregistrement.pkl', 'rb') as f:
        enregistrement = pickle.load(f)

    # on sort nos modèles qu'on met dans des variables
    encoding = enregistrement['pretraitement']['encoding']
    scaler_X = enregistrement['scaler_X']
    scaler_Y = enregistrement['scaler_Y']
    model = enregistrement['models']["RandomForestRegressor"]

    # on sort nos colonnes utilent
    haut_tot = data[["haut_tot"]]
    haut_tronc = data[["haut_tronc"]]
    diam_tronc = data[["tronc_diam"]]
    age_estim = data[["age_estim"]]
    stade_dev = data[["fk_stadedev"]]


    # concaténation des caractéristiques dans un seul DataFrame et on traite nos données
    X = pd.concat([haut_tot, haut_tronc, diam_tronc, stade_dev], axis=1)
    X['fk_stadedev'] = encoding.transform(stade_dev)
    Y = age_estim

    X_scaled = pd.DataFrame(scaler_X.transform(X), columns=X.columns)
    Y = scaler_Y.transform(Y)

    # on prédit l'âge
    prediction_scaled = model.predict(X_scaled)


    accuracy = r2_score(Y, prediction_scaled)
    print("accuracy random forest: ", accuracy)

    # on enlève le scaling pour obtenir les prédictions de l'âge dans la bonne échelle
    prediction = scaler_Y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    #on met nos prédictions dans un dataframe qu'on transforme en format json
    new_prediction = pd.DataFrame(prediction, columns=['age_estim']).to_json('Besoin_Client_2/prediction_age.json')
    print(prediction[:5])

    return new_prediction

predictions = prediction_de_age("Besoin_Client_2/df_test.json")
print(predictions)
