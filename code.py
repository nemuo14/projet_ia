import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Besoin_Client_1/Data_Arbre.csv')

#print(data["clc_quartier"]) #data est un dictionnaire / un data frame


#1) on sélectionne nos colonnes utiles :
haut_tot = data[["haut_tot"]]
haut_tronc = data[["haut_tronc"]]
diam_tronc = data[["tronc_diam"]]
age_estim = data[["age_estim"]]


#2) on encode nos données :
encoding = OrdinalEncoder()
enc_haut_tot = encoding.fit_transform(haut_tot)
enc_haut_tronc = encoding.fit_transform(haut_tronc)
enc_diam_tronc = encoding.fit_transform(diam_tronc)
enc_age_estim = encoding.fit_transform(age_estim)

# print(enc_age_estim)
print(enc_haut_tronc)
# print(enc_haut_tot)
# print(enc_diam_tronc)

#on les mets dans un data frame :
# new_data = {"haut_tot" : data[["haut_tot"]], "haut_tronc" : data[["haut_tronc"]], "diam_tronc" : data[["tronc_diam"]], "age_estim" : data[["age_estim"]]}
#print(new_data)


