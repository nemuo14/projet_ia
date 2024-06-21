Besoin client 2 : 

Ce besoin a été répondu par Marine BONY en A3.

Pour pouvoir prédire l'âge des arbres, vous devez : 
- lancer le fichier notebook_client_2.ipynb qui va :
  -     Prétraiter les données.
  -     Créé une database de test en format .JSON qui n'a pas été traité.
  -     Entrainer les 5 différents classifieurs.
  -     Déterminer quel classifieur est le plus performant.
  -     Effectuer des GridSearchCV pour booster les performances.
  -     Sauvegarder les différents modèles de traitements de données et le classifieurs entrainé dans un fichier 'entrainement.pkl'.
- lancer le fichier script.py qui va :
  -     Lancer la fonction 'prediction_de_age' qui va traiter les donnée à partir des modèles de traitement enregistrer 
        et prédire les âges, qui seront retournés en format .JSON.

