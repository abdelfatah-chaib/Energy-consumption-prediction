# Data-Analysis-Project
- Projet de Fin de Semestre (S5) en Module d' Analyse de Données et Classification
- Titre du projet :  la consommation énergétique des appareils électroménagers 

•	Membre de groupe :
1.	DEMRAOUI Salma
2.	CHAIB Abdelfatah
3.	AIT AISSA Rachid

•	Description générale :
Ce projet vise à prédire la consommation énergétique des appareils électroménagers en s'appuyant sur diverses données environnementales, telles que la température, l'humidité, les vibrations, la lumière et le bruit. L'objectif principal est de réduire la consommation d'énergie et les émissions de carbone associées, contribuant ainsi à une gestion énergétique plus efficace.
Pour collecter ces données, des capteurs et appareils intelligents sont intégrés dans le système. Par exemple :
	Capteur de température et d'humidité (DHT22) : Ce capteur, précis et facile à utiliser, mesure les conditions environnementales comme la température et l'humidité. Ces données sont cruciales pour comprendre leur influence sur la consommation énergétique des appareils.

•	Problématique :
Avec la hausse constante de la consommation d'énergie, les émissions de carbone et de gaz à effet de serre continuent d'augmenter, représentant une menace pour l'environnement. Bien que le secteur industriel soit le principal consommateur d'énergie, le secteur résidentiel joue également un rôle significatif. Les appareils électroménagers représentent une part importante de la consommation d'énergie dans les résidences. Comprendre et prévoir les comportements de consommation énergétique dans ce contexte est essentiel pour identifier des opportunités de réduction d'énergie et d'émissions.

•	Objectif du projet :
Le projet a pour objectif d’analyser ; interpréter et prédire la consommation énergétique des appareils électroménagers d'une maison. Les données utilisées incluent des variables continues comme :
	Température ;  Humidité ;  Pression ;  Vibrations ;  Lumière ;  Bruit

- Technologies Utilisées
  
•	Python
•	Pandas, NumPy, Matplotlib, Seaborn
•	Scikit-learn (Regression Linéaire, Random Forest, Extra Trees)
•	XGBoost
•	Streamlit (pour la visualisation interactive)

-Structure du Projet

1.	Exploration des Données
•	Chargement et nettoyage des données
•	Visualisation des corrélations
•	Analyse des tendances temporelles

2.	Prétraitement des Données
•	Gestion des valeurs manquantes
•	Normalisation et standardisation des variables
•	Sélection des features pertinentes (ACP)

3.	Modélisation
•	Application de plusieurs modèles de régression : 
	Regression Linéaire
	Random Forest Regressor
	Extra Trees Regressor
	XGBoost
•	Optimisation des hyperparamètres (GridSearchCV)

4.	Évaluation des Modèles
•	Calcul des métriques : 
	Mean Squared Error (MSE)
	Mean Absolute Error (MAE)
	R² Score
•	Comparaison des performances des modèles

5.	Optimisation du Modèle
•	Réduction du sur-apprentissage
•	Amélioration des hyperparamètres

6.	Visualisation des Résultats
•	Graphiques de performance
•	Comparaison des modèles
•	Tableau des prédictions


Résultats Obtenus

•	XGBoost a donné les meilleures performances avec un R² de 0.67.
•	Extra Trees Regressor a été l'un des modèles les plus performants avec un R² de 0.72.
•	L'optimisation des features a permis d'améliorer les prédictions en supprimant les variables inutiles.

Améliorations Futures
•	Test de modèles plus avancés
•	Intégration de nouvelles variables pour affiner les prédictions


