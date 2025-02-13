import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Configuration de la page
st.set_page_config(page_title="Projet Analyse de Données", layout="wide")

# Charger les données
data = pd.read_csv("energydata.csv")

# Sélection des variables de température et d'humidité
temp_cols = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']  # Températures
humidity_cols = ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_6', 'RH_7', 'RH_8', 'RH_9']  # Humidité
external_cols = ['Press_mm_hg', 'T_out', 'RH_out']  # Conditions extérieures

# Navigation via la barre latérale
st.sidebar.image("images\logo-UIT.png", width=250)
st.sidebar.title("Analyse de Données et Classification")
# Contenu pour chaque page  
menu = st.sidebar.radio("", ["ACCUEIL","EXPLORATION", "VISUALISATION", "PRE-TRAITEMENT", "REGRESSION"])
# Pied de page dans la barre latérale
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 - IAID - FSK - UIT")

# Contenu de la page d'accueil
if menu == "ACCUEIL": 
        st.title("Bienvenue dans notre projet !")
        st.header("Sujet : La consommation énergétique des appareils électroménagers")
        # Informations principales
        st.markdown("""<h3 style="color:#2F4F4F">Membres du groupe :</h3>""", unsafe_allow_html=True)
        st.markdown("""
        1. **DEMRAOUI Salma**  
        2. **CHAIB Abdelfatah**  
        3. **AIT AISSA Rachid**
        """)
        st.markdown("""<h3 style="color:#2F4F4F">Description générale :</h3>""", unsafe_allow_html=True)
        st.markdown("""
        Ce projet vise à prédire la consommation énergétique des appareils électroménagers en s'appuyant sur diverses
        données environnementales, telles que la température, l'humidité, la pression, les vibrations, la lumière et le bruit.\n
        L'objectif principal est de réduire la consommation d'énergie et les émissions de carbone associées,
        contribuant ainsi à une gestion énergétique plus efficace.
        Pour collecter ces données, des capteurs et appareils intelligents sont intégrés dans le système. 
        """)
        st.markdown("""<h3 style="color:#2F4F4F">Problématique :</h3>""", unsafe_allow_html=True)
        st.markdown("""
        Avec la hausse constante de la consommation d'énergie, les émissions de carbone et de gaz à effet de serre continuent d'augmenter, représentant une menace pour l'environnement.  
        Bien que le secteur industriel soit le principal consommateur d'énergie, le secteur résidentiel joue également un rôle significatif. Les appareils électroménagers représentent une part importante de la consommation d'énergie dans les résidences.  
        Comprendre et prévoir les comportements de consommation énergétique est essentiel pour identifier des opportunités de réduction d'énergie et d'émissions.
        """)
        st.markdown("""<h3 style="color:#2F4F4F">Les outils utilisés :</h3>""", unsafe_allow_html=True)
        st.markdown("""
        Dans ce projet d'analyse de données, plusieurs outils et bibliothèques du Python ont été utilisés pour traiter, analyser et visualiser les informations.    
        - **Pandas** pour la manipulation des données
        - **NumPy** pour les calculs numériques
        - **Matplotlib** et **Seaborn** pour la visualisation
        - **Scikit-Learn** pour les algorithmes d'apprentissage automatique
        - **Streamlit** pour créer rapidement l'applications web interactive \n
        Ces outils, combinés, ont facilité une analyse approfondie et structurée des données.
        """)

#Contenu de la page Exploration
if menu == "EXPLORATION":
    st.title("Exploration des Données")
    st.markdown("Cette section présentera une analyse exploratoire des données pour comprendre la distribution, les statistiques descriptives et les relations entre les variables.")
    # Afficher les premières lignes du dataset
    st.subheader("Aperçu du Jeu de Données")
    nombre = st.number_input("Entrez le nombre de lignes à afficher :", min_value=1, max_value=len(data), value=5)
    # Bouton pour afficher les données
    if st.button("Afficher"):
      st.write(data.head(nombre))
    # Ajouter un filtre pour visualiser les données par colonne
    st.subheader("Filtrer par colonnes")
    selected_columns = st.multiselect("Choisissez les colonnes à afficher", options=data.columns)
    st.write(data[selected_columns].head(10) if selected_columns else "Aucune colonne sélectionnée.")

    # Affichage de la taille du dataset
    st.subheader("La taille du dataset")
    st.write("Le nombre de Colonnes est :", data.shape[1])
    st.write("Le nombre de Lignes est : ", data.shape[0])

    # Affichage des statistiques descriptives
    st.subheader("Statistiques Descriptives")
    st.write(data.describe())

    #Affichage des types de variables
    st.subheader("Types de Variables")
    st.write(data.dtypes)

    #Affichage de plot de types de variables
    st.write(data.dtypes.value_counts().plot.pie(autopct='%1.1f%%'))
    st.pyplot(plt)
    
    # Affichage des valeurs manquantes
    st.subheader("Valeurs Manquantes")
    missing_values = data.isnull().sum()
    st.write(missing_values)
    st.write("Nous constatons que notre dataset est complet, aucune donnée n'est manquante.")

#Contenu de la page Visualisation
elif menu == "VISUALISATION":
    st.title("Visualisation des Données")
    st.markdown("Cette section présentera des visualisations pour mieux comprendre la distribution des variables et les relations entre elles.")

    # Création de la grille avec des boutons
    col1, col2, col3, col4, col5 = st.tabs(["Distribution Consommation","Distribution Températures","Distribution Humidité","Boxplot","Pairplot"])

    with col1:
        st.markdown("<h3 style='color:#5F9EA0'>Distribution de la consommation énergétique (Appliances)</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data["Appliances"], bins=30, kde=True, ax=ax, color="blue")
        ax.set_xlabel("Consommation énergétique")
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)
        #Interprétation
        st.markdown("""<h4 style="color:#2F4F4F">Interprétation de l'histogramme :</h4>""", unsafe_allow_html=True)
        st.markdown("""<p style="background-color:#F0FFF0">
                    1. La distribution est asymétrique à droite.<br>
                        2. La majorité des valeurs de consommation énergétique des appareils se situent entre 0 et 200 Wh.<br>
                        3. Quelques observations ont des consommations beaucoup plus élevées (au-delà de 200 Wh), mais elles sont rares.<br>
                        4. La plupart des données sont concentrées autour de 0 à 100 Wh, avec un pic majeur près de 0 Wh. Cela indique que dans la plupart des cas, les appareils consomment peu d'énergie. 5. On observe une longue traîne vers la droite (jusqu'à environ 1000 Wh). Cela pourrait indiquer des cas de consommation élevée qui sont rares mais significatifs.
                    </p>""", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<h3 style='color:#5F9EA0'>Distribution des températures</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in temp_cols:
            sns.kdeplot(data[col], label=col, ax=ax)
        ax.legend()
        plt.xlabel('Valeurs')
        plt.ylabel('Densité')
        st.pyplot(fig)
        #Interprétation
        st.markdown("""<h4 style="color:#2F4F4F">Interprétation :</h4>""", unsafe_allow_html=True)
        st.markdown("""<p style="background-color:#F0FFF0">
                    A partir la visualisation de la distribution des variables du température nous constatons que :<br>
                    1. Les valeurs des températures sont principalement concentrées entre 15 et 25°C.<br>
                    2. Les queues des distributions pour certaines variables (par exemple, T6 ou T3) s'étendent vers des valeurs extrêmes négatives ou positives. Cela peut être dû à des anomalies, des erreurs de capteur, ou des événements rares.
                    </p>""", unsafe_allow_html=True)

    with col3:
        st.markdown("<h3 style='color:#5F9EA0'>Distribution de l'humidité</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in humidity_cols:
            sns.kdeplot(data[col], label=col, ax=ax)
        ax.legend()
        plt.xlabel('Valeurs')
        plt.ylabel('Densité')
        st.pyplot(fig)
        #Interprétation
        st.markdown("""<h4 style="color:#2F4F4F">Interprétation :</h4>""", unsafe_allow_html=True)
        st.markdown("""<p style="background-color:#F0FFF0">
                    1. Les valeurs des humidités sont principalement concentrées entre '30%' et '50%'.<br>
                    2. Les pics de densité indiquent une forte fréquence d'observations autour de ces valeurs pour toutes les variables d'humidité.<br>
                    3. Les courbes de densité pour certaines variables (par exemple, RH_5 et RH_6) s'étendent vers des valeurs extrêmes négatives ou positives. C'est-à-dire il y a des anomalies.
                    </p>""", unsafe_allow_html=True)

    with col4:    
        st.markdown("<h3 style='color:#5F9EA0'>Boxplot des variables de température et d'humidité</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=data[temp_cols + humidity_cols])
        plt.xticks(rotation=90)
        st.pyplot(fig)
     
    with col5:
        st.markdown("<h3 style='color:#5F9EA0'>Corrélation avec Pairplot</h3>", unsafe_allow_html=True)
        selected_cols = ["Appliances"] + temp_cols[:3] + humidity_cols[:3]  # Sélection de quelques variables
        fig = sns.pairplot(data[selected_cols], diag_kind="kde")
        st.pyplot(fig)

# Contenu de la page Pré-traitement  
elif menu == "PRE-TRAITEMENT":
    st.title("Pré-traitement des Données")
    st.markdown("Nettoyage des données, gestion des valeurs manquantes et réduction de dimensions.")
    st.subheader("Mise à l'échelle des données")
    
    # IQR Scaling pour la variable cible
    st.markdown("<h3 style='color:#5F9EA0'>Méthode IQR pour la consommation énergétique</h3>", unsafe_allow_html=True)
    st.image("images\IQR.png", width=500)
    scaler_iqr = RobustScaler()
    data["Appliances"] = scaler_iqr.fit_transform(data[["Appliances"]])
    plt.title("Distribution après IQR")
    st.write(data["Appliances"].plot.hist(bins=50))
    st.pyplot(plt)
    
    # Normalisation des températures
    st.markdown("<h3 style='color:#5F9EA0'>Normalisation des variables de température</h3>", unsafe_allow_html=True)
    st.image("images/Norm.jpg", width=500)
    scaler_minmax = MinMaxScaler()
    data[temp_cols] = scaler_minmax.fit_transform(data[temp_cols])
    st.write("Distribution après normalisation")
    st.write(pd.DataFrame(data[temp_cols]).describe())
    plt.figure(figsize=(12,6))
    plt.hist(data[temp_cols], bins=50, alpha=0.7)
    plt.title('Température après la mise à l échelle')
    st.pyplot(plt)
    
    # Standardisation des variables d'humidité
    st.markdown("<h3 style='color:#5F9EA0'>Standardisation des variables d'humidité</h3>", unsafe_allow_html=True)
    st.image("images\Standardization.png")
    scaler_standard = StandardScaler()
    data[humidity_cols] = scaler_standard.fit_transform(data[humidity_cols])
    st.write("Distribution après standardisation")
    st.write(pd.DataFrame(data[humidity_cols]).describe())
    plt.figure(figsize=(12,6))
    plt.hist(data[humidity_cols], bins=50, alpha=0.7)
    plt.title('Humidité après la mise à l échelle')
    st.pyplot(plt)
    
    # ACP
    st.markdown("<h3 style='color:#5F9EA0'>Analyse en Composantes Principales (ACP)</h3>", unsafe_allow_html=True)
    pca = PCA(n_components=6)
    principal_components = pca.fit_transform(data[temp_cols + humidity_cols + external_cols])
    
    # Affichage de l'inertie expliquée
    def plot_var_explique (acp):
        var_explique = acp.explained_variance_ratio_
        plt.figure(figsize=(12,6))
        plt.bar(np.arange(len(var_explique ))+1, var_explique )
        plt.plot(np.arange(len(var_explique ))+1, var_explique .cumsum(),c="red",marker='o')
        plt.xlabel("la rang de l'axe d'inertie")
        plt.ylabel("pourcentage d'inertie")
        plt.title(" Eboulis des valeurs propres")
        st.pyplot(plt)
    plot_var_explique (pca)
    st.write("Variance expliquée par les composantes principales")
    st.write(pd.DataFrame({"Composante": [f"PC{i+1}" for i in range(6)], "Variance expliquée": pca.explained_variance_ratio_}))
    
    # Affichage des composantes principales
    st.write("Composantes principales")
    st.write(pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(6)]).head())

# Contenu de la page Régression        
elif menu == "REGRESSION":
    st.title("Modèles de Régression")
    st.markdown("Cette section présentera les modèles de régression linéaire et de classification pour prédire la consommation énergétique des appareils.")
    #Mise à L'échelle des données
    scaler_iqr = RobustScaler()
    data["Appliances"] = scaler_iqr.fit_transform(data[["Appliances"]])
    scaler_minmax = MinMaxScaler()
    data[temp_cols] = scaler_minmax.fit_transform(data[temp_cols])
    scaler_standard = StandardScaler()
    data[humidity_cols] = scaler_standard.fit_transform(data[humidity_cols])
    data[external_cols] = scaler_standard.fit_transform(data[external_cols])
    # Sélection des variables indépendantes (exclure "Appliances" qui est la cible)
    X = data[temp_cols + humidity_cols + external_cols]
    y = data["Appliances"]

# Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    # Bouton pour la régression linéaire
    if st.button("Prédire avec Régression Linéaire"):
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)

        # Calcul des métriques
        mae = mean_absolute_error(y_test, y_pred_lr)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        r2 = r2_score(y_test, y_pred_lr)

        # Affichage des résultats
        st.subheader("Métriques de la Régression Linéaire")
        st.write(f"**MAE :** {mae:.2f}")
        st.write(f"**RMSE :** {rmse:.2f}")
        st.write(f"**R² :** {r2:.2f}")

        # Affichage du nuage de points
        st.subheader("Nuage de points : Prédictions vs Valeurs réelles")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_lr, alpha=0.5, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
        ax.set_xlabel("Valeurs Réelles")
        ax.set_ylabel("Prédictions")
        st.pyplot(fig)

    # Bouton pour RandomForestRegressor
    if st.button("Prédire avec RandomForestRegressor"):
        model_rf = RandomForestRegressor(random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)

        # Calcul des métriques
        mae = mean_absolute_error(y_test, y_pred_rf)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
        r2 = r2_score(y_test, y_pred_rf)

        # Affichage des résultats
        st.subheader("Métriques du Modèle RandomForestRegressor")
        st.write(f"**MAE :** {mae:.2f}")
        st.write(f"**RMSE :** {rmse:.2f}")
        st.write(f"**R² :** {r2:.2f}")

        # Affichage du nuage de points
        st.subheader("Nuage de points : Prédictions vs Valeurs réelles")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_rf, alpha=0.5, color="blue")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
        ax.set_xlabel("Valeurs Réelles")
        ax.set_ylabel("Prédictions")
        st.pyplot(fig)

    




