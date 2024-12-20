import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import processed_data_dir, raw_data_file_path
from imblearn.over_sampling import SMOTE, ADASYN 
from imblearn.under_sampling import TomekLinks, NearMiss
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from imblearn.combine import SMOTEENN , SMOTETomek

def preprocessing_data(test_size: float = 0.2, random_state: int = 42) -> None:
    """Prétraitement des données pour le tp de Credit Card.

    Args:
        test_size (float): Proportion des données utilisées pour le test.
        random_state (int): Seed pour assurer la reproductibilité.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Prétraitement des données commencé.")


    # 📂 **1. Vérification de l'existence des fichiers**
    if not os.path.exists(raw_data_file_path):
        raise FileNotFoundError(f"❌ Le fichier spécifié n'existe pas : {raw_data_file_path}")

    # 📖 **2. Chargement des données**
    print("\n🌐 Chargement des données...")
    df = pd.read_csv(raw_data_file_path)
    

    print("\n🔍 **Aperçu des données :**")
    print(df.head())
    print("\n📊 Dimensions du dataset (lignes, colonnes) :", df.shape)
    print("\n🔑 Colonnes disponibles :", list(df.columns))

    # 🧹 **3. Nettoyage des données**
    print("\n🧹 Suppression des doublons...")
    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    cleaned_rows = df.shape[0]
    print(f"✅ Nombre de doublons supprimés : {initial_rows - cleaned_rows}")
    print(f"✅ Nombre de lignes après nettoyage : {cleaned_rows}")

    # 🚨 **4. Vérification de la colonne cible**
    target = 'Class'
    if target not in df.columns:
        raise KeyError(f"❌ La colonne cible '{target}' est absente du dataset.")

    # 🗂️ **5. Séparation des caractéristiques et de la cible**
    print("\n✂️ Séparation des caractéristiques et de la colonne cible...")
    features = [col for col in df.columns if col != target]
    X = df[features]
    y = df[target]
    print(f"🔢 Nombre de caractéristiques : {len(features)}")

    # ⚙️ **6. Normalisation des données**
    print("\n⚙️ Normalisation des caractéristiques...")
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # ✂️ **7. Séparation des données en ensembles d'entraînement et de test**
    print("\n✂️ Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"📊 Taille de l'ensemble d'entraînement : {X_train.shape[0]} lignes")
    print(f"📊 Taille de l'ensemble de test : {X_test.shape[0]} lignes")

    # Avant SMOTE OR ADASYN (dataset complet)
    print("📊 Distribution des classes Equilibrage :")
    print(y_train.value_counts())
    plot_class_distribution(y_train, "Distribution des classes avant SMOTE OR ADASYN")

    # # 🟢 Application de SMOTE
    # print("\n🟢 Application de SMOTE pour équilibrer les classes...")
    # smote = SMOTE(random_state=random_state)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    # # Après SMOTE (ensemble d'entraînement)
    # print("\n🔍 Distribution des classes après SMOTE (Ensemble d'entraînement) :")
    # print(y_train.value_counts())
    # plot_class_distribution(y_train, "Distribution des classes après SMOTE (Entraînement)")


    ## 🟢 Application de ADASYN
    # print("\n🟢 Application de ADASYN pour équilibrer les classes...")
    # adasyn = ADASYN(random_state=random_state, n_neighbors=5)
    # X_train, y_train = adasyn.fit_resample(X_train, y_train)

    # # Après ADASYN (ensemble d'entraînement)
    # print("\n🔍 Distribution des classes après ADASYN (Ensemble d'entraînement) :")
    # print(y_train.value_counts())
    # plot_class_distribution(y_train, "Distribution des classes après ADASYN (Entraînement)")

    
    # # 🟢 Application de SMOTE-SVM (SMOTE + SVM)
    # print("\n🟢 Application de SMOTE pour équilibrer les classes et SVM pour la classification...")

    # # SMOTE avec SVM (en utilisant SMOTEENN pour équilibrer les classes et appliquer SVM)
    # smote_svm = SMOTEENN(random_state=random_state)
    # X_train, y_train = smote_svm.fit_resample(X_train, y_train)

    # # Après SVM (ensemble d'entraînement après SMOTE)
    # print("\n🔍 Distribution des classes après SMOTE-SVM (Ensemble d'entraînement) :")
    # print(y_train.value_counts())
    # plot_class_distribution(y_train, "Distribution des classes après SMOTE-SVM (Entraînement)")

    # 🟢 Application de Tomek Links pour équilibrer les classes
    # print("\n🟢 Application de Tomek Links pour nettoyer les données...")

    # # Initialisation et application de Tomek Links
    # tomek_links = TomekLinks()
    # X_train, y_train = tomek_links.fit_resample(X_train, y_train)

    # # Afficher la distribution des classes après l'application de Tomek Links
    # print("\n🔍 Distribution des classes après Tomek Links (Ensemble d'entraînement) :")
    # print(y_train.value_counts())
    # plot_class_distribution(y_train, "Distribution des classes après Tomek Links (Entraînement)")

    # 🟢 Application de NearMiss
    print("\n🟢 Application de NearMiss pour équilibrer les classes...")
    nearmiss = NearMiss()
    X_train, y_train = nearmiss.fit_resample(X_train, y_train)

    # Après NearMiss (ensemble d'entraînement)
    print("\n🔍 Distribution des classes après NearMiss (Ensemble d'entraînement) :")
    print(y_train.value_counts())
    plot_class_distribution(y_train, "Distribution des classes après NearMiss (Entraînement)")



    # 🟢 Application de SMOTE-Tomek
    # print("\n🟢 Application de SMOTE-Tomek pour équilibrer les classes...")
    # smote_tomek = SMOTETomek(random_state=42)
    # X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

    # # Après SMOTE-Tomek (ensemble d'entraînement)
    # print("\n🔍 Distribution des classes après SMOTE-Tomek (Ensemble d'entraînement) :")
    # print(y_train.value_counts())
    # plot_class_distribution(y_train, "Distribution des classes après SMOTE-Tomek (Entraînement)")



    # 💾 **8. Enregistrement des ensembles préparés**
    print("\n💾 Enregistrement des fichiers prétraités...")
    os.makedirs(processed_data_dir, exist_ok=True)

    cleaned_file_path = os.path.join(processed_data_dir, "cleaned_dataset.csv")
    X_train_file = os.path.join(processed_data_dir, "X_train.csv")
    X_test_file = os.path.join(processed_data_dir, "X_test.csv")
    y_train_file = os.path.join(processed_data_dir, "y_train.csv")
    y_test_file = os.path.join(processed_data_dir, "y_test.csv")

    # Enregistrer le dataset nettoyé et les ensembles d'entraînement/test
    df.to_csv(cleaned_file_path, index=False)
    X_train.to_csv(X_train_file, index=False)
    X_test.to_csv(X_test_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    y_test.to_csv(y_test_file, index=False)

    print(f"✅ Dataset nettoyé enregistré : {cleaned_file_path}")
    print(f"✅ Fichiers d'entraînement et de test enregistrés dans le dossier : {processed_data_dir}")

    # 🎉 **Fin du prétraitement**
    print("\n🎉 Prétraitement des données terminé avec succès !")


def plot_class_distribution(y, title: str):
    """Visualiser la distribution des classes."""
    class_labels = y.value_counts().index.map(str)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y, palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("Classe", fontsize=12)
    plt.ylabel("Nombre d'échantillons", fontsize=12)
    plt.xticks(ticks=range(len(class_labels)), labels=class_labels)
    plt.show()
