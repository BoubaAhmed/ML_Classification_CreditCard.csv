import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from config import raw_data_file_path, output_dir, processed_data_dir, cleaned_data_file_path
import plotly.graph_objects as go

def exploration():
    # 1. Vérifier si le fichier existe
    if not os.path.exists(raw_data_file_path):
        raise FileNotFoundError(f"{raw_data_file_path} does not exist.")

    # 2. Lecture des données
    df = pd.read_csv(raw_data_file_path)
    df_cleaned = pd.read_csv(cleaned_data_file_path)
    print(df['Class'].value_counts())
    print(df_cleaned['Class'].value_counts())
    # 🚀 **1. Aperçu des données**
    print("\n🌐 **Aperçu des données :**")
    print(df.head())
    print("\n✅ Types des colonnes :")
    print(df.dtypes)
    print("\n📝 Dimensions du dataset (lignes, colonnes) :")
    print(df.shape)
    print("\nLes colonnes sont : :")
    print(list (df. columns ))

    # 🚨 **2. Vérification des doublons**
    print("\n🔍 **Vérification des doublons :**")
    total_duplicates = df.duplicated().sum()
    print(f"Nombre total de doublons dans le dataset : {total_duplicates}")
    
    if total_duplicates > 0:
        print("\n🔄 **Aperçu des doublons :**")
        print(df[df.duplicated()])
        
    print("\n🔍 **Vérification des doublons par certaines colonnes :**")
    duplicate_subset = df.duplicated(subset=['Amount', 'Class']).sum()  # Example subset
    print(f"Doublons trouvés pour les colonnes Amount et Class : {duplicate_subset}")


    # 🚨 **4. Valeurs manquantes**
    print("\n📊 **Valeurs manquantes pour chaque colonne :**")
    print(df.isnull().sum())

    # # 🚀 **3. Statistiques descriptives des colonnes numériques**
    print("\n📈 **Résumé des statistiques descriptives :**")
    print(df.describe())
    summary = df.describe()

    fig = go.Figure(data=[go.Table(
        header=dict(values=[''] + list(summary.columns), align='left'),
        cells=dict(values=[summary.index] + [summary[col].tolist() for col in summary.columns], align='left'))
    ])
    fig.show()

    # 📊 **5. Matrice de corrélation des variables numériques**
    print("\n📊 **Matrice de corrélation des variables numériques CLEAN DATA:**")
    correlation_matrix = df_cleaned.corr()
    print(correlation_matrix)
    
    
    # 🔥 **Visualisation de la matrice de corrélation**
    plt.figure(figsize=(16, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('🔗 Matrice de corrélation')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))  # Optional: Save the figure
    plt.show()


    # Find the 3 features most correlated with Amount
    top_3_features = correlation_matrix['Amount'].abs().sort_values(ascending=False).head(4)
    print("\nLes 3 features les plus corrélées avec le budget :")
    print(top_3_features)

    for feature in top_3_features.index[1:]:  
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=df[feature], y=df['Amount'])
        plt.title(f"Amount vs {feature}")
        plt.xlabel(feature)
        plt.ylabel('Amount')
        plt.savefig(os.path.join(output_dir, f'amount_vs_{feature}.png'))
        plt.show()


    # 🎯 **4. Valeurs uniques pour certaines colonnes**
    print("\n🔍 **Valeurs uniques pour la colonne 'Amount' :**")
    print(df['Amount'].unique())
    print("\n🔍 **Valeurs uniques pour la colonne 'Class' :**")
    print(df['Class'].unique())
    print("\n🔍 **Valeurs uniques pour la colonne 'V1' :**")
    print(df['V1'].unique())


    # 🚀 **5. Distributions des colonnes sélectionnées dans une seule grille**
    columns_to_plot = ['V1', 'V2', 'Class', 'Amount']

    plt.figure(figsize=(16, 10))  # Taille de la grille

    for i, column in enumerate(columns_to_plot, start=1):
        plt.subplot(2, 2, i)  # Grille de 2x2
        sns.histplot(df_cleaned[column].dropna(), kde=True, bins=30)
        plt.title(f'📊 Distribution de {column}')
        plt.xlabel(column)
        plt.ylabel('Fréquence')

    plt.tight_layout()  # Ajuste les marges pour éviter le chevauchement
    plt.savefig(os.path.join(output_dir, "grid_distributions.png"))  # Sauvegarde dans le dossier
    plt.show()

    # Distribution de la colonne Amount
    sns.histplot(df['Amount'], kde=True, bins=30)
    plt.title('Distribution de Amount')
    plt.show()

    sns.countplot(x='Class', data=df)
    plt.title("Distribution des classes (0 = Non-frauduleux, 1 = Frauduleux)")
    plt.show()

    # Relation entre Amount et Class
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Relation entre Class et Amount')
    plt.show()

    # Boxplot pour Amount
    sns.boxplot(x=df['Amount'])
    plt.title('Distribution de Amount avec outliers')
    plt.show()

    # Scatter plot entre V1 et V2 coloré par Class
    sns.scatterplot(x='V1', y='V2', hue='Class', data=df)
    plt.title('Relation entre V1 et V2')
    plt.show()


    distribution = df['Class'].value_counts()
    print('Distribution des classes :')
    print(distribution)
    distribution_sd = df_cleaned['Class'].value_counts()
    print('Distribution des classes :')
    print(distribution_sd)




    # Tracer un graphique en barres avec Seaborn
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Class', data=df_cleaned, palette='viridis')
    plt.title('Distribution des Classes')
    plt.xlabel('Classes')
    plt.ylabel('Nombre d\'occurrences')
    plt.show()


    print("\n✅ **Fin de l'exploration complète des données Credit Card**")

exploration()