import os
import joblib
from Evaluation import evaluate_model
from config import models_dir, X_train, y_train
from myModels import models
import time

def train_model() -> dict:
    """
    Entraîne les modèles définis, les évalue, et les sauvegarde dans un répertoire.

    Returns:
        dict: Dictionnaire contenant les métriques des modèles entraînés.
    """
    metrics = {}
    
    try:
        print("\n🔧 Début de l'entraînement des modèles...")

        for name, model in models.items():
            print(f"\n🚀 Entraînement du modèle : {name}")
            
            start_time = time.perf_counter()
            
            model.fit(X_train, y_train)
            
            end_time = time.perf_counter()
            training_time = end_time - start_time
            
            print(f"✅ Modèle {name} entraîné avec succès en {training_time:.2f} secondes.")

            print(f"📊 Évaluation du modèle : {name}")
            model_metrics = evaluate_model(name, model, training_time)
            
            metrics[name] = model_metrics
            print(f"✅ Métriques du modèle {name} : {metrics[name]}")

            # Save the model to a file
            model_file = os.path.join(models_dir, f"{name}.pkl")
            joblib.dump(model, model_file)
            print(f"💾 Modèle sauvegardé sous : {model_file}")
            print("-" * 40)

    except Exception as e:
        print(f"❌ Une erreur s'est produite lors du processus d'entraînement : {e}")

    print("\n🎉 Processus terminé avec succès.")

    return metrics
