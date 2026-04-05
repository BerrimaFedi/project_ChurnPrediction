import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')  # important sur Windows
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from evaluate import evaluate_model, plot_roc_curve, compare_models

from mlflow.models.signature import infer_signature
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              roc_auc_score, confusion_matrix)
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.decomposition import PCA

from preprocessing import load_and_prepare

# ── MLflow setup ──
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Churn_Prediction")


def train_and_log(model, run_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=run_name):

        # Log params
        mlflow.log_params(params)
        mlflow.set_tag("dataset", "Telco Customer Churn")

        # Entraînement
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] \
                  if hasattr(model, "predict_proba") else None

        # Métriques
        metrics = {
            "accuracy":  round(accuracy_score (y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score   (y_test, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score       (y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score  (y_test, y_proba), 4)
                         if y_proba is not None else 0.0,
        }
        mlflow.log_metrics(metrics)

        # Sauvegarde modèle
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"churn_{run_name.lower().replace(' ','_')}"
        )

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        ax.set_title(f"Confusion Matrix — {run_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        img_path = f"cm_{run_name.replace(' ','_')}.png"
        fig.savefig(img_path)
        mlflow.log_artifact(img_path)
        plt.close()

        # Console
        print(f"\n{'─'*45}")
        print(f"  {run_name}")
        for k, v in metrics.items():
            print(f"  {k:<12} : {v}")
        print(f"{'─'*45}")

        return {**{"model": run_name}, **metrics}


if __name__ == "__main__":

    DATA_PATH = r"C:\project_ChurnPrediction\data\raw\Telecom Customers Churn.csv"
    X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)

    results = []

    # ── 1. KNN ──
    print("\n🔵 KNN...")
    for k in [3, 5, 7, 11]:
        p = {"n_neighbors": k, "weights": "uniform", "metric": "euclidean"}
        results.append(train_and_log(
            KNeighborsClassifier(**p), f"KNN_k{k}", p,
            X_train, X_test, y_train, y_test))

    # ── 2. SVM ──
    print("\n🔴 SVM...")
    for kernel in ["rbf", "linear"]:
        p = {"C": 1.0, "kernel": kernel, "gamma": "scale", "probability": True}
        results.append(train_and_log(
            SVC(**p), f"SVM_{kernel}", p,
            X_train, X_test, y_train, y_test))

    # ── 3. Random Forest ──
    print("\n🟢 Random Forest...")
    for n, depth in [(50, 5), (100, 10), (200, 15)]:
        p = {"n_estimators": n, "max_depth": depth,
             "min_samples_split": 2, "random_state": 42}
        results.append(train_and_log(
            RandomForestClassifier(**p), f"RF_n{n}_d{depth}", p,
            X_train, X_test, y_train, y_test))

    # ── 4. Logistic Regression ──
    print("\n🟡 Logistic Regression...")
    for C in [0.1, 1.0, 10.0]:
        p = {"C": C, "solver": "lbfgs", "max_iter": 1000, "random_state": 42}
        results.append(train_and_log(
            LogisticRegression(**p), f"LR_C{C}", p,
            X_train, X_test, y_train, y_test))

    # ── 5. PCA + Random Forest ──
    print("\n🟣 PCA + Random Forest...")
    pca = PCA(n_components=10, random_state=42)
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)
    print(f"   Variance expliquée : {pca.explained_variance_ratio_.sum()*100:.1f}%")

    p = {"n_estimators": 100, "max_depth": 10,
         "random_state": 42, "pca_components": 10}
    results.append(train_and_log(
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "RF_PCA_n100", p,
        Xtr_pca, Xte_pca, y_train, y_test))
    
    df_res, chart_path = compare_models(results)
    mlflow.log_artifact(chart_path)
    # ── Tableau final ──
    # df_res = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    # print("\n" + "="*60)
    # print("   TABLEAU COMPARATIF FINAL")
    # print("="*60)
    # print(df_res.to_string(index=False))
    # print("="*60)
    # best = df_res.iloc[0]
    # print(f"\n🏆 Meilleur modèle : {best['model']}")
    # print(f"   F1={best['f1_score']} | AUC={best['roc_auc']} | Acc={best['accuracy']}")