import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)

def evaluate_model(model, run_name, X_test, y_test):
    """
    Évalue un modèle déjà entraîné et affiche les résultats.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] \
              if hasattr(model, "predict_proba") else None

    # ── Métriques ──
    metrics = {
        "accuracy":  round(accuracy_score (y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score   (y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score       (y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score  (y_test, y_proba), 4)
                     if y_proba is not None else 0.0,
    }

    # ── Rapport complet ──
    print(f"\n{'='*50}")
    print(f"  Évaluation : {run_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred,
                                 target_names=['No Churn', 'Churn']))
    for k, v in metrics.items():
        print(f"  {k:<12} : {v}")
    print(f"{'='*50}")

    return metrics


def plot_confusion_matrix(model, run_name, X_test, y_test):
    """
    Génère et sauvegarde la matrice de confusion.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title(f"Confusion Matrix — {run_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    path = f"cm_{run_name.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close()
    print(f"  ✅ Matrice sauvegardée : {path}")
    return path


def plot_roc_curve(model, run_name, X_test, y_test):
    """
    Génère et sauvegarde la courbe ROC.
    """
    if not hasattr(model, "predict_proba"):
        print(f"  ⚠ {run_name} — predict_proba non disponible, ROC ignorée.")
        return None

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color='gray',
            linestyle='--', label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Courbe ROC — {run_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = f"roc_{run_name.replace(' ', '_')}.png"
    fig.savefig(path)
    plt.close()
    print(f"  ✅ Courbe ROC sauvegardée : {path}")
    return path


def compare_models(results: list):
    """
    Affiche un tableau comparatif de tous les modèles.
    """
    df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    df.reset_index(drop=True, inplace=True)

    print("\n" + "="*65)
    print("   TABLEAU COMPARATIF — TOUS LES MODÈLES")
    print("="*65)
    print(df.to_string(index=False))
    print("="*65)

    best = df.iloc[0]
    print(f"\n🏆 Meilleur modèle : {best['model']}")
    print(f"   F1      = {best['f1_score']}")
    print(f"   AUC     = {best['roc_auc']}")
    print(f"   Accuracy= {best['accuracy']}")

    # ── Graphique comparatif ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # F1 Score
    axes[0].barh(df['model'], df['f1_score'], color='steelblue')
    axes[0].set_xlabel('F1 Score')
    axes[0].set_title('Comparaison F1 Score')
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(df['f1_score']):
        axes[0].text(v + 0.01, i, str(v), va='center', fontsize=9)

    # ROC-AUC
    axes[1].barh(df['model'], df['roc_auc'], color='darkorange')
    axes[1].set_xlabel('ROC-AUC')
    axes[1].set_title('Comparaison ROC-AUC')
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(df['roc_auc']):
        axes[1].text(v + 0.01, i, str(v), va='center', fontsize=9)

    plt.tight_layout()
    path = "models_comparison.png"
    fig.savefig(path)
    plt.close()
    print(f"\n  ✅ Graphique comparatif sauvegardé : {path}")

    return df, path