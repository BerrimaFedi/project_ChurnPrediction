# Churn Prediction — Projet MLA

## Tâche 3 : Expérimentation ML avec MLflow

### Dataset
Telco Customer Churn — 7 043 clients, 21 features

### Algorithmes testés
| Modèle | Paramètres testés |
|--------|-------------------|
| KNN    | k = 3, 5, 7, 11   |
| SVM    | kernel = rbf, linear |
| Random Forest | n=50/100/200, depth=5/10/15 |
| Logistic Regression | C = 0.1, 1.0, 10.0 |
| RF + PCA | n_components = 10 |

### Meilleur modèle
**Logistic Regression (C=1.0)** — F1=0.6032 | AUC=0.842

### Lancer le projet
```bash
# Terminal 1
python -m mlflow ui --port 5000

# Terminal 2
cd src
python train.py
```

### Structure
```
project_ChurnPrediction/
├── data/raw/          # dataset CSV
├── models/            # modèles sauvegardés
├── mlruns/            # runs MLflow
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```
