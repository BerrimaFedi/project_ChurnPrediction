import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['tenure'] = df['tenure'].astype(int)

    df['Churn'] = df['Churn'].str.capitalize().map({'Yes': 1, 'No': 0})
    df.dropna(subset=['Churn'], inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns.drop('customerID')
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    X = df.drop(['Churn', 'customerID'], axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test