import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



def train_diagnostic_classifier(csv_path='results/master_pca_features.csv'):
    """
    """
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith('PC')]
    x = df[feature_cols]
    y = df['Target_Condition']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size =0.20, stratify=y, random_state = 42
    )
    print(f"ML Training Set: {len(x_train)} runs | Evaluation Set: {len(x_test)} runs")

    print("Training Random Forest Fault Classifier")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("\n============== MODEL PERFOMRANCE REPORT ==============")
    print(classification_report(y_test, y_pred))
    print("========================================================")

    unique_classes = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    plt.close('all')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_classes,
                yticklabels=unique_classes)
    plt.title('CNC Condition Monitoring Confusion Matrix')
    plt.xlabel('Predicted Operational State')
    plt.ylabel('Actual Truth State')
    plt.tight_layout()


    matrix_path = 'results/diagnostic_confusion_matrix.png'
    plt.savefig(matrix_path, dpi=300)
    print(f"Success! Confusion matrix chart saved to '{matrix_path}'")
    plt.show()














    



























