import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def train_diagnostic_classifier(csv_path='results/master_pca_features.csv'):
    """
    """
    # load data
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith('PC')]
    x = df[feature_cols]
    y = df['Target_Condition']

    # Train and test split 
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size =0.20, stratify=y, random_state = 42
    )
    print(f"ML Training Set: {len(x_train)} runs | Evaluation Set: {len(x_test)} runs")

    # Dictionary of classifiers to train 
    classifiers = {'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                   'Decision Tree': DecisionTreeClassifier(random_state=42),
                   'Support Vector Machine': SVC(kernel='rbf', C=1.0, random_state=42),
                   'k-Nearest Neighbours': KNeighborsClassifier(n_neighbors=5)
                   }
    # Dictionary for final test accuracies of the models for a comparison
    results_comparison = {}
    unique_classes = sorted(y.unique())

    # Loop throuhg models, train and evaluate performance 
    for name, model in classifiers.items():
        print(f"============== TRAINING {name.upper()} ==============")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    
        acc = accuracy_score(y_test, y_pred)
        results_comparison[name] = acc
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
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
    file_save_name = name.lower().replace(" ", "_")
    matrix_path = f'results/confusion_matrix_{file_save_name}.png'
    plt.savefig(matrix_path, dpi=300)
    print(f"Success! Confusion matrix chart saved to '{matrix_path}'")
    plt.show()
    

    # print("\n============== MODEL PERFOMRANCE REPORT ==============")
    # print(classification_report(y_test, y_pred))
    # print("========================================================")
    
    print("\n============== MODEL COMPARISON REPORT ==============")
    print(f"{'Classifier Name':<30} | {'Test Accuracy':<15}")
    print("-" * 50)
    for model_name, accuracy in results_comparison.items():
        print(f"{model_name:<30} | {accuracy * 100:>13.2f}%")
    print("=======================================================")

    

 


















