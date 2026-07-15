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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



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
    
        results_comparison[name] = {"Accuracy":accuracy_score(y_test, y_pred),
                                    "Precision": precision_score(y_test, y_pred),
                                    "Recall": recall_score(y_test, y_pred),
                                    "F1": f1_score(y_test, y_pred)
                                    }
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        plt.close('all')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_classes,
                    yticklabels=unique_classes)
        model_name = name.lower()
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted Operational State')
        plt.ylabel('Actual Truth State')
        plt.tight_layout()
        file_save_name = name.lower().replace(" ", "_")
        matrix_path = f'results/confusion_matrix_{file_save_name}.png'
        plt.savefig(matrix_path, dpi=300)
        print(f"Success! Confusion matrix chart saved to '{matrix_path}'")
        plt.show()
        plt.close()
    
    
    print("\n============== MODEL COMPARISON REPORT ==============")
    print(f"{'Classifier Name':<30} | {'Test Accuracy':<15}")
    print("-" * 50)
    for model_name, accuracy in results_comparison.items():
        print(f"{model_name:<30} | {accuracy * 100:>13.2f}%")
    print("=======================================================")

    plt.close('all')
    plt.figure(figsize=(10, 5))
    palette = sns.color_palette("Set1", len(classifiers)) 
    plt.bar(results_comparison.keys(), [acc * 100 for acc in results_comparison.values()], color=palette, edgecolor = 'black', width = 0.5)
    plt.title('Perfomance Accuracy Comparison of Models')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for index, value in enumerate(results_comparison.values()):
        plt.text(index, (value*100)+1.5, f"{value*100:.2f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    comparison_chart_path = 'results/classifier_accuracy_comparison.png'
    plt.savefig(comparison_chart_path, dpi=300)
    print(f"Comparison performance chart saved to '{comparison_chart_path}'\n")
    plt.show()

 


if __name__ == '__main__':
    train_diagnostic_classifier('results/master_pca_features.csv')















