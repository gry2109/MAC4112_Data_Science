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
    Train and evaluate multiple machine learning classifiers for machining
    condition diagnosis.

    This function loads a preprocessed feature dataset, separates the
    feature variables from the target labels, and performs a train-test split 
    to preserve the class distribution. Four supervised
    classification algorithms are trained and evaluated using the same
    training and evaluation datasets to enable a fair comparison of their
    performance.

    For each classifier, the function calculates common performance
    metrics, generates a confusion matrix, and saves the resulting
    visualisations to the project's results directory. A grouped bar chart
    comparing the performance of all classifiers is also produced.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file containing the preprocessed feature dataset.
        The default is 'results/master_pca_features.csv'.

    Returns
    -------
    dict
        A dictionary containing the performance metrics (Accuracy,
        Precision, Recall and F1-score) for each trained classifier.

    Notes
    -----
    The classifiers currently evaluated are:

    - Random Forest
    - Decision Tree
    - Support Vector Machine 
    - k-Nearest Neighbours

    The train-test split uses a fixed random seed to ensure reproducible
    experimental results.
    """
    # load data
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("Run_Number", "Target_Condition")]
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
    
        results_comparison[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='macro'),
            "Recall": recall_score(y_test, y_pred, average='macro'),
            "F1": f1_score(y_test, y_pred, average='macro')
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
        # plt.show()
        plt.close()
    
    
    print("\n======================= MODEL COMPARISON REPORT =======================")
    print(f"{'Classifier Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 70)
    for model_name, metrics in results_comparison.items():
        # Retrieve 'Accuracy' from the metrics dictionary
        acc = metrics["Accuracy"]*100 
        pre = metrics["Precision"]*100
        rec = metrics["Recall"]*100
        f1 = metrics["F1"]*100
        print(f"{model_name:<25} | {acc:>8.2f}% | {pre:>8.2f}% | {rec:>8.2f}% | {f1:>8.2f}%")
    print("=========================================================================")


    performance_df = pd.DataFrame(results_comparison).T * 100
    
   
    plt.close('all')
    ax = performance_df.plot(kind='bar', figsize=(12, 6), edgecolor='black', width=0.8)
    
    plt.title('Multi-Metric Performance Comparison of Diagnostic Classifiers', fontsize=14, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Classifier Model', fontsize=12)
    plt.xticks(rotation=0) # Keep model names horizontal so they are readable
    plt.ylim(0, 140)      # Give room for the legend and labels
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Place the legend in a clean spot
    plt.legend(loc='upper right', framealpha=0.9)
    
    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=8)
        
    plt.tight_layout()
    comparison_chart_path = 'results/classifier_multi_metric_comparison.png'
    plt.savefig(comparison_chart_path, dpi=300)
    print(f"Grouped performance chart saved to '{comparison_chart_path}'\n")
    # plt.show()

    return results_comparison

if __name__ == '__main__':
    train_diagnostic_classifier('results/master_pca_features.csv')
