import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def perform_pca(df, feature_cols, n_components=15):
    """
    Reduce the dimensionality of the engineered feature dataset using
    Principal Component Analysis (PCA).

    This function applies PCA to the selected numerical feature columns,
    transforming the original feature space into a smaller set of
    principal components. The resulting components capture the
    majority of the variation present in the original dataset while
    reducing redundancy between correlated variables. Metadata columns are
    preserved and reattached to the transformed dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing both metadata and numerical features.

    feature_cols : list[str]
        List of numerical feature columns to be included in the PCA.

    n_components : int, optional
        Number of principal components to retain.

    Returns
    -------
    tuple
        A tuple containing:

        - pandas.DataFrame
            DataFrame containing the retained principal components together
            with the original metadata columns.

        - numpy.ndarray
            Explained variance ratio for each retained principal component.
    """
    # Seperate the features and the identity columns again 

    data_features = df[feature_cols]
    data_identity = df.drop(columns=feature_cols).reset_index(drop=True)
    
    # Run the PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_features)

    # Create a DataFrame for the principal components
    pc_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(data=principal_components, columns=pc_columns)

    # Combine the identity columns with the principal components
    result_df = pd.concat([data_identity, df_pca], axis=1)
    
    return result_df, pca.explained_variance_ratio_



def plot_pca(df_pca, variance_ratio):
    """
    Generate and save a 3D visualisation of the principal
    component analysis.

    This function creates a 3D scatter plot of the first three principal
    components, with observations coloured according to their target
    operating condition. The percentage of variance explained by each
    principal component is displayed on the corresponding axis to aid
    interpretation of the dimensionality reduction.

    Parameters
    ----------
    df_pca : pandas.DataFrame
        DataFrame containing the principal components and associated
        metadata, including the target condition labels.

    variance_ratio : numpy.ndarray
        Explained variance ratio for each retained principal component,
        used to annotate the plot axes.

    Returns
    -------
    None
        The PCA visualisation is saved to the project's `results/`
        directory and summary information is printed to the terminal.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # colour code the plot
    conditions = df_pca['Target_Condition'].unique()
    palette = sns.color_palette("Set1", len(conditions)) 

    for cond, colour in zip(conditions, palette):
        subset = df_pca[df_pca['Target_Condition'] == cond]
        ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], label=cond, alpha=0.7, color=colour)

    # labels on the axis
    ax.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({variance_ratio[1]*100:.1f}% variance)')
    ax.set_zlabel(f'PC3 ({variance_ratio[2]*100:.1f}% variance)')
    ax.set_title('PCA of Extracted Features')
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=20, azim=30)  # Adjust the viewing angle for better visualization
    fig.savefig('results/pca_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("PCA scatter plot saved to 'results/pca_scatter_plot.png'")
    print("Explained variance ratio per component:", variance_ratio)
    print("Total explained variance:", sum(variance_ratio))



if __name__ == "__main__":
    print("Running standalone PCA test...")
    
    input_path = "results/master_features_standardised.csv"
    output_csv = "results/master_pca_features.csv"
    output_plot = "results/pca_scatter_plot.png"
    
    try:
        # 1. Load standardised data
        df_scaled = pd.read_csv(input_path)
        metadata_cols = ['Run_Number', 'Target_Condition']
        features = [col for col in df_scaled.columns if col not in metadata_cols]
        
        # 2. Run PCA
        df_pca_output, variance = perform_pca(df_scaled, features)
        
        # 3. Save PCA data to CSV
        df_pca_output.to_csv(output_csv, index=False)
        print(f"PCA successful! Saved to {output_csv}")
        
        # 4. Generate and save the plot
        plot_pca(df_pca_output, variance)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{input_path}'. Make sure standardisation ran first!")




