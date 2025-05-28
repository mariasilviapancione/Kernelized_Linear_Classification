# Import libraries for data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data exploration and preprocessing steps
class DataPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the preprocessor with default settings.
        
        Args:
            test_size (float): Proportion of the dataset to use as test set.
            random_state (int): Random seed to ensure reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state

        # Main dataset
        self.df = None

        # Train/test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # To store means and stds for standardization
        self.scaler_params = {}

    def load_data(self, path):
        """
        Loads a CSV dataset from the given file path.
        """
        self.df = pd.read_csv(path)
        return self.df

    def check_missing(self):
        """
        Prints the number of missing values per column.
        """
        return self.df.isnull().sum()

    def plot_distributions(self):
        """
        Plots variable distributions.
        """
        plt.figure(figsize=(20, 15))
        for i, column in enumerate(self.df.columns, 1):
            plt.subplot(4, 3, i)
            sns.histplot(self.df[column], kde=True)
            plt.grid(True, linestyle='--', color='gray', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_pairplot(self):
        """
        Plots pairwise relationships between numerical features.
        """
        g = sns.pairplot(self.df, hue='y', diag_kind='hist')
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        plt.tight_layout()
        plt.show()

    def split(self):
        """
        Splits the dataset into training and test sets.
        """
        np.random.seed(self.random_state)

        # Separate features and labels
        X = self.df.iloc[:, :-1]
        y = self.df.iloc[:, -1]

        # Shuffle indices
        indices = np.random.permutation(len(X))

        # Define the split point
        split_idx = int(len(X) * (1 - self.test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        # Create train/test sets and reset indices
        self.X_train = X.iloc[train_idx].reset_index(drop=True)
        self.X_test = X.iloc[test_idx].reset_index(drop=True)
        self.y_train = y.iloc[train_idx].reset_index(drop=True)
        self.y_test = y.iloc[test_idx].reset_index(drop=True)

        print(f"Training set shape: {self.X_train.shape}") 
        print(f"Test set shape: {self.X_test.shape}")

    def plot_boxplots(self):
        """
        Plots boxplots for each feature in the training set to inspect outliers.
        """
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(self.X_train.columns, 1):
            plt.subplot(2, 5, i)
            sns.boxplot(y=self.X_train[col])
            plt.title(col)
            plt.ylabel("")
            plt.tight_layout()
            
        plt.show()

    def remove_outliers(self, factor=1.5):
        """
        Removes outliers in training set using the IQR method.
        
        Args:
            factor (float): The IQR multiplier (default 1.5).
        """
        Q1 = self.X_train.quantile(0.25)
        Q3 = self.X_train.quantile(0.75)
        IQR = Q3 - Q1

        # Keep only rows where all features are within IQR bounds
        mask = ~((self.X_train < (Q1 - factor * IQR)) | (self.X_train > (Q3 + factor * IQR))).any(axis=1)
        n_removed = (~mask).sum()
        self.X_train = self.X_train[mask].reset_index(drop=True)
        self.y_train = self.y_train[mask].reset_index(drop=True)

        print(f"{n_removed} outliers removed from the training set.")

    def plot_correlations(self):
        """
        Plots the correlation matrix (heatmap) for the training set.
        """
        corr = self.X_train.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.tight_layout()
        plt.show()

    def remove_highly_correlated(self, threshold=0.95):
        """
        Removes features from the training and test sets that are highly correlated.
        
        Args:
            threshold (float): Correlation threshold above which one variable is dropped.
        """
        corr_matrix = self.X_train.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identify columns to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Keep only uncorrelated features
        self.X_train = self.X_train.drop(columns=to_drop)
        self.X_test = self.X_test.drop(columns=to_drop)

        print(f"Removed {len(to_drop)} highly correlated features: {', '.join(to_drop)}")

    def standardize(self):
        """
        Standardizes features (z-score normalization) based on training set stats.
        """
        mean = self.X_train.mean()
        std = self.X_train.std().replace(0, 1.0)  # Avoid division by zero

        self.scaler_params['mean'] = mean
        self.scaler_params['std'] = std

        self.X_train = (self.X_train - mean) / std
        self.X_test = (self.X_test - mean) / std

    def check_label_distribution(self):
        """
        Displays the distribution of the target labels in both training and test sets.
        """
        train_counts = self.y_train.value_counts(normalize=True).sort_index()
        test_counts = self.y_test.value_counts(normalize=True).sort_index()

        print("Label distribution in training set:")
        for label, pct in train_counts.items():
            print(f"  Label {label}: {pct:.2%}")

        print("\nLabel distribution in test set:")
        for label, pct in test_counts.items():
            print(f"  Label {label}: {pct:.2%}")


