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

# Perceptron
class Perceptron:
    def __init__(self, max_epochs_list, k_folds=5, random_state=42):
        """
        Initializes the Perceptron model with candidate epoch values 
        and cross-validation configuration.

        Args:
            max_epochs_list (list): List of candidate values for max_epochs (e.g., [10, 100, 300]).
            k_folds (int): Number of folds for k-fold cross-validation.
            random_state (int): Random seed to ensure reproducibility.
        """
        self.max_epochs_list = max_epochs_list
        self.k_folds = k_folds
        self.random_state = random_state

        self.best_epochs = None    # Optimal number of epochs (found via cross-validation)
        self.weights = None        # Final model weights

    def _ensure_numpy(self, *arrays):
        """
        Ensures all inputs are NumPy arrays.

        Args:
            *arrays: Variable-length list of input arrays.

        Returns:
            list: Converted NumPy arrays.
        """
        return [np.asarray(arr) for arr in arrays]

    def train(self, X, y, max_epochs, verbose=True):
        """
        Trains the Perceptron model using the given data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels in {-1, +1}.
            max_epochs (int): Maximum number of epochs.
            verbose (bool): Whether to print convergence messages.

        Returns:
            np.ndarray: Final weight vector.
        """
        X, y = self._ensure_numpy(X, y)
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)

        for epoch in range(max_epochs):
            updated = False
            for i in range(n_samples):
                if y[i] * np.dot(weights, X[i]) <= 0:
                    weights += y[i] * X[i]
                    updated = True
            if not updated:
                if verbose:
                    print(f"Training converged after {epoch + 1} epochs.")
                break
        else:
            if verbose:
                print(f"Max epochs ({max_epochs}) reached. No convergence.")

        return weights

    def predict(self, X, weights=None):
        """
        Predicts the labels for input data using the model.

        Args:
            X (np.ndarray): Input feature matrix.
            weights (np.ndarray, optional): Custom weight vector. 
                                            Defaults to self.weights if not specified.

        Returns:
            np.ndarray: Predicted labels in {-1, +1}.
        """
        X = np.asarray(X)
        if weights is None:
            weights = self.weights
        return np.sign(np.dot(X, weights))

    def cross_validate(self, X, y):
        """
        Performs k-fold cross-validation to select the best value for max_epochs.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training label vector.
        """
        X, y = self._ensure_numpy(X, y)
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))
        fold_size = len(X) // self.k_folds

        best_loss = float("inf")
        best_epochs = None

        for max_epochs in self.max_epochs_list:
            fold_losses = []

            for i in range(self.k_folds):
                start = i * fold_size
                end = (i + 1) * fold_size if i < self.k_folds - 1 else len(X)

                val_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                temp_weights = self.train(X_train, y_train, max_epochs, verbose=False)
                y_pred = self.predict(X_val, weights=temp_weights)

                loss = np.mean(y_pred != y_val)  # 0-1 loss
                fold_losses.append(loss)

            avg_loss = np.mean(fold_losses)
            print(f"Max epochs = {max_epochs}, Average 0-1 Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epochs = max_epochs

        self.best_epochs = best_epochs
        print(f"\nBest number of epochs: {self.best_epochs} (CV 0-1 Loss: {best_loss:.4f})")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Trains the final model using the best number of epochs and evaluates its performance
        on both training and test sets.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            tuple: (training_loss, test_loss)
        """
        X_train, y_train, X_test, y_test = self._ensure_numpy(X_train, y_train, X_test, y_test)
        if self.best_epochs is None:
            raise ValueError("Run cross_validate() before calling evaluate().")

        self.weights = self.train(X_train, y_train, self.best_epochs, verbose=True)

        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        train_loss = np.mean(y_pred_train != y_train)
        test_loss = np.mean(y_pred_test != y_test)

        print(f"\nTraining 0-1 Loss: {train_loss:.4f}")
        print(f"Test 0-1 Loss:     {test_loss:.4f}")

        return train_loss, test_loss

# Pegasos
class PegasosSVM:
    def __init__(self, lambda_candidates, iteration_candidates, k_folds=5, random_state=42):
        self.lambda_candidates = lambda_candidates
        self.iteration_candidates = iteration_candidates
        self.k_folds = k_folds
        self.random_state = random_state

        self.best_lambda = None
        self.best_iterations = None
        self.weights = None

    def _ensure_numpy(self, *arrays):
        return [np.asarray(arr) for arr in arrays]

    def train(self, X, y, lambda_, num_iterations, verbose=True):
        """
        Trains the Pegasos model and returns the average of weight vectors.
        """
        X, y = self._ensure_numpy(X, y)
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        weight_sum = np.zeros(n_features)

        for t in range(1, num_iterations + 1):
            i = np.random.randint(0, n_samples)
            x_i, y_i = X[i], y[i]
            eta = 1.0 / (lambda_ * t)

            if y_i * np.dot(w, x_i) < 1:
                w = (1 - eta * lambda_) * w + eta * y_i * x_i
            else:
                w = (1 - eta * lambda_) * w

            weight_sum += w

        average_weights = weight_sum / num_iterations

        if verbose:
            print(f"Finished training with λ = {lambda_}, iterations = {num_iterations}")
        return average_weights

    def predict(self, X, weights=None):
        X = np.asarray(X)
        if weights is None:
            weights = self.weights
        return np.sign(np.dot(X, weights))

    def cross_validate(self, X, y):
        X, y = self._ensure_numpy(X, y)
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))
        fold_size = len(X) // self.k_folds

        best_loss = float("inf")
        best_lambda, best_iterations = None, None

        for lambda_ in self.lambda_candidates:
            for iterations in self.iteration_candidates:
                fold_losses = []

                for i in range(self.k_folds):
                    start = i * fold_size
                    end = (i + 1) * fold_size if i < self.k_folds - 1 else len(X)
                    val_idx = indices[start:end]
                    train_idx = np.concatenate([indices[:start], indices[end:]])

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    weights = self.train(X_train, y_train, lambda_, iterations, verbose=False)
                    y_pred = self.predict(X_val, weights)

                    loss = np.mean(y_pred != y_val)
                    fold_losses.append(loss)

                avg_loss = np.mean(fold_losses)
                print(f"λ = {lambda_}, iterations = {iterations}, CV 0-1 Loss = {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_lambda, best_iterations = lambda_, iterations

        self.best_lambda = best_lambda
        self.best_iterations = best_iterations
        print(f"\nBest hyperparameters: λ = {self.best_lambda}, iterations = {self.best_iterations} (CV 0-1 Loss = {best_loss:.4f})")

    def evaluate(self, X_train, y_train, X_test, y_test):
        X_train, y_train, X_test, y_test = self._ensure_numpy(X_train, y_train, X_test, y_test)
        if self.best_lambda is None or self.best_iterations is None:
            raise ValueError("Run cross_validate() before evaluate().")

        self.weights = self.train(X_train, y_train, self.best_lambda, self.best_iterations)

        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        train_loss = np.mean(y_pred_train != y_train)
        test_loss = np.mean(y_pred_test != y_test)

        print(f"\nTraining 0-1 Loss: {train_loss:.4f}")
        print(f"Test 0-1 Loss:     {test_loss:.4f}")

        return train_loss, test_loss
    
    
