# Import libraries for data handling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

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

    def remove_highly_correlated(self, threshold=0.90):
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
            max_epochs_list (list): List of candidate values for max_epochs.
            k_folds (int): Number of folds for k-fold cross-validation.
            random_state (int): Random seed to ensure reproducibility.
        """
        self.max_epochs_list = max_epochs_list
        self.k_folds = k_folds
        self.random_state = random_state

        self.best_epochs = None    # Optimal number of epochs
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
            weights (np.ndarray, optional): Custom weight vector (default: self.weights).

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
            print(f"Max epochs = {max_epochs}, CV 0-1 Loss = {avg_loss:.4f}")

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

# SVM Pegasos
class PegasosSVM:
    def __init__(self, reg_params, n_iterations, k_folds=5, random_state=42):
        """
        Initializes the Pegasos SVM classifier with hyperparameter candidates and cross-validation settings.

        Args:
            reg_params (list): List of candidate values for the regularization parameter λ.
            n_iterations (list): List of candidate values for number of training iterations.
            k_folds (int): Number of folds for k-fold cross-validation (default is 5).
            random_state (int): Seed for reproducibility (default is 42).
        """
        self.reg_params = reg_params
        self.n_iterations = n_iterations
        self.k_folds = k_folds
        self.random_state = random_state

        self.best_reg = None      # Best λ value selected by cross-validation
        self.best_n_iter = None   # Best number of iterations
        self.weights = None       # Final model weights after training

    def _ensure_numpy(self, *arrays):
        """
        Ensures all inputs are NumPy arrays.

        Args:
            *arrays: Variable-length list of input arrays.

        Returns:
            list: Converted NumPy arrays.
        """
        return [np.asarray(arr) for arr in arrays]

    def train(self, X, y, reg, n_iter, verbose=True):
        """
        Trains the SVM model using the Pegasos algorithm.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector with labels in {-1, +1}.
            reg (float): Regularization parameter λ.
            n_iter (int): Number of training iterations.
            verbose (bool): Whether to print progress messages.

        Returns:
            np.ndarray: Averaged weight vector over all iterations.
        """
        X, y = self._ensure_numpy(X, y)
        n_samples, n_features = X.shape
        w = np.zeros(n_features)         
        weight_sum = np.zeros(n_features)  

        for t in range(1, n_iter + 1):
            i = np.random.randint(0, n_samples)   # Random sample index
            x_i, y_i = X[i], y[i]
            eta = 1.0 / (reg * t)                 # Learning rate

            # Sub-gradient update rule based on hinge loss
            if y_i * np.dot(w, x_i) < 1:
                w = (1 - eta * reg) * w + eta * y_i * x_i
            else:
                w = (1 - eta * reg) * w

            weight_sum += w 

        average_weights = weight_sum / n_iter

        if verbose:
            print(f"Finished training with λ = {reg}, iterations = {n_iter}")
        return average_weights

    def predict(self, X, weights=None):
        """
        Predicts labels for input data using the trained model.

        Args:
            X (np.ndarray): Input feature matrix.
            weights (np.ndarray, optional): Optional custom weights (default: self.weights).

        Returns:
            np.ndarray: Predicted labels in {-1, +1}.
        """
        X = np.asarray(X)
        if weights is None:
            weights = self.weights
        return np.sign(np.dot(X, weights))

    def cross_validate(self, X, y):
        """
        Performs k-fold cross-validation to select the best λ and number of iterations.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target label vector.
        """
        X, y = self._ensure_numpy(X, y)
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))
        fold_size = len(X) // self.k_folds

        best_loss = float("inf")

        for reg in self.reg_params:
            for n_iter in self.n_iterations:
                fold_losses = []

                for i in range(self.k_folds):
                    start = i * fold_size
                    end = (i + 1) * fold_size if i < self.k_folds - 1 else len(X)
                    val_idx = indices[start:end]
                    train_idx = np.concatenate([indices[:start], indices[end:]])

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    weights = self.train(X_train, y_train, reg, n_iter, verbose=False)
                    y_pred = self.predict(X_val, weights)
                    loss = np.mean(y_pred != y_val) 
                    fold_losses.append(loss)

                avg_loss = np.mean(fold_losses)
                print(f"λ = {reg}, iterations = {n_iter}, CV 0-1 Loss = {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.best_reg = reg
                    self.best_n_iter = n_iter

        print(f"\nBest hyperparameters: λ = {self.best_reg}, iterations = {self.best_n_iter} (CV 0-1 Loss = {best_loss:.4f})")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Trains the model using the best hyperparameters and evaluates on train and test sets.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            tuple: (training_loss, test_loss)
        """
        X_train, y_train, X_test, y_test = self._ensure_numpy(X_train, y_train, X_test, y_test)

        if self.best_reg is None or self.best_n_iter is None:
            raise ValueError("Run cross_validate() before evaluate().")

        # Final training with best hyperparameters
        self.weights = self.train(X_train, y_train, self.best_reg, self.best_n_iter)

        # Predict and compute 0-1 loss
        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        train_loss = np.mean(y_pred_train != y_train)
        test_loss = np.mean(y_pred_test != y_test)

        print(f"\nTraining 0-1 Loss: {train_loss:.4f}")
        print(f"Test 0-1 Loss:     {test_loss:.4f}")

        return train_loss, test_loss

# Regularized logistic classification
class RegularizedLogistic:
    def __init__(self, reg_params, n_iterations, k_folds=5, random_state=42):
        """
        Initializes the Logistic Regression classifier.

        Args:
            reg_params (list): List of candidate values for the regularization parameter λ.
            n_iterations (list): List of candidate values for the number of training iterations.
            k_folds (int): Number of folds for k-fold cross-validation (default is 5).
            random_state (int): Seed for reproducibility (default is 42).
        """
        self.reg_params = reg_params
        self.n_iterations = n_iterations
        self.k_folds = k_folds
        self.random_state = random_state

        self.best_reg = None        # Best λ selected by cross-validation
        self.best_n_iter = None     # Best number of iterations selected by cross-validation
        self.weights = None         # Final weight vector after training

    def _ensure_numpy(self, *arrays):
        """
        Converts all inputs to NumPy arrays.
        """
        return [np.asarray(arr) for arr in arrays]

    def _logistic_gradient(self, w, x_i, y_i):
        """
        Computes the gradient of the logistic loss for a single sample.

        Args:
            w (np.ndarray): Current weight vector.
            x_i (np.ndarray): Feature vector of one sample.
            y_i (int): Label of the sample (+1 or -1).

        Returns:
            np.ndarray: Gradient vector.
        """
        z = np.clip(y_i * np.dot(w, x_i), -100, 100)  # prevent numerical overflow
        return -y_i * x_i / (1 + np.exp(z))

    def train(self, X, y, reg, n_iter, verbose=True):
        """
        Trains the model using stochastic gradient descent with logistic loss.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Label vector with values in {-1, +1}.
            reg (float): Regularization parameter λ.
            n_iter (int): Number of SGD iterations.
            verbose (bool): Whether to print progress messages.

        Returns:
            np.ndarray: Averaged weight vector over all iterations.
        """
        X, y = self._ensure_numpy(X, y)
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        weight_sum = np.zeros(n_features)

        for t in range(1, n_iter + 1):
            i = np.random.randint(0, n_samples)
            x_i, y_i = X[i], y[i]
            eta = 1.0 / (reg * t)

            grad = self._logistic_gradient(w, x_i, y_i)
            w = (1 - eta * reg) * w - eta * grad
            weight_sum += w

        average_weights = weight_sum / n_iter

        if verbose:
            print(f"Finished training with λ = {reg}, iterations = {n_iter}")
        return average_weights

    def predict(self, X, weights=None):
        """
        Predicts binary labels using the trained model.

        Args:
            X (np.ndarray): Input feature matrix.
            weights (np.ndarray, optional): Custom weight vector (default is self.weights).

        Returns:
            np.ndarray: Predicted labels in {-1, +1}.
        """
        X = np.asarray(X)
        if weights is None:
            weights = self.weights
        return np.sign(np.dot(X, weights))

    def cross_validate(self, X, y):
        """
        Performs k-fold cross-validation to select the best regularization and iteration settings.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Label vector.
        """
        X, y = self._ensure_numpy(X, y)
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))
        fold_size = len(X) // self.k_folds

        best_loss = float("inf")

        for reg in self.reg_params:
            for n_iter in self.n_iterations:
                fold_losses = []

                for i in range(self.k_folds):
                    start = i * fold_size
                    end = (i + 1) * fold_size if i < self.k_folds - 1 else len(X)
                    val_idx = indices[start:end]
                    train_idx = np.concatenate([indices[:start], indices[end:]])

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    weights = self.train(X_train, y_train, reg, n_iter, verbose=False)
                    y_pred = self.predict(X_val, weights)
                    loss = np.mean(y_pred != y_val)
                    fold_losses.append(loss)

                avg_loss = np.mean(fold_losses)
                print(f"λ = {reg}, iterations = {n_iter}, CV 0-1 Loss = {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.best_reg = reg
                    self.best_n_iter = n_iter

        print(f"\nBest hyperparameters: λ = {self.best_reg}, iterations = {self.best_n_iter} (CV 0-1 Loss = {best_loss:.4f})")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Trains the model with optimal hyperparameters and evaluates it on train and test sets.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
            X_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): Test labels.

        Returns:
            tuple: (training_loss, test_loss)
        """
        X_train, y_train, X_test, y_test = self._ensure_numpy(X_train, y_train, X_test, y_test)

        if self.best_reg is None or self.best_n_iter is None:
            raise ValueError("Run cross_validate() before evaluate().")

        self.weights = self.train(X_train, y_train, self.best_reg, self.best_n_iter)

        y_pred_train = self.predict(X_train)
        y_pred_test = self.predict(X_test)

        train_loss = np.mean(y_pred_train != y_train)
        test_loss = np.mean(y_pred_test != y_test)

        print(f"\nTraining 0-1 Loss: {train_loss:.4f}")
        print(f"Test 0-1 Loss:     {test_loss:.4f}")

        return train_loss, test_loss
    
# Polynomial feature expansion
class PolynomialFeatureExpansion:
    """
    Expands features to second-degree polynomial terms and generates corresponding feature names.
    """

    def __init__(self):
        self.original_feature_names = None
        self.expanded_feature_names = None

    def expand_features(self, X):
        """
        Applies polynomial expansion (degree 2) to the input matrix X.

        Args:
            X (pd.DataFrame or np.ndarray): Input feature matrix

        Returns:
            np.ndarray: Expanded feature matrix
        """
        if isinstance(X, pd.DataFrame):
            self.original_feature_names = X.columns.tolist()
            X = X.values
        else:
            n_features = X.shape[1]
            self.original_feature_names = [f"x{i+1}" for i in range(n_features)]

        # Generate feature names
        self._generate_feature_names(self.original_feature_names)

        # Apply expansion
        return self._expand_polynomial(X)

    def _expand_polynomial(self, X):
        """
        Internal method to compute polynomial features.

        Args:
            X (np.ndarray): Input feature matrix

        Returns:
            np.ndarray: Expanded feature matrix
        """
        n_samples, n_features = X.shape
        expanded = []

        for i in range(n_features):
            expanded.append(X[:, i])               # linear term
            expanded.append(X[:, i] ** 2)          # squared term

            for j in range(i + 1, n_features):
                expanded.append(X[:, i] * X[:, j])  # interaction term

        return np.vstack(expanded).T

    def _generate_feature_names(self, feature_names):
        """
        Generates names for the polynomial-expanded features.

        Args:
            feature_names (list of str): Names of the original features
        """
        self.expanded_feature_names = []

        for i, f1 in enumerate(feature_names):
            self.expanded_feature_names.append(f1)         # linear
            self.expanded_feature_names.append(f"{f1}^2")  # squared

            for j in range(i + 1, len(feature_names)):
                self.expanded_feature_names.append(f"{f1}*{feature_names[j]}")  # interaction

    def describe_expansion(self, X, X_expanded):
        """
        Prints the shape and feature names after expansion.

        Args:
            X (np.ndarray or pd.DataFrame): Original feature matrix
            X_expanded (np.ndarray): Expanded feature matrix
        """
        print(f"\nOriginal shape: {X.shape}")
        print(f"Expanded shape: {X_expanded.shape}")
        print("\nExpanded features:")
        print(self.expanded_feature_names)


class ModelWeightComparer:
    def __init__(self):
        self.features = None                # Linear feature names
        self.poly_features = None           # Polynomial-expanded feature names
        self.weight_df = None
        self.poly_weight_df = None

    def generate_latex_table(self, perceptron_weights, pegasos_weights, logistic_weights, dataframe):
        """
        Creates a LaTeX table for linear model weights.

        Args:
            perceptron_weights (np.ndarray)
            pegasos_weights (np.ndarray)
            logistic_weights (np.ndarray)
            dataframe (pd.DataFrame): original input dataframe
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Expected a Pandas DataFrame.")

        self.features = dataframe.columns.tolist()
        n = len(self.features)

        for weights, name in zip(
            [perceptron_weights, pegasos_weights, logistic_weights],
            ["perceptron_weights", "pegasos_weights", "logistic_weights"]
        ):
            if len(weights) != n:
                raise ValueError(f"{name} must have {n} weights, found {len(weights)}.")

        self.weight_df = pd.DataFrame({
            'Feature': self.features,
            'Perceptron': perceptron_weights,
            'Pegasos': pegasos_weights,
            'Logistic': logistic_weights
        }).set_index('Feature')

        latex_output = self.weight_df.to_latex(index=True)
        with open('weights_linear.tex', 'w') as f:
            f.write(latex_output)

        return self.weight_df

    def plot_weight_comparison(self, perceptron_weights, pegasos_weights, logistic_weights):
        """
        Plots weight comparison for linear features.
        """
        if self.features is None:
            raise RuntimeError("Run generate_latex_table() first.")

        plt.figure(figsize=(12, 8))
        plt.plot(self.features, perceptron_weights, label='Perceptron', marker='o')
        plt.plot(self.features, pegasos_weights, label='Pegasos', marker='s')
        plt.plot(self.features, logistic_weights, label='Logistic', marker='^')
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Weights")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def set_polynomial_feature_names(self, feature_names):
        """
        Stores the list of polynomial-expanded feature names.

        Args:
            feature_names (list of str)
        """
        self.poly_features = feature_names

    def generate_polynomial_latex_table(self, perceptron_weights_poly, pegasos_weights_poly, logistic_weights_poly):
        """
        Creates a LaTeX table for polynomial-expanded model weights.

        Args:
            perceptron_weights_poly (np.ndarray)
            pegasos_weights_poly (np.ndarray)
            logistic_weights_poly (np.ndarray)
        """
        if self.poly_features is None:
            raise RuntimeError("Polynomial feature names not set. Use set_polynomial_feature_names().")

        n = len(self.poly_features)

        for weights, name in zip(
            [perceptron_weights_poly, pegasos_weights_poly, logistic_weights_poly],
            ["perceptron_weights_poly", "pegasos_weights_poly", "logistic_weights_poly"]
        ):
            if len(weights) != n:
                raise ValueError(f"{name} must have {n} weights, found {len(weights)}.")

        self.poly_weight_df = pd.DataFrame({
            'Feature': self.poly_features,
            'Perceptron': perceptron_weights_poly,
            'Pegasos': pegasos_weights_poly,
            'Logistic': logistic_weights_poly
        }).set_index('Feature')

        latex_output = self.poly_weight_df.to_latex(index=True)
        with open('weights_polynomial.tex', 'w') as f:
            f.write(latex_output)

        return self.poly_weight_df

# Kernel methods
class KernelizedPerceptron:
    def __init__(self, kernel_type='rbf', kernel_param_grid=None, max_epochs_list=None, k_folds=5, random_state=42):
        """
        Initialize the Kernelized Perceptron with parameter grids and settings.

        Args:
            kernel_type (str): 'rbf' for Gaussian kernel, 'poly' for Polynomial kernel.
            kernel_param_grid (dict): Parameters to test for the kernel function.
                - For 'rbf': {'sigma': [...]}
                - For 'poly': {'degree': [...]}
            max_epochs_list (list): List of values for max training epochs to test.
            k_folds (int): Number of folds for cross-validation.
            random_state (int): Seed for reproducibility.
        """
        self.kernel_type = kernel_type
        self.kernel_param_grid = kernel_param_grid
        self.max_epochs_list = max_epochs_list 
        self.k_folds = k_folds
        self.random_state = random_state
        self.best_params = None
        self.alpha = None  # Dual weights
        self.X_train = None  # Stored training set for prediction
        self.K_train = None  # Precomputed kernel matrix

    def _compute_kernel_matrix(self, X1, X2, params):
        """
        Compute the full kernel matrix between two sets of vectors.

        Args:
            X1, X2 (ndarray): Input matrices.
            params (dict): Kernel parameters.

        Returns:
            ndarray: Kernel matrix.
        """
        if self.kernel_type == 'rbf':
            sigma = params['sigma']
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            dists = X1_sq - 2 * np.dot(X1, X2.T) + X2_sq
            return np.exp(-dists / (2 * sigma ** 2))
        elif self.kernel_type == 'poly':
            degree = params['degree']
            return (1 + np.dot(X1, X2.T)) ** degree
        else:
            raise ValueError("Unsupported kernel type")
    
    def train(self, X, y, epochs, kernel_params, verbose=False):
        """
        Train the Kernel Perceptron using precomputed kernel matrix.

        Args:
            X (ndarray): Training data.
            y (ndarray): Training labels.
            epochs (int): Number of training epochs.
            kernel_params (dict): Kernel parameters.
        """
        self.X_train = X
        self.y_train = y
        n_samples = len(X)
        self.alpha = np.zeros(n_samples)
        self.K_train = self._compute_kernel_matrix(X, X, kernel_params)

        converged = False

        for epoch in range(epochs):
            updates = 0
            for i in range(n_samples):
                prediction = np.sign(np.sum(self.alpha * y * self.K_train[:, i]))
                if prediction != y[i]:
                    self.alpha[i] += 1
                    updates += 1

            if updates == 0:  # Check for convergence
                if verbose:
                    print(f"Converged after {epoch + 1} epochs.")
                converged = True
                break

        if not converged and verbose:
            print("Did not converge within the maximum number of epochs.")

    def predict(self, X, kernel_params):
        """
        Predict the labels for new input data using the trained model.

        Args:
            X (ndarray): Test data.
            kernel_params (dict): Kernel parameters.

        Returns:
            ndarray: Predicted labels.
        """
        K_test = self._compute_kernel_matrix(self.X_train, X, kernel_params)
        return np.sign(np.dot(self.alpha * self.y_train, K_test))

    def cross_validate(self, X, y):
        """
        Perform k-fold cross-validation to find the best hyperparameters.

        Args:
            X (ndarray): Input data.
            y (ndarray): Target labels.
        """
        np.random.seed(self.random_state)
        X, y = np.asarray(X), np.asarray(y)
        indices = np.random.permutation(len(X))
        fold_size = len(X) // self.k_folds

        best_loss = float('inf')
        best_params = {}

        # Build hyperparameter grid
        if self.kernel_type == 'rbf':
            param_grid = list(product(self.kernel_param_grid.get('sigma', [1.0]), self.max_epochs_list))
            param_names = ['sigma', 'epochs']
        elif self.kernel_type == 'poly':
            param_grid = list(product(self.kernel_param_grid.get('degree', [2, 3]), self.max_epochs_list))
            param_names = ['degree', 'epochs']
        else:
            raise ValueError("Unsupported kernel type")

        for params in param_grid:
            param_dict = dict(zip(param_names, params))
            fold_losses = []

            for k in range(self.k_folds):
                start, end = k * fold_size, (k + 1) * fold_size if k < self.k_folds - 1 else len(X)
                val_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                self.train(X_train, y_train, param_dict['epochs'], param_dict, verbose=False)
                self.y_train = y_train  # Needed for prediction
                y_pred = self.predict(X_val, param_dict)
                loss = np.mean(y_pred != y_val)
                fold_losses.append(loss)

            avg_loss = np.mean(fold_losses)
            print(f"{param_names[0]} = {params[0]}, epochs = {params[1]}, CV 0-1 Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = param_dict

        self.best_params = best_params
        print(f"\nBest parameters: {self.best_params} (CV 0-1 Loss: {best_loss:.4f})")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Train the model on the full training set and evaluate on both training and test sets.

        Args:
            X_train, y_train: Training data and labels.
            X_test, y_test: Test data and labels.

        Returns:
            Tuple: Training and test loss (0-1 loss).
        """
        if self.best_params is None:
            raise ValueError("Run cross_validate() before evaluate().")

        self.train(X_train, y_train, self.best_params['epochs'], self.best_params, verbose=True)
        self.y_train = y_train  # Needed for prediction
        y_pred_train = self.predict(X_train, self.best_params)
        y_pred_test = self.predict(X_test, self.best_params)

        train_loss = np.mean(y_pred_train != y_train)
        test_loss = np.mean(y_pred_test != y_test)

        print(f"\nTraining 0-1 Loss: {train_loss:.4f}")
        print(f"Test 0-1 Loss:     {test_loss:.4f}")

        return train_loss, test_loss
    
class KernelizedPegasos:
    def __init__(self, kernel_type='rbf', kernel_param_grid=None, reg_params=None,
                 n_iterations_list=None, k_folds=5, random_state=42):
        """
        Initialize the Kernelized Pegasos SVM.

        Parameters:
        - kernel_type: 'rbf' (Gaussian) or 'poly' (polynomial)
        - kernel_param_grid: dict of kernel parameters.
        - reg_params: list of lambda values to test
        - n_iterations_list: list of iteration counts to test
        - k_folds: number of folds for cross-validation
        - random_state: seed for reproducibility
        """
        self.kernel_type = kernel_type
        self.kernel_param_grid = kernel_param_grid
        self.reg_params = reg_params
        self.n_iterations_list = n_iterations_list
        self.k_folds = k_folds
        self.random_state = random_state

        self.best_params = None
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None

    def _kernel(self, X1, X2, params):
        """Compute the kernel matrix between two sets of vectors."""
        if self.kernel_type == 'rbf':
            sigma = params['sigma']
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            dists = X1_sq - 2 * np.dot(X1, X2.T) + X2_sq
            return np.exp(-dists / (2 * sigma**2))
        elif self.kernel_type == 'poly':
            degree = params['degree']
            return (1 + np.dot(X1, X2.T)) ** degree
        else:
            raise ValueError("Unsupported kernel type")

    def train(self, X, y, lambda_, n_iterations, kernel_params):
        """
        Train the Pegasos algorithm with the chosen kernel.

        Parameters:
        - X: training data
        - y: training labels
        - lambda_: regularization parameter
        - n_iterations: number of iterations
        - kernel_params: dict of kernel parameters
        """
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        K = self._kernel(X, X, kernel_params)

        for t in range(1, n_iterations + 1):
            i = np.random.randint(0, n_samples)
            margin = (1 / (lambda_ * t)) * np.sum(alpha * y * K[i])
            if y[i] * margin < 1:
                alpha[i] += 1

        self.alpha = alpha
        self.support_vectors = X
        self.support_labels = y

    def predict(self, X, kernel_params):
        """
        Predict the class labels for a given input using the trained model.

        Parameters:
        - X: input samples
        - kernel_params: dict of kernel parameters

        Returns:
        - predictions: array of predicted labels
        """
        K = self._kernel(X, self.support_vectors, kernel_params)
        predictions = np.sign(np.dot(K, self.alpha * self.support_labels))
        return predictions

    def cross_validate(self, X, y):
        """
        Perform k-fold cross-validation to select the best hyperparameters.
        """
        np.random.seed(self.random_state)
        X, y = np.asarray(X), np.asarray(y)
        indices = np.random.permutation(len(X))
        fold_size = len(X) // self.k_folds

        best_loss = float('inf')
        best_params = {}

        if self.kernel_type == 'rbf':
            param_grid = list(product(self.kernel_param_grid.get('sigma', [1.0]),
                                      self.reg_params, self.n_iterations_list))
            param_names = ['sigma', 'lambda', 'n_iter']
        elif self.kernel_type == 'poly':
            param_grid = list(product(self.kernel_param_grid.get('degree', [2, 3]),
                                      self.reg_params, self.n_iterations_list))
            param_names = ['degree', 'lambda', 'n_iter']
        else:
            raise ValueError("Unsupported kernel type")

        for params in param_grid:
            param_dict = dict(zip(param_names, params))
            fold_losses = []

            for k in range(self.k_folds):
                start = k * fold_size
                end = (k + 1) * fold_size if k < self.k_folds - 1 else len(X)
                val_idx = indices[start:end]
                train_idx = np.concatenate([indices[:start], indices[end:]])

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                self.train(X_train, y_train, param_dict['lambda'], param_dict['n_iter'], param_dict)
                y_pred = self.predict(X_val, param_dict)
                loss = np.mean(y_pred != y_val)
                fold_losses.append(loss)

            avg_loss = np.mean(fold_losses)
            print(f"{param_names[0]} = {params[0]}, lambda = {params[1]}, n_iter = {params[2]}, CV 0-1 Loss = {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = param_dict

        self.best_params = best_params
        print(f"\nBest parameters: {self.best_params} (CV 0-1 Loss: {best_loss:.4f})")

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate the model on both training and test datasets.

        Returns:
            tuple: Training loss and test loss.
        """
        if self.best_params is None:
            raise ValueError("Run cross_validate() before evaluate().")

        self.train(X_train, y_train,
                   self.best_params['lambda'],
                   self.best_params['n_iter'],
                   self.best_params)

        y_pred_train = self.predict(X_train, self.best_params)
        y_pred_test = self.predict(X_test, self.best_params)

        train_loss = np.mean(y_pred_train != y_train)
        test_loss = np.mean(y_pred_test != y_test)

        print(f"\nTraining 0-1 Loss: {train_loss:.4f}")
        print(f"Test 0-1 Loss:     {test_loss:.4f}")

        return train_loss, test_loss

