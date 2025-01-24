from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import Utils
from kneed import KneeLocator

ALLOWED_ALGORITHMS = ['kmeans'] # i initially planned to use more clustering algorithms, such as dbscan, i swear

class Clusterer:
    def __init__(self):
        self._algorithm = None

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        if algorithm not in ALLOWED_ALGORITHMS:
            raise ValueError(f"Invalid algorithm. Must be one of {ALLOWED_ALGORITHMS}")
        self._algorithm = algorithm

    def fit_predict(self, X, **kwargs):
        """
        Performs a clustering operation on the input data using the specified algorithm, currently only 'kmeans' is supported.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to be clustered.
        **kwargs : dict, optional
            Additional keyword arguments for the chosen clustering algorithm.
            - For 'dbscan': eps : float, min_samples : int
            - For 'kmeans': n_clusters : int

        Returns
        -------
        tuple of (numpy.ndarray, float)
            If the 'kmeans' algorithm is used, returns a tuple containing:
            - labels (numpy.ndarray): The cluster labels for each data point.
            - inertia (float): The sum of squared distances of samples to their closest cluster center.
        """
        if self.algorithm == 'dbscan': # it is not allowed hence you won't ever enter here, but i'll keep it for an eventual implementation
            db = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples']).fit(X)
            return db.labels_
        elif self.algorithm == 'kmeans':
            kmeans = KMeans(n_clusters=kwargs['n_clusters'], init='k-means++').fit(X)
            return kmeans.labels_, kmeans.inertia_
    
    def labels_to_raster(self, labels: np.ndarray, shape: tuple) -> np.ndarray:
        """
        Reshapes a one-dimensional array of labels into a rasterized array with the specified shape.

        Args:
            labels (np.ndarray): One-dimensional array of label values.
            shape (tuple): Target shape for the rasterized array (e.g., height, width).

        Returns:
            np.ndarray: Rasterized array reshaped to the specified dimensions.
        """
        return labels.reshape(shape)
    
    
    def elbow_method(self, X, max_clusters: int, save_path: str = None) -> int:
        """
        Determines the optimal number of clusters using the elbow method.
        Args:
            X (array-like): The dataset to be clustered.
            max_clusters (int): The maximum number of clusters to evaluate.
            save_path (str, optional): The file path to save the elbow plot. Defaults to None.
        Returns:
            int: The optimal number of clusters identified by the elbow method.
        """
        wcss = [] #  within-cluster sums of squares
        cluster_range = range(2, max_clusters + 1)
        for i in tqdm(cluster_range):
            _, inertia = self.fit_predict(X, n_clusters = i)
            wcss.append(inertia)
        
        knee_locator = KneeLocator(cluster_range, wcss, curve="convex", direction="decreasing")
        optimal_k = knee_locator.knee
        
        if save_path is not None:
            plt.plot(cluster_range, wcss, label='WCSS')
            plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', label='Optimal k', color='red')
            plt.legend()
            plt.grid()
            plt.title('Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig(save_path)
            plt.close()
        return optimal_k
    
    def clusters_summary(self,
                         X: np.ndarray,
                         labels: np.ndarray,
                         save_path:str = None) -> pd.DataFrame:
        # combined_bands expected shape (rows, cols, depth)
        if X.ndim == 4:
            # 4D (years, height, width, depth)
            depth = X.shape[-1]
            reshaped_bands = X.reshape(-1, depth)  # Flatten to 2D for calculations
            if labels.ndim > 1:
                labels = labels.flatten() # Flatten labels to 1D if needed
        elif X.ndim == 3:
            # 3D (height, width, depth)
            depth = X.shape[-1]
            reshaped_bands = X.reshape(-1, depth)  # Flatten to 2D for calculations
            if labels.ndim > 1:
                labels = labels.flatten() # Flatten labels to 1D if needed
        elif X.ndim == 2:
            # 2D (n_samples, n_features)
            depth = X.shape[1]
            reshaped_bands = X
            # Labels should already be 1D in this case
        else:
            raise ValueError("Invalid shape for combined_bands. Must be 2D, 3D, or 4D.")

        
        k = len(np.unique(labels))
        cluster_means = np.zeros((k, depth))          
        cluster_stds = np.zeros((k, depth))

        for i in range(k):
            cluster_means[i] = np.mean(reshaped_bands[labels == i], axis=0)
            cluster_stds[i] = np.std(reshaped_bands[labels == i], axis=0)

        df = pd.DataFrame(np.hstack((cluster_means, cluster_stds)),
                          columns=["NDWI Mean",
                                   "MSI Mean", 
                                   "NDMI Mean", 
                                   "MSAVI2 Mean", 
                                   "NDWI Std", 
                                   "MSI Std", 
                                   "NDMI Std", 
                                   "MSAVI2 Std"])

        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df
    
    def sort_labels_by_ndwi_ndmi(self, combined_bands, labels):       
        """
        Sorts cluster labels based on NDMI (Normalized Difference Moisture Index) and NDWI (Normalized Difference Water Index) metrics.
        This function calculates the mean NDWI, MSI, NDMI, and MSAVI2 values for each cluster, sorts the clusters
        by ascending NDMI and NDWI means, and then reassigns the labels accordingly. An additional check is performed
        to detect any cluster with NaN values (and sets them to zero), and a zero-valued cluster, if present, is moved
        to the first position.
        Parameters:
            combined_bands (numpy.ndarray):
                The array containing the spectral indices to be clustered. Accepts 2D, 3D, or 4D arrays:
                - 2D: (n_samples, n_features)
                - 3D: (height, width, depth)
                - 4D: (years, height, width, depth)
            labels (numpy.ndarray):
                The cluster labels corresponding to each element in the `combined_bands` array.
                Can be 1D or higher dimensions matching the shape of `combined_bands` in its spatial/temporal axes.
        Returns:
            numpy.ndarray:
                A 1D or multi-dimensional array of the same shape as `labels` (except for any flattening
                done internally), re-labeled such that clusters are sorted first by ascending NDMI mean
                values, then by ascending NDWI mean values, with any zero-valued cluster moved to the first position.
        Raises:
            ValueError:
                If `combined_bands` has a dimensionality outside of the accepted 2D, 3D, or 4D range.
        """
        if combined_bands.ndim == 4:
            # 4D (years, height, width, depth)
            depth = combined_bands.shape[-1]
            reshaped_bands = combined_bands.reshape(-1, depth)  # Flatten to 2D
            if labels.ndim > 1:
                labels = labels.flatten()
        elif combined_bands.ndim == 3:
            # 3D (height, width, depth)
            depth = combined_bands.shape[-1]
            reshaped_bands = combined_bands.reshape(-1, depth)  # Flatten to 2D
            if labels.ndim > 1:
                labels = labels.flatten()
        elif combined_bands.ndim == 2:
            # 2D (n_samples, n_features)
            depth = combined_bands.shape[1]
            reshaped_bands = combined_bands
        else:
            raise ValueError(f"Invalid shape for combined_bands. Must be 2D, 3D, or 4D. Found {combined_bands.ndim}D.")
        
        k = len(np.unique(labels)) # it works, but perhaps i should use np.max(labels), to cover the case where there is a missing cluster
        cluster_means = np.zeros((k, depth))      
        
        for i in range(k):
            cluster_means[i] = np.mean(reshaped_bands[labels == i], axis=0)

        df = pd.DataFrame(cluster_means, columns=["NDWI Mean", "MSI Mean", "NDMI Mean", "MSAVI2 Mean"]) # it's a bit hardcoded, but it's fine for now

        # check if there's a row of nan
        nan_rows = df.isnull().any(axis=1)
        if nan_rows.any():
            df = df.fillna(0)
        
        # sort by NDMI (ascending) and then NDWI (ascending)
        df = df.sort_values(by=["NDMI Mean", "NDWI Mean"], ascending=[True, True])

        sorted_indexes = np.array(df.index)

        # move the background class (all zeros cluster) to the top
        zero_index = np.where(np.all(df.to_numpy() == 0, axis=1))
        if len(zero_index[0]) > 0:
            zero_index = zero_index[0][0]
            sorted_indexes[0], sorted_indexes[zero_index] = sorted_indexes[zero_index], sorted_indexes[0]

        sorted_labels = np.zeros_like(labels)
        for i, idx in enumerate(sorted_indexes):
            sorted_labels[labels == idx] = i

        return sorted_labels
    
    def inter_cluster_distance_mat(self, X, labels, n_clusters):
        """
        Compute the pairwise distances between cluster centers using Euclidean distance.
        Parameters:
            X (numpy.ndarray): 
                The input data array of shape (N, D) or shape compatible 
                with reshaping to (N, D), where N is the number of samples 
                and D is the number of features.
            labels (numpy.ndarray): 
                An array of integer cluster labels of shape (N,), 
                indicating the cluster assignments for each sample.
            n_clusters (int): 
                The number of distinct clusters.
        Returns:
            numpy.ndarray:
                A square symmetric positive semi-definite matrix of shape (n_clusters, n_clusters), 
                representing the Euclidean distance between each pair 
                of cluster centers.
        """
        X = X.reshape(-1, X.shape[-1])
        if labels.ndim > 1:
            labels = labels.flatten()
        
        cluster_centers = np.zeros((n_clusters, X.shape[-1]))
        for i in range(n_clusters):
            cluster_centers[i] = np.mean(X[labels == i], axis=0)        
        
        inter_cluster_distance = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                inter_cluster_distance[i, j] = np.linalg.norm(cluster_centers[i] - cluster_centers[j]) # euclidean distance
        return inter_cluster_distance
    
    def plot_labels_raster(self,
                           labels: np.ndarray,
                           title: str,
                           n_clusters: int,
                           cmap,
                           save_path: str = None) -> None:
        """
        Plot a color-encoded raster representation of cluster labels.

        Parameters
        ----------
        labels : np.ndarray
            A 2D array of cluster labels to plot.
        title : str
            Title for the plot.
        n_clusters : int
            Number of clusters to display in the colormap.
        cmap
            Colormap used to color the labels.
        save_path : str, optional
            File path to save the plot. If not provided, the plot is only shown and
            not saved.

        Returns
        -------
        None
            This function does not return a value. It either displays or saves
            the generated plot.
        """

        plt.figure(figsize=(12, 12))
        plt.imshow(labels, cmap=cmap, vmin=0, vmax=n_clusters)
        plt.title(title)
        plt.axis("off")
        plt.colorbar(ticks=np.arange(0, n_clusters+1))
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
        
        
    def plot_labels_histogram(self,
                              labels: np.ndarray, 
                              title: str, 
                              n_clusters: int,
                              save_path: str = None) -> None:
        """
        Plots a histogram displaying the distribution of cluster labels.
        This method computes the normalized distribution of the provided cluster
        labels and creates a bar plot showing the frequency of each cluster.
        Parameters:
            labels (np.ndarray):
                A NumPy array containing the cluster labels for the data points.
            title (str):
                The title of the plot.
            n_clusters (int):
                The number of clusters.
            save_path (str, optional):
                If provided, the plot is saved to this file path. Defaults to None.
        Returns:
            None
        """
        
        cluster_distrbution = self.cluster_distribution(labels, n_clusters) # already normalized
        
        plt.figure(figsize=(12, 10))
        plt.bar(range(1, n_clusters+1), cluster_distrbution, color="skyblue", alpha=1, width=0.7)
        plt.title(title)
        plt.xticks(range(1, n_clusters+1))
        plt.xlabel("Cluster")
        plt.ylabel("Frequency")
        plt.grid()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()
        
    def cluster_distribution(self, labels: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Computes the normalized distribution of cluster labels, removing any NaN values.
        Parameters
        ----------
        labels : np.ndarray
            An array of cluster labels, which may contain NaN values.
        n_clusters : int
            The total number of clusters expected.
        Returns
        -------
        np.ndarray
            A 1D array of length n_clusters representing the normalized distribution of
            each cluster label, where the sum of all elements is 1.0.
        """
        # remove nan values, if any
        labels = labels[~np.isnan(labels)]
        cluster_distribution = np.bincount(labels.flatten(), minlength=n_clusters)
        
        # normalize the distribution
        cluster_distribution = cluster_distribution / cluster_distribution.sum()
        return cluster_distribution