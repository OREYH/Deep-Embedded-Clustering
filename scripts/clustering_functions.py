# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:45:45 2022

@author: Jip de Kok

This file contains custom functions for the clustering functionality.
"""
# import packages
import numpy as np
import torch

# Set random seed for reproducibility
np.random.seed(5192)
torch.manual_seed(5192)

# import packages
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedKFold
from metrics import calculate_metrics
import pickle
import hdbscan
import time
import warnings
import gower
import scipy
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import local functions
from xvae import xvae, load_xvae_model
from data_functions import scale_data
from pytorch_dec import train_dec


def target_distribution(q):
    '''
    Computes target distribution from soft labels

    Parameters
    ----------
    q : ndarray
        Array of soft labels.

    Returns
    -------
    ndarray
        Array of target distribution.

    '''
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def dec_cluster(feature_array, y, n_classes, seed=5192,
                return_extra=False, hidden_dim=64, latent_dim=8,
                batch_size=256, pretrain_epochs=50, dec_epochs=100,
                lr=1e-3, device="cpu"):
    """Run DEC on the given feature array using PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    preds, Z, model, dec_layer = train_dec(
        feature_array,
        n_clusters=n_classes,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        batch_size=batch_size,
        pretrain_epochs=pretrain_epochs,
        dec_epochs=dec_epochs,
        lr=lr,
        device=device,
    )

    with torch.no_grad():
        q = dec_layer(torch.tensor(Z, dtype=torch.float32)).numpy()
    centroids = dec_layer.clusters.cpu().numpy()

    if return_extra:
        return preds, q, centroids, (model, dec_layer)
    return preds, q

def dec_vae_cluster(*args, **kwargs):
    """PyTorch VAE-based DEC is not implemented."""
    raise NotImplementedError("VAE-based DEC is not implemented in this PyTorch version.")


def cluster_mlp_autoencoder(feature_array, n_classes, y=None, neurons_h=64,
                            neurons_e=8, epochs=500, batch_size=64, seed=5192,
                            check_stability=False, stability_check_it=100,
                            order_by_size=False,
                            return_extra=False, verbose=0):
    """Train a DEC model using PyTorch."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    preds, Z, model, dec_layer = train_dec(
        feature_array,
        n_clusters=n_classes,
        hidden_dim=neurons_h,
        latent_dim=neurons_e,
        batch_size=batch_size,
        pretrain_epochs=epochs,
        dec_epochs=epochs,
        device="cpu",
    )

    with torch.no_grad():
        q = dec_layer(torch.tensor(Z, dtype=torch.float32)).numpy()
    centroids = dec_layer.clusters.cpu().numpy()

    if order_by_size:
        counts = np.bincount(preds)
        mapping = np.argsort(-counts)
        preds = np.take(mapping, preds)

    if return_extra:
        encoder = model.encoder
        return preds, q, Z, centroids, model, encoder, encoder
    return preds, q


def cluster_mlp_vae(feature_array, n_classes, y=None, ds1 = 52,ds2 = 5,
                    ds12 = 48, ls = 32, act = "elu", dropout = 0.2,
                    distance = "mmd", beta = 25, batch_size=64, seed=5192,
                    weighted = True,
                    check_stability = False, stability_check_it = 100,
                    order_by_size = False, return_extra = False, verbose = 0):
    """X-DEC using PyTorch is not implemented."""
   
    raise NotImplementedError("VAE-based clustering is not implemented in this PyTorch version.")

def compute_cluster_stability(feature_array, n_cluster, k = 10, rep=5,
                              neurons_h=64, neurons_e=8, epochs=500,
                              batch_size=64, ds1 = 52,ds2 = 5, ds12 = 48,
                              ls = 32, act = "elu", dropout = 0.2,
                              distance = "mmd", beta = 25, mapper = "centroids",
                              retrain_autoencoder = True, weighted = True,
                              majority_vote = False,
                              disable_resampling = False,
                              seed=5192, return_extra = False,
                              model_type = "autoencoder", save= True,
                              save_name = "stability_check"):
    '''
    Calculates cluster stability by retraining a clustering model many times.
    Can be performed for DEC as well as X-DEC.

    Parameters
    ----------
    feature_array : ndarray
        2-dimensional array where rows represent samples and columns are
        the input variables for the clustering model.
    n_cluster : int
        number of clusters to identify.
    k : int, optional
        Number of folds in k-repeated cross validation. The default is 10.
    rep : int, optional
        Number of repeats in k-fold repeated cross validation. The default is 5.
    neurons_h : int, optional
        Number of neurons in the hidden layer in DEC. Only used if model_type
        is "autoencoder". The default is 64.
    neurons_e : int, optional
        Number of neurons in the encoding layer in DEC. Only used if model_type
        is "autoencoder".. The default is 8.
    epochs : int, optional
        Number of epochs. The default is 500.
    batch_size : int, optional
        Batch size. The default is 64.
    ds1 : int, optional
        integer specifying the number of neurons in the first hidden layer of
        input set 1 for vae models. Only used if model_type="vae". The default
        is 52.
    ds2 : int, optional
        integer specifying the number of neurons in the first hidden layer of
        input set 2 for vae models. Only used if model_type="vae". The default
        is 5.
    ds12 : int, optional
        integer specifying the number of neurons in the hidden layer that joins
        input set 1 and 2 for vae models. Only used if model_type="vae". The
        default is 48.
    ls : int, optional
        integer specifying the number of neurons in the embedding layer, also
        known as the bottleneck layer, which stores the latent features. Only
        used if model_type="vae". The default is 32 for vae models.
     act : str, optional
         String indicating which activation function to use in the hidden
         layers. Only used if model_type="vae". The default is "elu".
     dropout : float, optional
         Dropout rate. Only used if model_type="vae". The default is 0.2.
     distance : str, optional
         Distance metric to use for regularisation in the objective function.
         Only used if model_type="vae". The default is "mmd".
     beta : float, optional
         The influence of the disentanglement factor. Only used if
         model_type="vae". The default is 25.
    mapper : str, optional
        String indicating how clusters should be mapped. For example, how 
        should the model determine which cluster from iteration two is equal
        to cluster one from the first iteration. Possible options are 
        "centroids", "overlap", and "jaccard". The default is "centroids".
    retrain_autoencoder : bool, optional
        Boolean indicating whether the autoencoder (or xvae) should be
        completely retrained in each iterations. The default is True.
    weighted : bool, optional
        Boolean indicating whether the influence of input set 1 and 2 should be
        scaled according to the number of values they contain. This can prevent
        sets with very few variables to be overrepresented. The default is True.
    majority_vote : bool, optional
        Boolean indicating whether the reference cluster for each sample to
        base its stability on is based on a majority vote of all its cluster
        assignments over all iterations. If False, per sample, the first cluter
        assignment will be used as reference cluster. The default is False.
    disable_resampling : bool, optional
        Whether to disable the whole resampling procedure. If True, 'k' and
        'rep' are ignored, and all samples are used during each iteration to
        determine baseline model stability. Total number of iteration will
        still be based on the product of 'k' and 'rep'. The default is False.
    seed : int, optional
        The random seed. The default is 5192.
    return_extra : bool, optional
        Boolean indicating whether to return additional variables. If set to 
        True, also Z, cluster_labels, centroids and y_pred will be returned.
        The default is False.
    model_type : str, optional
        String specifying what type of clustering model to run. You can specify
        "autoencoder" to run DEC with an MLP autoencoder, or "vae" to run
        X-DEC with an X-shape variational autoencoder. The default is "autoencoder".
    save : bool, optional
        Boolean indicating whether to locally save results. The default is True.
    save_name : str, optional
        String specifying where and under what name to save the results locally
        if save=True. The default is "stability_check".

    Returns
    -------
    cluster_stab pandas.DataFrame
        DataFrame containing the cluster-wise stabilities. Each row represents
        a cluster, and each column represents one iteration/subset.
    Z pandas.DataFrame
        DataFrame containing the latent features (columns) from the clustering
        model trained on the full dataset. Only returned if return_extra=True.
    cluster_labels : ndarray
        The cluster labels for each sample.Only returned if return_extra=True.
    centroids : ndarray, optional
        the centroids of the clusters in the Z-space. Only returned if
        return_extra=True.
    y_pred pandas.DataFrame
        DataFrame containing the cluster memberships for all samples (rows) for
        each iteration/subset (columns). Only returned if return_extra=True.
    model : Keras object
        The clustering model.
    encoder: Keras object, optional
        The encoder of clustering model. Only returned if return_extra=True.

    '''
    
    if type(feature_array) == list:
        if type(feature_array[0]) == pd.core.frame.DataFrame:
            warnings.warn("Data was supplied as DataFrame, converting it to a"\
                          "Numpy array.")
            feature_array[0] = np.array(feature_array[0])
            feature_array[1] = np.array(feature_array[1])
    elif type(feature_array) == pd.core.frame.DataFrame:
        warnings.warn("Data was supplied as DataFrame, converting it to a"\
                      "Numpy array.")
        feature_array = np.array(feature_array)
        
        
    # Run the clustering algorithm once to obtain the latent feature space Z, 
    # the centroid positions, ande the clustering model.
    if model_type == "autoencoder":
        cluster_labels, _, Z, centroids, model, encoder, encoder_unoptimised = cluster_mlp_autoencoder(
            feature_array, n_cluster, neurons_h = neurons_h, neurons_e = neurons_e,
            epochs = epochs, batch_size = batch_size, seed = seed,
            return_extra = True)
        # Initialise y_pred so it can store cluster labels for all resamples
        y_pred = pd.DataFrame(index = range(feature_array.shape[0]),
                          columns=range(k*rep))
    elif model_type == "vae":
        x = feature_array.copy()
        x1 = np.array(x[0])
        x2 = np.array(x[1])
        feature_array = feature_array[0]
        cluster_labels, _, Z, centroids, model, encoder, encoder_unoptimised = cluster_mlp_vae(
            x, n_cluster, ds1 = ds1,ds2 = ds2, ds12 = ds12,
            ls = ls, act = act, dropout = dropout, weighted = weighted,
            distance = distance, beta = beta, seed = seed,
            return_extra = True)
        # Initialise y_pred so it can store cluster labels for all resamples
        y_pred = pd.DataFrame(index = range(x1.shape[0]),
                          columns=range(k*rep))
        if retrain_autoencoder == False:
            encoder_unoptimised.save_model("temp_xvae_model_copy")
    
    
    # Run repeated K-fold cross validation
    i = 0
    rkf = RepeatedKFold(n_splits=k, n_repeats=rep, random_state=seed)
    for train_index, test_index in rkf.split(feature_array):
        if disable_resampling:
            train_index = np.arange(0, len(feature_array))
            test_index = np.arange(0, len(feature_array))
            
        # Store current time
        t = time.time()
        # Set current train and test set
        if model_type == "autoencoder":
            X_train, X_test = feature_array[train_index],feature_array[test_index]
            
            # Run cluster analysis
            y_pred_it, y_proba_it, Z_it, _, _, _, _ = cluster_mlp_autoencoder(
                X_train, n_cluster, neurons_h = neurons_h, neurons_e = neurons_e,
                epochs = epochs, batch_size = batch_size, seed = seed,
                return_extra = True)
        elif model_type == "vae":
            X_train = [x1[train_index], x2[train_index]]
            X_test = [x1[test_index], x2[test_index]]
            
            # Run cluster analysis
            if retrain_autoencoder:
                y_pred_it, y_proba_it, Z_it, _, _, _, _ = cluster_mlp_vae(
                    X_train, n_cluster, ds1 = ds1,ds2 = ds2, ds12 = ds12,
                    ls = ls, act = act, dropout = dropout, weighted = weighted,
                    distance = distance, beta = beta, seed = seed,
                    return_extra = True)
            else:
                encoder_copy = load_xvae_model("temp_xvae_model_copy")
                y_pred_it, y_proba_it, _, _ = dec_vae_cluster(
                    encoder_copy, X_train, y = None, n_classes = 6, n_init = 100,
                    maxiter = 1600, batch_size = 256, return_extra = True)
                Z_it = Z.copy()
        
        if mapper == "centroids":
            # Define cluster numbers based on distance to initial centroids
            y_pred_it = map_cluster_to_centroid(Z_it, y_pred_it, centroids)
        elif mapper == "overlap":
            # Define cluster numbers based on maximum sample overlap
            y_pred_it = map_cluster_to_overlap(y_pred_it, cluster_labels, train_index)
        elif not mapper == "jaccard":
            raise ValueError(f"'{mapper}' mapper not recognised! must be"\
                             "'centroids', 'overlap' or 'jaccard'.")
        
        # Store cluster prediction in y_pred
        y_pred.iloc[train_index,i] = y_pred_it
        i += 1
        
        elapsed = np.round((time.time() - t)/60)
        print(f'finished iteration {i}/{k*rep} took {elapsed} minutes')
    
    if mapper == "jaccard":
        # Compute jaccard score per cluster, and remap clusters on max jaccard
        y_pred, cluster_stab = jaccard_similarity(cluster_labels, y_pred)
        if majority_vote:
            y_pred, cluster_stab = jaccard_similarity(cluster_labels, y_pred,
                                                      True)
    else:
        # Compute cluster stability per sample
        cluster_stab = sample_cluster_stability(majority_vote, y_pred)
    
    if save:
        cluster_stab.to_csv(f"stats/{save_name}.csv")
        y_pred.to_csv(f"stats/{save_name}_y_pred.csv")
    
    if return_extra:
        return cluster_stab, Z, cluster_labels, centroids, y_pred, model, encoder
    return cluster_stab
    


def sample_cluster_stability(majority_vote, y_pred,
                             *args, average = False):
    '''
    This function computes the cluster stability of each sample given a set of
    predicted cluster labels.

    Parameters
    ----------
    *args : Pandas.Series
        multiple (at least 3) sets of cluster label predictions.
    average : boolean
        Whether to take the average of the stability across all samples.
    majority_vote : Boolean
        Whether 'true' cluster labels are based on a majority vote. If true 
        then also supply y_pred.
    y_pred : Series
        DataFrame containing the 'true' cluster labels. Required only if
        majority_vote=False.
        

    Returns
    -------
    pred_stability: Pandas.Series
        A Series containing a percentage per sample indicating how often a
        given sample was clustered into the cluster that it was clsutered into
        most often across all cluster predictions.

    '''
    
    # Put all cluster label predictions in one DataFrame
    pred_mat = pd.concat(args, axis = 1)
    
    # Define a final cluster label through majortiy voting
    if majority_vote == True:
        pred_final = pred_mat.mode(axis=1)[0].astype(int)
    else:
        pred_final = y_pred
    
    # Compute how often each sample was clustered into the same cluster as its
    # majority voted cluster in "pred_final".
    # At the end we divide the number of overlapping cluster labels
    # by the number of colummns minus the number of missings to ensure that
    # only assigned cluster labels are used to compute the stability.
    pred_stability = pred_mat.apply(lambda x: check_overlap(x, pred_final),
                                    axis = 0).mean(axis = 1)
    
    if average:
        pred_stability = pred_stability.mean()
        
    return (pred_stability)

def check_overlap(x, y):
    '''
    Checks which elements in two arrays are identical. This function is used
    for calculating cluster stability.

    Parameters
    ----------
    x : ndarray
        Array of predicted cluster labels.
    y : ndarray
        Array of reference cluster labels.

    Returns
    -------
    res : ndarray
        Array of booleans indicating if elements matched or not. 

    '''
    res = x == y
    res[x.isna()] = np.nan
    return(res)

def map_cluster_to_overlap(y_pred, cluster_labels, train_index):
    '''
    Maps cluster labels according to maximum overlap.

    Parameters
    ----------
    y_pred : ndarray
        Predicted cluster labels.
    cluster_labels : ndarray
        Reference cluster labels.
    train_index : ndarray
        Index of the samples for which cluster labels were predicted.

    Returns
    -------
    y_pred_mapped : ndarray
        The mapped cluster labels. The predicted cluster labels, but now with
        the labels matching to the reference label with which they have maximum
        overlap.

    '''
    # Initiate mapper which will store counts of all overlapping cluster labels
    mapper = pd.DataFrame(index=[0,1,2,3,4,5], columns=[0,1,2,3,4,5])
    
    # Initiate y_pred_mapped which will store the new mapped cluster labels
    y_pred_mapped = pd.Series(index = np.arange(len(y_pred)), dtype = np.float64)
    
    # Per cluster, note the number of overlapping samples with the original clusters
    mapper[0] = pd.Series(cluster_labels[train_index][y_pred == 0]).value_counts()
    mapper[1] = pd.Series(cluster_labels[train_index][y_pred == 1]).value_counts()
    mapper[2] = pd.Series(cluster_labels[train_index][y_pred == 2]).value_counts()
    mapper[3] = pd.Series(cluster_labels[train_index][y_pred == 3]).value_counts()
    mapper[4] = pd.Series(cluster_labels[train_index][y_pred == 4]).value_counts()
    mapper[5] = pd.Series(cluster_labels[train_index][y_pred == 5]).value_counts()
    
    # Convert mapper to matrix
    mapper = np.matrix(mapper).astype(float)
    
    # Get index of max overlap
    while not np.isnan(mapper).all():
        idx_max = np.where(mapper == np.nanmax(mapper))
        if len(idx_max) > 1:
            # Select the smallest cluster to map
            smallest_cluster = np.nansum(mapper[:,idx_max[1]], axis = 0).argmin()
            if len(idx_max[1][idx_max[1] == idx_max[1][smallest_cluster]]) > 1:
                original_cluster = idx_max[0][idx_max[1] == idx_max[1][
                    smallest_cluster]][pd.Series(cluster_labels).value_counts(
                        )[idx_max[0][idx_max[1] == idx_max[1][
                            smallest_cluster]]].argmin()]
                new_cluster = idx_max[1][smallest_cluster]
            else:
                original_cluster = idx_max[0][smallest_cluster]
                new_cluster = idx_max[1][smallest_cluster]
        else:
            original_cluster = idx_max[0][0]
            new_cluster = idx_max[1][0]
        y_pred_mapped[y_pred == new_cluster] = original_cluster
        
        # Remove mapped cluster from mapper
        mapper[original_cluster, :] = np.nan
        mapper[:, new_cluster] = np.nan
    
    return y_pred_mapped
        
def map_cluster_to_centroid(Z, y_pred, centroids):
    '''
    Maps cluster labels based on cluster centroid distances
    
    Parameters
    ----------
    Z : ndarray
        samples in the latent feature space of the new samples you want to map.
    y_pred : ndarray
        predicted cluster labels for the new samples you want to map.
    centroids : ndarray
        centroid position for each cluster. Clusters should be represented by
        the rows, and the columns should represent the latent features. These
        are the centroids of the original clusters, which you now want to map
        you new clusters onto.

    Returns
    -------
    y_pred : pandas.Series
        remapped predicted cluster labels for all samples..

    '''
    # If Z was inputted as dataframe, convert it to numpy array
    if(type(Z) == pd.core.frame.DataFrame):
        Z = np.asarray(Z)
    
    
    # Initialise distance matrix
    dist = pd.DataFrame(index = np.unique(y_pred),
                        columns = range(1, centroids.shape[0]+1),
                        dtype = int)
    
    # For each cluster label in y_pred, compute the mean euclidean distance
    # from the points in that cluster to the centroid of the different clusters
    for cl in range(np.nanmax(y_pred).astype(int)+1):
        Z_cl = Z[y_pred == cl]
        for cent in range(centroids.shape[0]):
            dist.iloc[cl, cent] = np.mean(np.linalg.norm(Z_cl - centroids[cent],
                                                    axis = 1))
    
    # Initialise the mapping matrix which will store the cluster label mapping
    cl_mapping = np.full((2, centroids.shape[0]), -999)
    cl_mapping[0,:] = np.unique(y_pred)
    
    # Identify the cluster that is closest to one of the centroids and map its 
    # label to that centroid's cluster label. Then, set the distances of that
    # cluster and centroid to infinity so it cannot be re-picked. Repeat untill
    # all clusters are mapped.
    for cent in range(centroids.shape[0]):
        closest_cl = np.where(dist == dist.min().min())
        cl_mapping[1, closest_cl[1][0]] = closest_cl[0][0]
        dist.iloc[closest_cl[0][0], :] = np.inf
        dist.iloc[:, closest_cl[1][0]] = np.inf
    
    # map predicted cluster label to the cluster label of their closest centroid
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.replace(cl_mapping[1,:], cl_mapping[0,:])
    
    return y_pred

def jaccard_similarity(x, x_pred, majority_vote = False):
    '''
    Calculate Jaccard similarity coefficient between an array of reference
    cluster labels and multiple columns of predicted cluster labels.

    Parameters
    ----------
    x : pandas.Series, ndarray, or pandas.DataFrame with only one column
        An array containing the cluster memberships per sample.
    x_pred : pandas.DataFrame
        A DataFrame containing the cluster memberships for all samples (rows)
        per iteration (columns) of resampling. Can contain missing values.

    Returns
    -------
    y_pred_mapped : pandas.DataFrame
        DataFrame of mapped cluster labels
    jac_mat_final : 

    '''
    
    # Define a final cluster label through majortiy voting
    if majority_vote == True:
        x = x_pred.mode(axis=1)[0].astype(int)
        
    if type(x) == pd.DataFrame:
        x = pd.Series(x.iloc[:,0])
    else:
        x = pd.Series(x)
    
    # Initiate dataframe to store max jaccard scores per cluster
    jac_mat_final = pd.DataFrame(index=np.arange(x.min(), x.max()+1),
                                 columns = x_pred.columns)
    
    # Initiate y_pred_mapped to store mapped cluster labels
    y_pred_mapped = pd.DataFrame(index = np.arange(len(x)),
                                 columns = x_pred.columns, dtype = np.float64)
    
    # Check if cluster labels match
    x_unique = x.unique()
    x_pred_unique = x_pred.iloc[:,0].unique()[~pd.isna(
        x_pred.iloc[:,0].unique())]
    if not (np.isin(x_unique, x_pred_unique).all()) or not (
            np.isin(x_pred_unique, x_unique).all()):
        print("One of the clusters is not present in the new predicted"\
                   " set. Are you sure cluster labels match? Otherwise "\
                       "spurious results could be obtained!")
    
    for pred in x_pred.columns:
        x_pred_i = x_pred[pred]
        
        jac_mat = pd.DataFrame(index=np.arange(x.min(), x.max()+1),
                               columns = np.arange(x.min(), x.max()+1))
        
        # Remove samples from x that are not present in pred_i
        x_i = x[~x_pred_i.isna()]
        
        for i in np.arange(x.min(), x.max()+1):
            i_samples = x_i.index[x_i == i].values
            for j in np.arange(x.min(), x.max()+1):
                j_samples = x_pred_i.index[x_pred_i == j].values
                
                # Compute intersection
                intersection = len(np.intersect1d(i_samples, j_samples))
                # Compute union
                union = len(np.union1d(i_samples, j_samples))
                
                jac_mat.loc[i,j] = intersection/union
        
        
        mapper = np.matrix(jac_mat).astype(float)
        
        # Get index of max jaccard score
        while not np.isnan(mapper).all():
            idx_max = np.where(mapper == np.nanmax(mapper))

            original_cluster = idx_max[0][0]
            new_cluster = idx_max[1][0]
            y_pred_mapped.loc[x_pred_i == new_cluster, pred] = original_cluster
            
            # Store max jaccard score in jac_mat_final
            jac_mat_final.loc[original_cluster, pred] = np.nanmax(mapper)
            
            # Remove mapped cluster from mapper
            mapper[original_cluster, :] = np.nan
            mapper[:, new_cluster] = np.nan
            
    return(y_pred_mapped, jac_mat_final)





def compare_clusters(clusters1 , clusters2, Z1, Z2, df1, df2, descriptives1,
                     descriptives2, exclusive_mapping = True,
                     outcomes = None, save = False,
                     savename = "cluster mappings", filetype = "png"):
    '''
    Compares clusters between two data sets. This can be used to asses cluster
    generalisability. It produces a figure indicating which cluster across two
    data sets are most similar based on three different types of variables.

    Parameters
    ----------
    clusters1 : ndarray, pandas.Series, or pandas.DataFrame with one column
        Array of clusters labels from dataset set 1.
    clusters2 : ndarray, pandas.Series, or pandas.DataFrame with one column
        Array of clusters labels from dataset set 2.
    Z1 : pandas.DataFrame
        Latent features of data set 1.
    Z2 : pandas.DataFrame
        Latent features of data set 2.
    df1 : pandas.DataFrame
        Input data of data set 1.
    df2 : pandas.DataFrame
        Input data of data set 1.
    descriptives1 : pandas.DataFrame
        descriptives dataframe of data set 1. Contains additiona variables
        describing the samples, not present in the input data.
    descriptives2 : pandas.DataFrame
        descriptives dataframe of data set 2. Contains additiona variables
        describing the samples, not present in the input data.
    exclusive_mapping : Boolean, optional
        Boolean indicating whether mappings should be exclusive. This means
        that if cluster x is mapped to cluster y, both clusters will not be
        available for future mappings. If set to false, multiple clusters can
        be mapped to the same cluster, for example if cluster x closest to y,
        and cluster z is also closest to y, then both clusters will be mapped
        to cluster y. If false, we look from the perspective of set 2, so per
        cluster in set 2, we find the cluster in set 1 that is most similar.
        The default is True.
    outcomes : ndarray
        Array of strings describing which outcome variables to include, can
        contain any of the following: ["vasoactive", "mort_icu", "icu_los", 
        "apacheIV_subgroup1"]. The default is None, which includes all outcome
        variables.
    save : boolean
        boolean indicating whether to locally save figure. The fedault is False.
    savename : str
        String specifying the name (and optionally directory) to save the
        figure to. The default is "cluster mappings"
    filetype : str
        String specifying the filetype to save figure as. For example "png", or
        "svg". The default is "png".

    Returns
    -------
    None.

    '''
    # Will crash when clusters start  from 1 and not 0
    
    if outcomes == None:
        outcomes_unspecified = True
        outcomes = ["vasoactive", "mort_icu", "icu_los", "apacheIV_subgroup1"]
    else:
        outcomes_unspecified = False
    
    # Make sure cluster labels are stored in np array
    if type(clusters1) != np.ndarray:
        clusters1 = clusters1.to_numpy()
    if type(clusters2) != np.ndarray:
        clusters2 = clusters2.to_numpy()
    
    # ==================== #
    # Z Euclidean distance #
    # ==================== #
    
    # Compute
    Z_dist = scipy.spatial.distance.cdist(Z1, Z2)
    
    # Compte distance between all clusters based on Z euclidean distance
    Z_cluster_dist_mat = pd.DataFrame(index = np.unique(clusters1),
                                      columns = np.unique(clusters1))
    for i in Z_cluster_dist_mat.index:
        for j in Z_cluster_dist_mat.columns:
            Z_cluster_dist_mat.iloc[i,j] = pd.DataFrame(Z_dist).loc[
                clusters1==i, clusters2==j].mean().mean()
            
    # Check for most similar clusters based on Z euclidean distance
    mapper = Z_cluster_dist_mat.copy()
    Z_clusters_mapped = np.unique(clusters1).copy()
    while not mapper.isna().all().all():
        idx_max = np.where(mapper == np.nanmin(mapper))

        Z_clusters_mapped[idx_max[1][0]] = idx_max[0][0]
        
        # Remove mapped cluster from mapper
        if exclusive_mapping:
            mapper.iloc[idx_max[0][0], :] = np.nan
        mapper.iloc[:, idx_max[1][0]] = np.nan
        print(f"MUMC+ cluster {idx_max[1][0]+1} is most similar to SICS cluster {idx_max[0][0]+1}")
    
    # ================ #
    # x Gower distance #
    # ================ #
    
    # Check for most similar clusters based on x gower distance
    x1 = np.array(df1)
    x2 = np.array(df2)
    
    # Compute gower distance matrix based on the original feature space
    x_dist = gower.gower_matrix(x1, x2, cat_features = df1.dtypes == "category")
    
    # Compte distance between all clusters based on Z euclidean distance
    x_cluster_dist_mat = pd.DataFrame(index = np.unique(clusters1),
                                      columns = np.unique(clusters1))
    for i in x_cluster_dist_mat.index:
        for j in x_cluster_dist_mat.columns:
            x_cluster_dist_mat.iloc[i,j] = pd.DataFrame(x_dist).loc[
                clusters1==i, clusters2==j].mean().mean()
    
    mapper = x_cluster_dist_mat.copy()
    x_clusters_mapped = np.unique(clusters1).copy()
    while not mapper.isna().all().all():
        idx_max = np.where(mapper == np.nanmin(mapper))

        x_clusters_mapped[idx_max[1][0]] = idx_max[0][0]
        
        # Remove mapped cluster from mapper
        if exclusive_mapping:
            mapper.iloc[idx_max[0][0], :] = np.nan
        mapper.iloc[:, idx_max[1][0]] = np.nan
        
        print(f"MUMC+ cluster {idx_max[1][0]+1} is most similar to SICS cluster {idx_max[0][0]+1}")
        
    
    # ================ #
    # heatmap distance #
    # ================ #
    
    heatmap1 = descriptives1[outcomes]
    heatmap2 = descriptives2[outcomes]
    
    heatmap1 = pd.concat([heatmap1, pd.get_dummies(heatmap1.apacheIV_subgroup1)],
                         axis =1).drop("apacheIV_subgroup1", axis = 1)
    heatmap1, scaler = scale_data(heatmap1)
    
    heatmap2 = pd.concat([heatmap2, pd.get_dummies(heatmap2.apacheIV_subgroup1)],
                         axis =1).drop("apacheIV_subgroup1", axis = 1)
    heatmap2 = pd.DataFrame(scaler.transform(heatmap2), columns = heatmap1.columns)
    
    heatmap1.loc[:, "cluster"] = clusters1
    heatmap2.loc[:, "cluster"] = clusters2
    
    heatmap1 = heatmap1.groupby("cluster").mean()      
    heatmap2 = heatmap2.groupby("cluster").mean()
    
    # Compute distance between clusters
    heatmap_cluster_dist_mat = pd.DataFrame(scipy.spatial.distance.cdist(
        heatmap1, heatmap2))

    # Check for most similar clusters based on heatmap euclidean distance
    mapper = heatmap_cluster_dist_mat.copy()
    heatmap_clusters_mapped = np.unique(clusters1).copy()
    while not mapper.isna().all().all():
        idx_max = np.where(mapper == np.nanmin(mapper))

        heatmap_clusters_mapped[idx_max[1][0]] = idx_max[0][0]
        
        # Remove mapped cluster from mapper
        if exclusive_mapping:
            mapper.iloc[idx_max[0][0], :] = np.nan
        mapper.iloc[:, idx_max[1][0]] = np.nan
        print(f"MUMC+ cluster {idx_max[1][0]+1} is most similar to SICS cluster {idx_max[0][0]+1}")
    
    # If outcomes not specified, als create mapping without required vasoactive
    if outcomes_unspecified:
        print("Now without vasoactive medication")
        heatmap1.drop(["vasoactive"], axis = 1, inplace = True) 
        heatmap2.drop(["vasoactive"], axis = 1, inplace = True) 

        # Compute distance between clusters
        heatmap_cluster_dist_mat = pd.DataFrame(scipy.spatial.distance.cdist(
            heatmap1, heatmap2))
        # Check for most similar clusters based on heatmap euclidean distance
        mapper = heatmap_cluster_dist_mat.copy()
        heatmap_clusters_mapped2 = np.unique(clusters1).copy()
        while not mapper.isna().all().all():
            idx_max = np.where(mapper == np.nanmin(mapper))
    
            heatmap_clusters_mapped2[idx_max[1][0]] = idx_max[0][0]
            
            # Remove mapped cluster from mapper
            if exclusive_mapping:
                mapper.iloc[idx_max[0][0], :] = np.nan
            mapper.iloc[:, idx_max[1][0]] = np.nan
            print(f"MUMC+ cluster {idx_max[1][0]+1} is most similar to SICS cluster {idx_max[0][0]+1}")
    
    cluster_mappings = pd.DataFrame(index = np.unique(clusters1))
    
    cluster_mappings.loc[:, "original feature space"] = x_clusters_mapped
    #cluster_mappings.loc[:, "outcome variables"] = descriptives_clusters_mapped
    cluster_mappings.loc[:, "heatmap"] = heatmap_clusters_mapped
    if outcomes_unspecified:
        cluster_mappings.loc[:,"heatmap filtered"] = heatmap_clusters_mapped2
    cluster_mappings.loc[:, "latent feature space"] = Z_clusters_mapped
    
    cluster_mappings.index += 1
    cluster_mappings.rename('MUMC+ cluster {}'.format, inplace = True)
    cluster_mappings += 1
    
    plt.figure(figsize = (7,7))
    ax = sns.heatmap(cluster_mappings, cmap = sns.color_palette("deep")[0:6],
                     linewidths = 1, square = True, annot = True, alpha = 0.8)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks(np.linspace(1.5,5.5, 6))
    colorbar.set_ticklabels(['SICS Cluster 1',
                             'SICS Cluster 2',
                             'SICS Cluster 3',
                             'SICS Cluster 4',
                             'SICS Cluster 5',
                             'SICS Cluster 6'])
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(np.linspace(0.5, 5.5, 6),
               labels = ['MUMC+ Cluster 1',
                         'MUMC+ Cluster 2',
                         'MUMC+ Cluster 3',
                         'MUMC+ Cluster 4',
                         'MUMC+ Cluster 5',
                         'MUMC+ Cluster 6'],
               rotation=0, ha="right")
    plt.tight_layout()
    
    if save:
        plt.savefig(f"figures/{savename}_multiplot.{filetype}")
    
    return(cluster_mappings)