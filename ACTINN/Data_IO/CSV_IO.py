import time

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from harmony import harmonize
from sklearn.manifold import TSNE


def more_data_get(train_path, test_path, slices, train_labels_path, hca_data_path):
    data = sc.read_h5ad(train_path)[0:slices]
    hca_data = sc.read_h5ad(hca_data_path)[0:slices]
    infer_set = sc.read_h5ad(test_path)[0:slices]

    data_label = pd.read_csv(train_labels_path)[0:slices]
    data.obs["cell_type"] = data_label.level1.values
    print(data.shape, hca_data.shape)
    sc.pp.recipe_zheng17(data, n_top_genes=15000)
    sc.pp.recipe_zheng17(hca_data, n_top_genes=15000)
    sc.pp.recipe_zheng17(infer_set, n_top_genes=15000)

    genesTrain = data.var_names
    genesQuery = hca_data.var_names
    cgenes = genesTrain.intersection(genesQuery)
    genesInfer = infer_set.var_names
    cgenes = cgenes.intersection(genesInfer)

    data = data[:, cgenes]
    hca_data = hca_data[:, cgenes]
    infer_set = infer_set[:, cgenes]
    print("after gene selection", data.shape, hca_data.shape, infer_set.shape)
    hca_data.obs['cell_type'] = hca_data.obs['cell_type'].values.astype('object')

    hca_data = hca_data[hca_data.obs['cell_type'] != 'NotAssigned']
    hca_data = hca_data[hca_data.obs['cell_type'] != 'doublets']
    #     hca_data = hca_data[hca_data.obs['cell_type'] != 'Atrial_Cardiomyocyte']
    #     hca_data = hca_data[hca_data.obs['cell_type'] != 'Ventricular_Cardiomyocyte']

    cell_type = hca_data.obs['cell_type'].values
    cell_type = np.where(cell_type == "Adipocytes", "Adipocyte", cell_type)
    cell_type = np.where(cell_type == "Endothelial", "Endothelial cell", cell_type)
    cell_type = np.where(cell_type == "Fibroblast", "Fibroblast", cell_type)
    cell_type = np.where(cell_type == "Lymphoid", "Lymphoid cell", cell_type)
    cell_type = np.where(cell_type == "Mesothelial", "Mesothelial cell", cell_type)
    cell_type = np.where(cell_type == "Myeloid", "Myeloid cell", cell_type)
    cell_type = np.where(cell_type == "Pericytes", "Pericyte", cell_type)
    cell_type = np.where(cell_type == "Smooth_muscle_cells", "Smooth muscle cell", cell_type)
    cell_type = np.where(cell_type == "Atrial_Cardiomyocyte", "Cardiomyocyte cell", cell_type)
    cell_type = np.where(cell_type == "Ventricular_Cardiomyocyte", "Cardiomyocyte cell", cell_type)

    cell_type = np.where(cell_type == "Neuronal", "Unknown", cell_type)

    hca_data.obs['cell_type'] = cell_type
    data = data.concatenate(hca_data)
    oringinal_data_len = data.shape[0]
    sc.pp.filter_cells(data, min_genes=170)
    print("After filter min_genes", data.shape[0] / oringinal_data_len)
    oringinal_data_len = data.shape[0]

    sc.pp.filter_cells(data, max_genes=2000)
    print("After filter max_genes", data.shape[0] / oringinal_data_len)

    #     print("after filter",data.shape, hca_data.shape, infer_set.shape)

    print(np.unique(data.obs.cell_type.values))

    return data, infer_set


def vars(a, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    """
    a_squared = a.copy()
    a_squared.data **= 2
    return a_squared.mean(axis) - np.square(a.mean(axis))


def stds(a, axis=None):
    """ Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    return np.sqrt(vars(a, axis))


def scale_sets(total_set, label):
    """
    Get common genes, normalize  and scale the sets
    INPUTS:
        sets-> a list of all the sets to be scaled

    RETURN:
        sets-> normalized sets
    """
    #
    # common_genes = set(sets[0].index)
    # for i in range(1, len(sets)):
    #     common_genes = set.intersection(set(sets[i].index), common_genes)
    # common_genes = sorted(list(common_genes))

    # total_set = np.array(pd.concat(sets, axis=0, sort=False), dtype=np.float32)
    total_set / total_set.sum(1) * 20000
    # total_set = np.divide(np.array(total_set), np.expand_dims(np.array(np.sum(total_set, axis=1)),axis=1)) * 20000
    total_set = total_set.log1p()
    expr = np.array(total_set.sum(1)).flatten()
    total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
    label = label[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99))]
    cv = np.array(np.array(stds(total_set, axis=1)) / total_set.mean(1)).flatten()
    total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
    label = label[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99))]

    return total_set, label


def get_data(train_path, test_path, slices, train_labels_path, hca_data_path):
    data = sc.read_h5ad(train_path)[0:slices]
    # hca_data = sc.read_h5ad(hca_data_path)[0:slices]
    infer_set = sc.read_h5ad(test_path)[0:slices]

    data_label = pd.read_csv(train_labels_path)[0:slices]

    print("==> Begin data processing ")
    data.obs["level1"] = data_label.level1.values
    data.obs["level2"] = data_label.level2.values
    data.obs["level3"] = data_label.level3.values
    data.obs["level4"] = data_label.level4.values

    print(data.shape)

    n_top_genes = 10000
    sc.pp.recipe_zheng17(data, n_top_genes=n_top_genes)
    sc.pp.recipe_zheng17(infer_set, n_top_genes=n_top_genes)

    genesTrain = data.var_names
    genesInfer = infer_set.var_names
    cgenes = genesTrain.intersection(genesInfer)

    data = data[:, cgenes]
    # hca_data = hca_data[:, cgenes]
    infer_set = infer_set[:, cgenes]
    print("after gene selection", data.shape, infer_set.shape)

    oringinal_data_len = data.shape[0]
    #     sc.pp.filter_cells(data, min_genes=100)
    print("After filter min_genes", data.shape[0] / oringinal_data_len)
    oringinal_data_len = data.shape[0]

    #     sc.pp.filter_cells(data, max_genes=4000)
    print("After filter max_genes", data.shape[0] / oringinal_data_len)

    return data, data_label, infer_set


def CSV_IO(train_path: str, train_labels_path: str, test_path: str, test_labels_path: str,
           batchSize: int = 128, workers: int = 12, fold=1, n_comps=2000, hca_data_path="../data/heart.h5ad",
           slices=800000):
    """
    This function allows the use of data that was generated by the original ACTINN code (in TF)

    INPUTS
        train_path-> path to the h5 file for the training data (dataframe of Genes X Cells)
        train_labels_path-> path to the csv file of the training data labels (cell type strings)
        test_path-> path to the h5 file of the testing data (dataframe of Genes X Cells)
        test_labels_path-> path to the csv file of the testl dataabels (cell type strings)

    RETURN
        train_data_loader-> training data loader consisting of the data (at batch[0]) and labels (at batch[1])
        test_data_loader-> testing data loader consisting of the data (at batch[0]) and labels (at batch[1])

    """
    t0 = time.time()
    #     slices = 800000
    print("==> Reading in H5ad Data frame (CSV) ")
    train_data, data_label, infer_set = get_data(train_path, test_path, slices, train_labels_path, hca_data_path)

    #     ------------------------------oringinal data------------------

    #     data = sc.read_h5ad(train_path)[0:slices]
    #     infer_set = sc.read_h5ad(test_path)[0:slices]
    #     data_label = pd.read_csv(train_labels_path)[0:slices]
    #     data.obs["cell_type"] = data_label.level1.values
    #     print("before pp",data.X.shape,infer_set.X.shape)
    #     sc.pp.filter_cells(data,min_genes=250)
    #     sc.pp.filter_cells(data,max_genes=3000)

    #     sc.pp.recipe_zheng17(data,n_top_genes=10000)
    #     sc.pp.recipe_zheng17(infer_set,n_top_genes=10000)

    #     genesTrain = data.var_names
    #     genesQuery = infer_set.var_names
    #     print("In pp",data.X.shape,infer_set.X.shape)
    #     data_label = data.obs["cell_type"].values
    #     print(data_label)

    #     cgenes = genesTrain.intersection(genesQuery)
    #     data = data[:,cgenes].X
    #     infer_set = infer_set[:,cgenes].X
    #     ------------------------------oringinal data------------------

    #     ------------------------------PCA------------------
    #     data = sc.read_h5ad(train_path)[0:slices].X
    #     len_train= data.shape
    #     infer_data = sc.read_h5ad(test_path)[0:slices].X
    #     len_test= infer_data.shape
    #     print("before pca",len_train,len_test)

    #     if isfile('allData_{}Dimen_PCA.npz.npy'.format(str(n_comps))):
    #         print("loading data")
    #         alldata = np.load('allData_{}Dimen_PCA.npz.npy'.format(str(n_comps)))
    #     else:

    #         alldata = bmat([[data], [infer_data]], format="csr")
    #         zero_indexs = np.array(np.sum(alldata, axis=0) > 0).flatten()

    #         alldata=alldata[:,zero_indexs]
    #         alldata=sc.pp.pca(alldata,copy=True,n_comps=n_comps)

    #         np.save('allData_{}Dimen_PCA.npz'.format(str(n_comps)), alldata)
    #     # alldata = alldata

    #     data = alldata[0:len_train[0],]
    #     infer_set = alldata[len_train[0]:,]
    #     ------------------------------PCA------------------

    print("after pre - processing", train_data.shape, infer_set.shape)
    print("using time :", (time.time() - t0) / 60)
    alldata = None

    return train_data, data_label, infer_set
