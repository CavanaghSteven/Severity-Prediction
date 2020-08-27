
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from util import reduce_mem_usage, NEW_DATA_DIR, load_data, create_dir
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D

create_dir('log')
create_dir('log/pca_eda')


for scale in [True, False]:

    print(f'Scale: {scale}')

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = load_data()

    columns = X_train.columns

    pca = PCA(n_components=3)
    # pca = PCA(n_components=0.99)
    scaler = StandardScaler()

    if scale:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    X_val = pca.transform(X_val)

    # print(f'Features Old: {len(columns)} \n New: {len(pca.components_)}')

    comp_columns = [f'comp {i}' for i in range(len(pca.components_))]
    # print(comp_columns)

    pca_train_df = pd.DataFrame(data=X_train, columns=comp_columns)
    pca_test_df = pd.DataFrame(data=X_test, columns=comp_columns)
    pca_val_df = pd.DataFrame(data=X_val, columns=comp_columns)

    pca_train_df['Severity'] = y_train
    pca_test_df['Severity'] = y_test
    pca_val_df['Severity'] = y_val

    if scale:
        d3_filename = f'log/pca_eda/3d_dimension_reduction_afterscale.png'
        d2_filename = f'log/pca_eda/2d_dimension_reduction_afterscale.png'

    else:
        d3_filename = f'log/pca_eda/3d_dimension_reduction_beforescale.png'
        d2_filename = f'log/pca_eda/2d_dimension_reduction_beforescale.png'

    plt.figure()
    sns.scatterplot(x=pca_train_df['comp 0'], y=pca_train_df['comp 1'], hue=pca_train_df['Severity'])
    plt.tight_layout()
    plt.savefig(d2_filename)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pca_train_df['comp 0'], pca_train_df['comp 1'], pca_train_df['comp 2'], c=pca_train_df['Severity'])
    # plt.tight_layout()
    plt.savefig(d3_filename)
