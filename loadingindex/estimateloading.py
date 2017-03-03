#===============================================================================
# DESCRIPTION
#    THIS FILE TRY TO ESTIMATE THE LOADING OBTAIN FROM THE PCA APPLIED ON THE STATIONS OBSERVATION AT EACH ARPS MODEL GRIDPOINT
#===============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPRegressor


if __name__=='__main__':

    #===========================================================================
    # READ  THE DATA
    #===========================================================================
    loading = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/loadings_pca_sta_coldpool.csv", index_col=0)
    loading_T = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/loadings_pca_sta_coldpool_T.csv", index_col=0)
    scores = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/scores_pca_sta_coldpool.csv", index_col=0)
    scores_T = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/scores_pca_sta_coldpool_T.csv", index_col=0)
    
    loading_true = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/loadings_pca_sta_coldpool_fullperiod.csv", index_col=0)
    scores_true = pd.read_csv("/home/thomas/phd/statmod/data/loadingindex/scores_pca_sta_coldpool_fullperiod.csv", index_col=0)
          
    
#     #===========================================================================
#     # BASIC PLOT
#     #===========================================================================
#     print "Loading from the model"

#     print loading.head()
#     print "Loading from the station observation"
#     print loading_true.head()
# #     
#     for col in loading.columns:
#         plt.scatter(loading.loc[:,col], loading_true.loc[:,col])
#         plt.show()

    #===============================================================================
    # PREPROCESSING
    #===============================================================================

    X = loading
    Y = loading_true.loc[:,'2']
    # keep 20% for the test dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

#     #===========================================================================
#     # ESTIMATION THROUGH LINEAR REGRESSION
#     #===========================================================================
#     reg = LinearRegression()
#     reg.fit(X_train, Y_train)

    #===========================================================================
    # NEURAL NETWORK ESTIMATION
    #===========================================================================
    reg = MLPRegressor(hidden_layer_sizes=(10,2), solver="lbfgs", alpha=1e-5, random_state=1)
    reg.fit(X_train,Y_train)

    #===============================================================================
    # PLOT RESULTS
    #===============================================================================

    print reg.score(X_test, Y_test)
    
    plt.scatter(Y_train, reg.predict(X_train), c ='g', label='train dataset')
    plt.scatter(Y_test, reg.predict(X_test), c='r', label='test dataset')
    plt.show()
    

    