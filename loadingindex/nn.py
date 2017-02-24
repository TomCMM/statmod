import pandas as pd
import matplotlib.pyplot as plt



if __name__=='__main__':
    loading = pd.read_csv("/home/thomas/phd/statmod/data/neuralnetwork/loadings_pca_sta_coldpool.csv", index_col=0)
    loading_T = pd.read_csv("/home/thomas/phd/statmod/data/neuralnetwork/loadings_pca_sta_coldpool_T.csv", index_col=0)
    scores = pd.read_csv("/home/thomas/phd/statmod/data/neuralnetwork/scores_pca_sta_coldpool.csv", index_col=0)
    scores_T = pd.read_csv("/home/thomas/phd/statmod/data/neuralnetwork/scores_pca_sta_coldpool_T.csv", index_col=0)
    
    loading_true = pd.read_csv("/home/thomas/phd/statmod/data/neuralnetwork/loadings_pca_sta_coldpool_fullperiod.csv", index_col=0)
    scores_true = pd.read_csv("/home/thomas/phd/statmod/data/neuralnetwork/scores_pca_sta_coldpool_fullperiod.csv", index_col=0)
          
     
    print loading
    print loading_true
    
    for col in loading.columns:
        plt.scatter(loading.loc[:,col], loading_true.loc[:,col])
        plt.show()
