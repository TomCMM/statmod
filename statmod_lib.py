#===============================================================================
#     DESCRIPTION
#        Contain the statistical model to perform the downscalling
#===============================================================================

#===============================================================================
# Library
#===============================================================================
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import glob
from LCBnet_lib import *
import statsmodels.api as sm

from sklearn.preprocessing import scale, StandardScaler
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
# sklearn.preprocessing.Imputer
from scipy.stats.stats import pearsonr   

from LCBnet_lib import *
from Irradiance import Irradiance_sim_obs
import glob
from compiler.ast import Const
from sklearn import metrics
from numpy import nan
from scipy.optimize import curve_fit

LCB_Irr = Irradiance_sim_obs.LCB_Irr


class StaMod():
    """
    Contain function and plot to perform Empirical Orthogonal Function
    
    PARAMETERS
        df: dataframe with a variable for each stations data
        AttSta: att_sta_object with the attribut of the stations
    """
    def __init__(self, df, AttSta):
        self.df = df
        
        self.AttSta = AttSta
        
        self.nb_PC = []
        self.eigenvalues = [] # or explained variance
        self.eigenvectors = [] # or loadings
        self.eigenpairs = [] # contain the eigen pairs of the PCA (scores and loadings)
        self.scores = pd.DataFrame([]) # contain the PC scores
        self.params_loadings =[] # contains the fit parameters for the PC loadings
        self.params_scores = [] # contains the fit parameters for the PC scores 
        self.topo_index = [] # contain the topographic index at each station point
        self.standard = False # flag to see if the input data has been standardize

    def pca_transform(self, nb_PC=4, standard = False, sklearn=False, cov=True):
        """
        Perform the Principal component analysis with SKlearn
        using singular value decomposition
        The dataframe is standardize
        
        
        
        parameters:
            standard: default = True, standardize the dataframe
            nb_PC: default = 4, number of principal components to be used
            sklearn: if True (default=False) use svd by sklearn
            cov: if true (by default) sue the correlation matrix to perform the PCA analysis
        
        Stock in the object
            Dataframe with:
                eigenvalues
                eigenvectors
                scores
            list of vectors:
                eigenpairs
        
        NOTE:
            By default sklearn remove the mean from the dataset. So I cant use it to perform the downscalling
        
        References:
            http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#projection-onto-the-new-feature-space
        """
        df = self.df
        self.nb_PC = nb_PC

        if standard:
            # standardize
#             df_std = StandardScaler().fit_transform(df)
            self.standard = True
            df = (df - df.mean(axis=0)) / df.std(axis=0) # another way to standardise
        
        
        #=======================================================================
        # Sklearn
        #=======================================================================
        if sklearn:
            print "o"*80
            print "SVD sklearn used"
            print "o"*80
            #Create a PCA model with nb_PC principal components
            pca = PCA(nb_PC)
    
            # fit data
            pca.fit(df)
             
            #Get the components from transforming the original data.
            scores = pca.transform(df) #  or PCs
            eigenvalues = pca.explained_variance_
            eigenvectors = pca.components_ # or loading 
             
            # Make a list of (eigenvalue, eigenvector) tuples     
            self.eigpairs = [(np.abs(self.eigenvalues[i]), self.eigenvector[i,:]) for i in range(len(self.eigenvalues))]



            
            

        #=======================================================================
        # Covariance Matrix
        #=======================================================================
        if cov:
            print "o"*80
            print "Covariance used"
            print "o"*80
            
            X = df.values
            cov_mat = np.cov(X.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
            
#             for ev in eigenvectors:
#                 np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#                 print('Everything ok!')


            scores = X.dot(eigenvectors)



            # I think their is no need for the next step
#             eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
#             eig_pairs.sort()
#             eig_pairs.reverse()
#             
#             # Visually confirm that the list is correctly sorted by decreasing eigenvalues
#             print('Eigenvalues in descending order:')
#             for i in eig_pairs:
#                 print(i[0])
#             print eig_pairs
#             print len(eig_pairs)
#             
#             
#             eigenvalues = np.array([])
#             eigenvectors = np.array([])
# 
#             for PC in range(0,len(eig_pairs)):
#                 eigenvalues = np.append(eigenvalues,eig_pairs[PC][0] )
#                 eigenvectors = np.append(eigenvectors, [eig_pairs[PC][1]])
#             
#             print eigenvalues
#             print eigenvectors
#             print eigenvectors.shape
# 
#             eigenvectors = np.reshape(eigenvectors,(len(eig_pairs), len(eig_pairs)))
#             print eigenvectors.shape
#             
#             print X.T.shape
#             
#             scores = X.T.dot(eigenvectors)

            scores = pd.DataFrame(scores, columns = np.arange(1,len(df.columns)+1), index=df.index)
            eigenvalues = pd.Series(eigenvalues, index= np.arange(1,len(df.columns)+1))
            eigenvectors = pd.DataFrame(eigenvectors.T, columns=df.columns, index=np.arange(1,len(df.columns)+1))
  
        self.scores = scores.ix[:, 0:nb_PC]
        self.eigenvalues =  eigenvalues[0:nb_PC]
        self.eigenvectors =  eigenvectors[0:nb_PC]
            # select only first components
#             scores = 
#             eigenvalues =
#             eigenvectors =
#             
        
        
#         self.scores = pd.DataFrame(scores, columns = np.arange(0,len(df.columns)), index=df.index)
#         self.eigenvalues = pd.Series(eigenvalues, index= np.arange(0,len(df.columns)))
#         self.eigenvectors = pd.DataFrame(eigenvectors, columns=df.columns, index=np.arange(0,len(df.columns)))
  
    def fit_loadings(self, params=None, fit=None):
        """
        DESCRIPTION
            Fit the loadings of the principal components with input independant variable
            (In theapplication of the LCB it is most probably the altitude or its derivative)
        RETURN
            The parameters of the linear regression
            params: topographic parameters to fit the loadings
                    namerasters = [
                   "8x8",
                   "topex",
                    "xx8_10000_8000____",
                    "xx8_20000_18000____"
                   ]
            params_sep: list of parameters value to perfom 2 linear regression
            curvfit: type of fit that you want to use, linear or poly
        """

        
        if not params:
            params = ['Alt']* self.nb_PC

        if not fit:
            fit = [lin]* self.nb_PC

        fit_parameters = []
    
        for PC, row in self.eigenvectors.iterrows():
            X = np.array(self.AttSta.getatt(self.df.keys(), params[PC - 1]))
            self.topo_index.append(X)
            popt, pcov = curve_fit(fit[PC-1], X, row)
            fit_parameters.append([x for x in popt])
            
        fit_parameters = np.vstack(fit_parameters)
        
        self.params_loadings = [pd.DataFrame(fit_parameters, index =range(1,self.nb_PC+1), columns = range(len(popt)))]
        self.fit_loadings = fit
        return self.params_loadings

    def fit_scores(self, predictors, fit=None):
        """
        DESCRIPTION
            Fit the Scores of the principal components with a variables
        Input
            A serie
        Return
            The parameters of the linear regression
        
        """

        if not fit:
            fit = [lin]* self.nb_PC
        
        scores = self.scores
        
        fit_parameters = []
        for i, predictor in enumerate(predictors):
            predictor = predictor.dropna(axis=0, how='any')
            score = scores.iloc[:,i]

            df = pd.concat([predictor, score], axis=1, join='inner')

            popt, pcov = curve_fit(fit[i-1], df.iloc[:,0], df.iloc[:,1])

            fit_parameters.append([x for x in popt])

        fit_parameters = np.vstack(fit_parameters)
        self.params_scores = [pd.DataFrame(fit_parameters, index =range(1,self.nb_PC+1), columns = range(len(popt)))]
        self.fit_scores = fit
        return self.params_scores

    def predict(self, predictors_loadings, predictors_scores, params_loadings=None, params_scores=None):
        """
        DESCRIPTION
            Return an 1/2d array estimated with the previously fit's parameters and predictos 
        RETURN
            a dictionnary of 3D numpy array containing the estimated "loadings", "scores" and "reconstructed variable"
        INPUT

            predictors_loadings: 1/2d array, to be used with the loading
                             parameters to create a loading 2d array
            predictors_scores: Serie, to be used with the scores params to reconstruct the scores
            
            params_loadings: (Optional) A dataframe of the parameters of the linear regression from 
                            fit_loadings. By default, the methods will look for params_loadings in the object. 
                            If it does not exist you should run fit_loadings
            parms_scores: (Optional) Parameters dataframe from fit_scores.  By default, the methods will look for params_loadings in the object. 
                            If it does not exist you should run fit_scores
        
        TODO
            I should implement a way to use a pandas serie for the input "predictor instead of a number
        """

        loadings = []
        scores = []
        predicted = []
        
        fit_loadings = self.fit_loadings
        fit_scores = self.fit_scores
        
        
        if (not params_loadings or not params_scores):
            print 'Getting parameters'
            params_loadings, params_scores = self.get_params()


        # NEED TO IMPLEMENT MATRIX MULTIPLICATION!!!!!!!!!!!!!! I use to much loop
        for PC_nb, fit_loading, fit_score, predictor_loadings, predictor_scores in zip(range(1,self.nb_PC+1),fit_loadings, fit_scores,predictors_loadings,predictors_scores ):
            loading_est = fit_loading(predictor_loadings, *params_loadings[0].loc[PC_nb,:])
            score_est =fit_score(predictor_scores, *params_scores[0].loc[PC_nb,:])
            
            score = pd.concat([score_est]*len(loading_est), axis=1)
            predict= score.multiply(loading_est)
                        
            loadings.append(loading_est)
            scores.append( score_est)
            predicted.append( predict)
        
        loadings = np.array(np.dstack(loadings))
        scores = np.array(np.dstack(scores))
        predicted = np.array(np.dstack(predicted))
        
        
        res = {'loadings':loadings, 'scores':scores, 'predicted': predicted}
        return res
    
#------------------------------------------------------------------------------ 

    def pca_reconstruct(self):
        """
        Reconstruct the original dataset with the "nb_PC" principal component
        Note:
            The idea is to reconstruct by hand to see if the downscalling is done correctly
        """
        eigenvectors = self.eigenvectors
        scores = self.scores
        
        df = pd.DataFrame(columns=eigenvectors.columns, index=scores.index)
        
        for sta in eigenvectors.columns:
            for PC in scores.columns:
                if PC ==1:
                    df[sta] = scores[PC]*eigenvectors[sta][PC]
                else:
                    df[sta] = df[sta] + scores[PC]*eigenvectors[sta][PC]
#         print df
    def skill(self, df_verif, predictors_scores, metrics, params_loadings=None, params_scores=None):
        """
        DESCRIPTION
            Compute bias and RMSE to assess the model performance 
        INPUT
            df_verif: dataframe with the observed values
            predictors: a list of pandas series which contains the predictors for the scores SHOULD NOT BE A LIST
            metrics: sklearn metric function to be used
                    see: http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
                example:
                    metrics.explained_variance_score(y_true, y_pred)     Explained variance regression score function
                    metrics.mean_absolute_error(y_true, y_pred)     Mean absolute error regression loss
                    metrics.mean_squared_error(y_true, y_pred[, ...])     Mean squared error regression loss
                    metrics.median_absolute_error(y_true, y_pred)     Median absolute error regression loss
                    metrics.r2_score(y_true, y_pred[, ...])     R^2 (coefficient of determination) regression score function.
        """
        if (not params_loadings or not params_scores):
            params_loadings, params_scores = self.get_params()

        topo_index = self.topo_index
        data = np.array([])
        res = self.predict(topo_index, predictors_scores)
        data = res['predicted'].sum(axis=2)
        df_rec = pd.DataFrame(data, columns = df_verif.columns, index = predictors_scores[0].index) # should improve this

        score = pd.Series()

        for sta in df_rec:
            df = pd.concat([df_verif[sta],df_rec[sta]], axis=1, join='inner')
            df = df.dropna(axis=0)
#             df.columns=['True', 'Pred']
            df.plot()
            plt.show()
            score[sta] = metrics(df.iloc[:,0], df.iloc[:,1])
        return score

    def get_params(self):
        """
        DESCRIPTION
            Return the params loadings and scores. and return and error if the model has not being fitted
        """
        
        try:
            params_loadings = self.params_loadings
            params_scores = self.params_scores
        except AttributeError:
            raise AttributeError( "The model has not been fitted, run fit_loadings or fit_scores")
        
        return params_loadings, params_scores
           
    def plot_exp_var(self):
        """
        DESCRIPTION
            Make a plot of the variance explaine by the principal components
        """
        print "Plot explainde variance"
        eig_vals = self.eigenvalues
        
        tot = sum(eig_vals)
        var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        
        plt.figure(figsize=(6, 4))
    
        plt.bar(range(4), var_exp, alpha=0.5, align='center',
                label='individual explained variance')
        plt.step(range(4), cum_var_exp, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()

    def plot_loading(self, params_topo=None, params_fit=None, output=False, fit=None):
        """
        DESCRIPTION
            Plot the loadings in function of some parameters
        Parameters
            output: if given save the plot at the path indicated
            params: parameters of the linear regression beetween
                 loadings and independant variables 
        """

        if not params_topo:
            params_topo = ['Alt']* self.nb_PC
        
        if not fit:
            fit = [None]* self.nb_PC
        
        
        for pc_nb, param_topo, func in zip(range(1,self.nb_PC+1), params_topo, fit):
            elev_real = self.AttSta.getatt(self.df.keys(),param_topo) 
            fig, ax = plt.subplots()
            plt.scatter(elev_real, self.eigenvectors.loc[pc_nb])

            if isinstance(params_fit, pd.DataFrame):
                x = np.linspace(min(elev_real), max(elev_real),100)
                p = params_fit.loc[pc_nb,:]
                y = func(x, *p)

                plt.plot(x,y)
                plt.title(param_topo+ "   " + str(pc_nb))
                plt.grid(True)
            
            for i, txt in enumerate(self.df.columns):
                ax.annotate(txt, (elev_real[i], self.eigenvectors.loc[pc_nb][i]))
            if output:
                plt.savefig(output)
            else:
                plt.show()
    
    def plot_scores(self, predictors, params_fit =None,fit=None, output=False):
        """
        DESCRIPTION
            Make a scatter plot of the Principal component scores in function of another variables
        INPUT
            var: time serie with the same index than the dataframe sued in the pca
        """
        
        
        scores = self.scores
        
        for i, predictor, func in zip(range(0,self.nb_PC), predictors, fit):
            predictor = predictor.dropna(axis=0, how='any')
            score = scores.iloc[:,i]
            df = pd.concat([predictor, score], axis=1, join='inner')

            plt.scatter(df.iloc[:,0],df.iloc[:,1] )
  
            x = np.linspace(min(predictor), max(predictor),100)
            p = params_fit.loc[i+1,:]
            y = func(x, *p)
            
            plt.plot(x,y)
            plt.grid(True)
            plt.show()

        if output:
            plt.savefig(output)
        else:
            plt.show()
 
    def plot_scores_ts(self, output=False):
        """
        DESCRIPTION
           Plot scores time serie
        Parameters
            output: if given save the plot at the path indicated
        """
        scores = self.scores
        scores.plot()
        plt.grid()
        if output:
            plt.savefig(output)
        else:
            plt.show()
                 
    def to_adas(self):
        """
        DESCRIPTION
            Write a file in the ARPS ADAS format 
        
        """  
        pass

    def to_swat(self):
        """
        DESCRIPTION
            Write the ouput in the Swat input format
        """

    def stepwise(self):
        """
        DESCRIPTION
            Perform stepwise linear regression with forward method.
        Return
            The best model based on the adjusted correlation score
        """

def pol(x, a, b, c):
    """
    Polynomial function
    """
    return a*x + x**b +c

def lin(x, a, b):
    """
    linear function function
    """
    return a*x +b
    
    
def theta(T,P):
    """
    Compute Potential temperature
    T: temperture n Degree
    P: Pressure in Hectopascale
    """
    Cp = 1004. # Specific heat at constant pressure J/(kg.K)
    R = 287. #Gaz constant ....
    P0 = 1000. #Standard pressure (Hpa)
    
    theta=T*(P0/P)**(Cp/R)
    
    return theta

def br(theta1, theta2, z1,z2):
    """
    Calcul Brunt Vaissalla frequency
    """
    g=9.81
    theta_mean = (theta1 + theta2) / 2.
    br = np.sqrt( (g / theta_mean)  * ((theta2 - theta1)/ (z2 - z1)) )
    
    return br

def froude(br, U, H):
    """
    Valley depth froude number 
    """
#     print br
#     print  br.shape
#     print br * H
#     print U 
    F = U / (br* H)  
     
    return F
    

if __name__ == '__main__':
    
#     #===========================================================================
#     # Fitting the model
#     #===========================================================================
    var = "Ta C"
    AttSta = att_sta(Path_att='/home/thomas/params_topo.csv')
    From = "2015-03-01 00:00:00"
    To = "2016-01-01 00:00:00"
    Lat = [-25,-21]
    Lon = [-48, -45]
    Alt = [400,5000]
       
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
    # Files=glob.glob(InPath+"*")
    AttSta.setInPaths(InPath)
         
          
    station_names =AttSta.stations(['Ribeirao'])
#     [station_names.remove(k) for k in ['C06','C11','C10','C12','C09', 'C08']] # Tmax Value error on the C10
    [station_names.remove(k) for k in ['C18','C17','C09']] # Tmin
#     [station_names.remove(k) for k in [ 'C16','C19']] # daily humidity
#     [station_names.remove(k) for k in [ 'C09','C06','C11','C12','C16']] # daily wind # 
#     [station_names.remove(k) for k in [ 'C10','C11','C12']]
#     [station_names.remove(k) for k in ['C18']]
#     [station_names.remove(k) for k in [ 'C09','C05']] # Daily U wind # 
#     [station_names.remove(k) for k in [ 'C09','C13', 'C05']] #  U night
#     [station_names.remove(k) for k in [ 'C09','C13']] # daily humidity only head
    # station_names.append('C10')
    # station_names =station_names + AttSta.stations(['Head', 'valley'])
    Files =AttSta.getatt(station_names,'InPath')
    net_LCB=LCB_net()
    net_LCB.AddFilesSta(Files)
#     net_LCB.dropstanan(perc=20, From=From, To = To)
        
#     X_LCB = net_LCB.getvarallsta(var=var, by='D')
#     X_LCB.plot()
    X_LCB = net_LCB.getvarallsta(var=var, by='D', how='min')
#     X_LCB.plot()
#     plt.show()
#     X_LCB = net_LCB.getvarallsta(var=var, by='D', how='max')
#     X_LCB.plot()
#     plt.show()
#     X_LCB = net_LCB.getvarallsta(var=var, by='H')
       
#     X_LCB = X_LCB.between_time('14:00','14:00')
#     X_LCB = net_LCB.getvarallsta(var=var, by='D')
    
    X = X_LCB.dropna(axis=0,how='any')
    params = pd.read_csv('/home/thomas/PhD/supmod/PCA_data/params_topo.csv', index_col=0)
       
    df_verif = X[:50]
    df_train = X[50:]
       
    stamod = StaMod(df_train, AttSta)
    stamod.pca_transform(nb_PC=3, standard=False)
       
    stamod.plot_scores_ts()
#     stamod.pca_reconstruct()

#     params_loadings =  stamod.fit_loadings()
#     print params
#     
#     stamod.plot_loading(params = params_loadings, params_topo= ["Alt","Alt","Alt"])

#     params_loadings =  stamod.fit_loadings(params=["xx8_10000_8000____","xx8_10000_8000____","xx8_10000_8000____"], param_seps=[50,50,0])
#     print params_loadings
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["xx8_10000_8000____","xx8_10000_8000____","xx8_10000_8000____"])
#     params_loadings =  stamod.fit_loadings(params=["Alt","Alt","Alt"], param_seps=[1220,1220,1220])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["Alt","Alt","Alt"])
    

#     
#     def lin(x, a, b, c=0):
#         return a*x + b +c

#     params_loadings =  stamod.fit_loadings(params=["Alt","Alt","Alt"], curvfit=pol)
#     print params_loadings
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["Alt","Alt","Alt"], curvfit=pol)
#     
# #     
#     params_loadings =  stamod.fit_loadings(params=["topex","topex","topex"], curvfit=lin)
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex","topex","topex"], curvfit=lin)
#      
#     params_loadings =  stamod.fit_loadings(params=["topex_n","topex_n","topex_n"])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_n","topex_n","topex_n"])
#      
#     params_loadings =  stamod.fit_loadings(params=["topex_s","topex_s","topex_s"])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_s","topex_s","topex_s"])
# # #         
#      
#     params_loadings =  stamod.fit_loadings(params=["topex_w","topex_w","topex_w"])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_w","topex_w","topex_w"])
# # #      
#     params_loadings =  stamod.fit_loadings(params=["topex_e","topex_e","topex_e"])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_e","topex_e","topex_e"])
# #      
#     params_loadings =  stamod.fit_loadings(params=["topex_ne","topex_ne","topex_ne"], curvfit=func)
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_ne","topex_ne","topex_ne"], curvfit=func)
# #       
#     params_loadings =  stamod.fit_loadings(params=["topex_nw","topex_nw","topex_nw"])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_nw","topex_nw","topex_nw"])
#       
#     params_loadings =  stamod.fit_loadings(params=["topex_se","topex_se","topex_se"], curvfit=func)
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_se","topex_se","topex_se"], curvfit=func)
# # #      
#     params_loadings =  stamod.fit_loadings(params=["topex_sw","topex_sw","topex_sw"])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["topex_sw","topex_sw","topex_sw"])
#       

#     params_loadings =  stamod.fit_loadings(params=["xx8_20000_18000____","xx8_20000_18000____","xx8_20000_18000____"], param_seps=[50,50,0])
#     print params_loadings
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["xx8_20000_18000____","xx8_20000_18000____","xx8_20000_18000____"])
  
#     stamod.pca_reconstruct()
#        
#        
#     inpath = "/home/thomas/PhD/supmod/PCA_data/"



    
    inpath = "/home/thomas/"        
    df_gfs = pd.read_csv(inpath+'gfs_data.csv', index_col =0, parse_dates=True )
#     df_gfs = df_gfs.between_time('15:00','15:00')
#     df_gfs = df_gfs.resample("D").mean()
#     print df_gfs

 #==============================================================================
 # Froude number -> To be implemented in LCBnet_lib or 
 #==============================================================================


#     t1 = df_gfs['TMP_2maboveground']-273.15
#     t2 = df_gfs['TMP_80maboveground']-273.15
#     p1 = df_gfs['PRES_surface']*10**-2
#     p2 = df_gfs['PRES_80maboveground']*10**-2
#     u_mean = (df_gfs['UGRD_10maboveground'] + df_gfs['UGRD_80maboveground'])/2
#     v_mean = (df_gfs['VGRD_10maboveground'] + df_gfs['VGRD_80maboveground'])/2
     

    df_gfs_r = df_gfs.between_time('03:00','03:00')
    df_gfs_r = df_gfs_r.resample("D").mean()
    t1 = df_gfs_r['TMP_900mb']-273.15
    t2 = df_gfs_r['TMP_850mb']-273.15
    p1 = 900.
    p2 = 850
    u_mean = (df_gfs_r['UGRD_850mb'] + df_gfs_r['UGRD_900mb'])/2
    v_mean = (df_gfs_r['VGRD_850mb'] + df_gfs_r['VGRD_900mb'])/2
 
    mean_speed , mean_dir = cart2pol(u_mean, v_mean)
    theta1 = theta(t1, p1)
    theta2 = theta(t2, p2)
       
    br = br(theta1, theta2, 2, 80)  
    fr = froude(br, mean_speed, 80.)
    plt.plot(fr)
    plt.show()

#===============================================================================
# Wind speed
#===============================================================================
#     params_scores = stamod.fit_scores(df_gfs['TMP_2maboveground'])
#     stamod.plot_scores(df_gfs['TMP_2maboveground'])

#     mean_speed , mean_dir = cart2pol( df_gfs['UGRD_10maboveground'],  df_gfs['VGRD_10maboveground'])
#     mean_speed , mean_dir = cart2pol( df_gfs['UGRD_850mb'],  df_gfs['VGRD_850mb'])
#     
#     
#     U_rot, V_rot = PolarToCartesian(mean_speed,mean_dir, rot=-45)
#      
#     plt.scatter(V_rot,df_gfs['VGRD_850mb'])
#     plt.show()
     
#     mean_speed , mean_dir = cart2pol( U_rot, V_rot)

#     params_scores = stamod.fit_scores(mean_speed)
#     stamod.plot_scores(mean_speed)
#     
    
#     print 'VGRD_850mb'
#     params_scores = stamod.fit_scores(df_gfs['VGRD_850mb'])
#     stamod.plot_scores(df_gfs['VGRD_850mb'])
#     
#     print 'UGRD_850mb'
#     params_scores = stamod.fit_scores(df_gfs['TMP_850mb'])
#     stamod.plot_scores(df_gfs['TMP_850mb'])
# #     
#     print 'Urot'
#     params_scores = stamod.fit_scores(U_rot)
#     stamod.plot_scores(U_rot)
# 
#     print 'Vrot'
#     params_scores = stamod.fit_scores(V_rot)
#     stamod.plot_scores(V_rot)
#===============================================================================
# 
#===============================================================================

#     params_scores = stamod.fit_scores(df_gfs['TMP_850mb'])
#     stamod.plot_scores(df_gfs['TMP_850mb'])

#     params_scores = stamod.fit_scores(df_gfs['RH_800mb'])
#     stamod.plot_scores(df_gfs['RH_800mb'])
#     
#     params_scores = stamod.fit_scores(df_gfs['RH_850mb'])
#     stamod.plot_scores(df_gfs['RH_850mb'])
#     
#     params_scores = stamod.fit_scores(df_gfs['RH_900mb'])
#     stamod.plot_scores(df_gfs['RH_900mb'])
#     
#     params_scores = stamod.fit_scores(df_gfs['RH_950mb'])
#     stamod.plot_scores(df_gfs['RH_950mb'])    
    
#     
#     params_scores = stamod.fit_scores(df_gfs['TMP_2maboveground'])
#     stamod.plot_scores(df_gfs['TMP_2maboveground'])
#     
#     params_scores = stamod.fit_scores(df_gfs['VGRD_850mb'])
#     stamod.plot_scores(df_gfs['VGRD_850mb'])


# #   
#     #===========================================================================
#     # Downscalling 
#     #===========================================================================
#     elev  = np.loadtxt('/home/thomas/PhD/supmod/PCA_data/100_15rasterelev.txt',delimiter=',')
#    
#     df = pd.concat([X, df_gfs['VVEL_950mb']], axis=1, join='inner')
#     maps = stamod.predict( elev,df_gfs['VVEL_950mb'][100])
# #     print maps['predicted'].shape
# #     print maps['predicted'].mean(axis=2).shape
#       
#        
# #     plt.contourf(elev, cmap='plasma')
# #     plt.colorbar()
# #     plt.show()
# #     plt.contourf(maps['loadings'][:,:,0], cmap='plasma')
# #     plt.colorbar()
# #     plt.show() 
# #     plt.contourf(maps['loadings'][:,:,1], cmap='plasma')
# #     plt.colorbar()
# #     plt.show()
#     
#     
# #     df = pd.concat([X, df_gfs['TMP_950mb']], axis=1, join='inner')
# #     print df
# #     print X
# #     print df_gfs['TMP_950mb'].index["2015-12-21 03:00:00"]
# #     print df_gfs['TMP_950mb']["2015-12-21 03:00:00"]
# #     print X['C08'][df_gfs['TMP_950mb'].index["2015-12-21 03:00:00"]]
# #     print df.ix[100,:]
# #   
# #     print maps['scores']
#   
#     plt.contourf(maps['predicted'][:,:,0], cmap='plasma', 
#                  levels=np.linspace(maps['predicted'][:,:,0].min(), maps['predicted'][:,:,0].max(),50))
#     plt.colorbar()
#     plt.show()
#     plt.contourf(maps['predicted'][:,:,1], cmap='plasma',
#                  levels=np.linspace(maps['predicted'][:,:,1].min(), maps['predicted'][:,:,1].max(),50))
#     plt.colorbar()
#     plt.show()
#             
#     plt.contourf(maps['predicted'].sum(axis=2), cmap='plasma',
#                  levels=np.linspace(maps['predicted'].sum(axis=2).min(), maps['predicted'].sum(axis=2).max(),50))
#     plt.colorbar()
#     plt.show()
# #      
# #     stamod.plot_scores_ts()
#     
#     #===========================================================================
#     # Verification
#     #===========================================================================
# #     MAE =  stamod.skill(df_verif, df_gfs['TMP_950mb'], metrics = metrics.mean_absolute_error)
# #     MSE =  stamod.skill(df_verif, df_gfs['TMP_950mb'], metrics = metrics.mean_squared_error)
# #     print MAE
# #     print MSE
#      
#        
#      
#      
# 
# 


#===============================================================================
# TEST fit loadings
#=============================================================================== 

 
# x = np.array([1186, 1061, 1075, 1225, 1356, 1113, 1069, 1140, 1206, 1127, 1077, 1031, 1005, 1078, 1342, 1279])
# x = x - x.min()
 
# y = np.array([ 0.2209593 ,  0.27516488,  0.26320392,  0.21992121,  0.22812927, 0.25035334,  0.25173381,  0.2475554 ,  0.2272659 ,  0.25835837,
#               0.26348386,  0.28175971,  0.30190411,  0.25368362,  0.21802705, 0.21954021])
 
 
# z = sorted(zip(x,y))
# x, y = zip(*z)
# 
# x = np.array(x)
# y = np.array(y)
 
# def f_lin(x, a, b):
#     return a*x + b
 
# def f_pol(x, a, b,c):
#     return a*x + x**b +c
 
# def f_log(x, a, b, c):
#     return a*np.log(x) + b
#  
# def f_exp(x,a,b,c):
#     return a*np.exp(-b*x) + c
 
# plt.scatter(x,y, c='r')
 
# popt, pcov = curve_fit(f_lin, x, y)
# print popt
# plt.scatter(x, f_lin(x, *popt), c='b')
     
# popt, pcov = curve_fit(f_pol, x, y)
# plt.scatter(x, f_pol(x, *popt), c='g')
 
# popt, pcov = curve_fit(f_log, x, y)
# plt.scatter(x, f_log(x, *popt), c='k')  
# #     
# popt, pcov = curve_fit(f_exp, x, y, p0=[1,1,0.3])
# plt.scatter(x, f_exp(x, *popt), c='k')    
#     