#===============================================================================
# DESCRIPTION
#    Downscale climatic variable in the Serra Da Mantiquera, Serra do Mar from the 
#    Global Forecasting System to make input for the meteorological model ARPS
#===============================================================================

from statmod_lib import *
from clima_lib.LCBnet_lib import *
# from mapstations import plot_local_stations
from numpy.testing.utils import measure
import datetime
import pickle
import sys

if __name__=='__main__':
    plt.style.use('ggplot')
    import os
    cwd = os.getcwd()
    print cwd
    AttSta = att_sta()
#     AttSta.addatt(path_df = '/home/thomas/arps_coldpool.csv')
#     AttSta.addatt(path_df ='/home/thomas/params_topo.csv')
    var = 'Ta C'
    From = "2015-03-01 00:00:00"
    To = "2015-11-15 00:00:00"
      
#     path_gfs = "/home/thomas/"        
#     df_gfs = pd.read_csv(path_gfs+'gfs_data.csv', index_col =0, parse_dates=True ) # GFS data
  
  
# #===============================================================================
# # Create GFS predictors dataframe
# #===============================================================================
#     path_gfs = "/home/thomas/"        
#     df_gfs = pd.read_csv(path_gfs+'gfs_data_levels_analysis.csv', index_col =0, parse_dates=True ) # GFS data
#     del  df_gfs['dirname']
#     del  df_gfs['basename']
#     del  df_gfs['time']
#     del  df_gfs['model']
#     del  df_gfs['InPath']
#     df_gfs = df_gfs.dropna(axis=1,how='all') 
#     df_gfs = df_gfs.dropna(axis=0,how='all')
#     print df_gfs
#  
#   
#===============================================================================
# Create surface observations dataframe
#===============================================================================
  
#------------------------------------------------------------------------------ 
#    Select stations
#------------------------------------------------------------------------------ 
#      Cantareira sistema
#     Lat = [-23.5,-21.5]
#     Lon = [-47.5, -45.5]  
#     Alt = [400,5000]
  
#    Serra Da Mantiquera
    Lat = [-24,-21]
    Lon = [-49, -45]
    Alt = [400,5000]
  
    net_sinda = LCB_net()
    net_inmet = LCB_net()
    net_iac = LCB_net()
    net_LCB = LCB_net()
    net_svg =  LCB_net()
    net_peg =  LCB_net()
                 
#     Path_Sinda = '/home/thomas/PhD/obs-lcb/staClim/Sinda/obs_clean/Sinda/'
    Path_INMET ='/home/thomas/phd/obs/staClim/inmet/full/'
    Path_IAC ='/home/thomas/phd/obs/staClim/iac/data/full/'
    Path_LCB='/home/thomas/phd/obs/lcbdata/obs/full_sortcorr/'
    Path_svg='/home/thomas/phd/obs/staClim/svg/SVG_2013_2016_Thomas_30m.csv'
    Path_peg='/home/thomas/phd/obs/staClim/peg/Th_peg_tar30m.csv'
    
    AttSta_IAC = att_sta()
    AttSta_Inmet = att_sta()
#     AttSta_Sinda = att_sta()
    AttSta_LCB = att_sta()
       
    AttSta_IAC.setInPaths(Path_IAC)
    AttSta_Inmet.setInPaths(Path_INMET)
#     AttSta_Sinda.setInPaths(Path_Sinda)
    AttSta_LCB.setInPaths(Path_LCB)
  
    stanames_IAC =  AttSta.stations(values=['IAC'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt}) # this does not work anymore
    stanames_Inmet = AttSta.stations(values=['Innmet'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt} )
#     stanames_Sinda = AttSta.stations(values=['Sinda'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt} )
    stanames_LCB = AttSta_LCB.stations(values = ['Ribeirao'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt})
      
    [stanames_IAC.remove(x) for x in ['pc58','sb69'] if x in stanames_IAC ] # Remove stations
#     [stanames_LCB.remove(x) for x in ['C10','C17','C12','C14','C08'] if x in stanames_LCB ] # Remove stations
    [stanames_Inmet.remove(x) for x in ['A706','A509', 'A531','A530'] if x in stanames_Inmet ] # Remove stations 
     
    stanames =  stanames_IAC + stanames_Inmet + stanames_Inmet 
     
#------------------------------------------------------------------------------ 
# Create Dataframe
#------------------------------------------------------------------------------ 
    Files_IAC =AttSta_IAC.getatt(stanames_IAC,'InPath')
    Files_Inmet =AttSta_Inmet.getatt(stanames_Inmet,'InPath')
#     Files_Sinda =AttSta_Sinda.getatt(stanames_Sinda,'InPath')
    Files_LCB =AttSta_LCB.getatt(stanames_LCB,'InPath')
       
#     net_sinda.AddFilesSta(Files_Sinda, net='Sinda')
    net_inmet.AddFilesSta(Files_Inmet, net='INMET')
    net_iac.AddFilesSta(Files_IAC, net='IAC')
    net_LCB.AddFilesSta(Files_LCB)
    net_svg.AddFilesSta([Path_svg], net='svg')
    net_peg.AddFilesSta([Path_peg], net='peg')
  
    df_iac = net_iac.getvarallsta(var=var,by='H',From=From, To = To)
    df_inmet = net_inmet.getvarallsta(var=var,by='H',From=From, To = To)
#     X_sinda = net_sinda.getvarallsta(var=var,by='3H',From=From, To = To)
    df_LCB = net_LCB.getvarallsta(var=var, by='H', From=From, To = To )
    df_svg = LCB_station(Path_svg, net='svg').getData(var=var, by='H', From=From, To = To )
    df_svg.columns =['svg']
    df_peg = LCB_station(Path_peg, net='peg').getData(var=var, by='H', From=From, To = To )
    df_peg.columns =['peg']
#     
#     
#     print df_peg
#     print df_svg
    df = pd.concat([df_iac,df_LCB, df_peg, df_svg], axis=1)
#     df = df.between_time('12:00','12:00')
#     df = df.resample("H").mean()
#     df = df.T
#     df.plot(legend=False)
#     plt.xlabel('Date')
#     plt.ylabel('Temperture (C)')
#     
#     plt.show()   
   
#     df = df.fillna(df.mean(), axis=0)
    df = df.dropna(axis=0,how='any')
    df.to_csv('../data/neuralnetwork/df_sta.csv')
  
#===============================================================================
# Create train and verify dataframe
#===============================================================================
# #------------------------------------------------------------------------------ 
# # Select same index
# #------------------------------------------------------------------------------
#     df = df[df.index.isin(df_gfs.index)]
#     df_gfs = df_gfs[df_gfs.index.isin(df.index)]
#  
#   
# #------------------------------------------------------------------------------ 
# # train dataset
# #------------------------------------------------------------------------------ 
# #     df_train = df[:-len(X)/7]
#     df_train = df
#     df_gfs_train = df_gfs[df_gfs.index.isin(df_train.index)]
#   
# #------------------------------------------------------------------------------ 
# # verify dataset
# #------------------------------------------------------------------------------ 
#     #     df_verif = df[-len(X)/7:]
#     df_verif=df
#     df_gfs_verif = df_gfs[df_gfs.index.isin(df_verif.index)]
#     df_verif = df_verif[df_verif.index.isin(df_gfs_verif.index)]
  
#===========================================================================
# Create model
# #=========================================================================== 
#------------------------------------------------------------------------------ 
#    PCA
#------------------------------------------------------------------------------ 
    stamod = StaMod(df, AttSta)
    stamod.pca_transform(nb_PC=6, standard=True, center =True)
    stamod.plot_exp_var()
    
    stamod.plot_scores_ts()
    stamod.scores.astype(float).to_csv('../data/neuralnetwork/scores_pca_sta_coldpool_fullperiod.csv',float_format=True)
#     stamod.eigenvectors.iloc[:,:].to_csv('../data/neuralnetwork/loadings_pca_sta_coldpool.csv')
    stamod.eigenvectors.T.astype(float).to_csv('../data/neuralnetwork/loadings_pca_sta_coldpool_fullperiod.csv',  float_format=True)

#     print stamod.eigenvectors.iloc[:,:]
#     stamod.eigenvectors.T.iloc[:,:].plot()
#     plt.show()
    
# #------------------------------------------------------------------------------ 
# #    Fit loadings
# #------------------------------------------------------------------------------ 
    params_loadings = stamod.fit_loadings(params=["Alt","Alt","Alt","Lon","Alt","Alt","Alt","Alt","Alt","Alt","Alt"], fit=[lin,lin,lin,pol2, lin, lin, lin, lin, lin, lin, lin])
    stamod.plot_loading(params_fit = params_loadings[0],params_topo= ["Alt","Alt","Alt","Lat","Alt","Alt","Alt","Alt","Alt","Alt","Alt"], fit=[lin,lin,lin,pol2, lin, lin, lin, lin, lin, lin, lin])
    
    fig, ax = plt.subplots()
    plt.scatter(stamod.eigenvectors.loc[3,:], stamod.eigenvectors.loc[4,:])
    for i, txt in enumerate(stamod.eigenvectors.columns):
                ax.annotate(txt, (stamod.eigenvectors.iloc[2,i], stamod.eigenvectors.iloc[3,i]))
    plt.show()
    pickle.dump(stamod.eigenvectors, open( "save.p", "wb" ))

# #     
# #     
#     
#     pickle.dump(stamod.eigenvectors, open( "save.p", "wb" ))


# 
# #===============================================================================
# # TEST ARPS index
# #===============================================================================
#     latsarps = pd.read_csv('/home/thomas/Lat.csv', index_col=0, header=None)
#     lonsarps = pd.read_csv('/home/thomas/Lon.csv', index_col=0, header=None)
#     
# 
#     latsarps = latsarps.iloc[1,:].values
#     lonsarps = lonsarps.iloc[1,:].values
#     print lonsarps
#     
#     print latsarps.min()
#     print latsarps.max()
#     print lonsarps.min()
#     print lonsarps.max()
#     
#     arps_corr_pc2 = pd.read_csv('/home/thomas/arps_corr_PC2.csv', index_col=0, header=None)
#      
#     def geo_idx(dd, dd_array):
#         """
#           search for nearest decimal degree in an array of decimal degrees and return the index.
#           np.argmin returns the indices of minium value along an axis.
#           so subtract dd from all values in dd_array, take absolute value and find index of minium.
#          """
# 
#         geo_idx = (np.abs(dd_array - dd)).argmin()
#         return geo_idx
#  
#     stalats = np.array(AttSta.getatt(df.columns, "Lat")).astype(np.float)
#     stalons = np.array(AttSta.getatt(df.columns, "Lon")).astype(np.float)
#     print stalats
#  
#     lat_idx = []
#     lon_idx = []
#      
#     for lat,lon in zip(stalats, stalons):
#          
#         lat_idx.append(geo_idx(float(lat), latsarps))
#         lon_idx.append(geo_idx(float(lon), lonsarps))
# 
#     arps_corr_pc2 = np.reshape(arps_corr_pc2.values, (1201,1201))
#     plt.scatter(arps_corr_pc2[lon_idx, lat_idx], stamod.eigenvectors.iloc[2,:])
#     plt.show()
#  
 
#------------------------------------------------------------------------------ 
#    Fit PCs
#------------------------------------------------------------------------------ 
#     stamod.stepwise(df_gfs_train,lim_nb_predictors=4)
  
# #===============================================================================
# # Field Results
# # #===============================================================================
#     load field
#     topo_val  = np.loadtxt('/home/thomas/ASTGTM2_S23W047_dem@PERMANENT',delimiter=',')
#     lat  = np.loadtxt('/home/thomas/latitude.txt',delimiter=',')
#     lon  = np.loadtxt('/home/thomas/longitude.txt',delimiter=',')
#   
#     shape_topo_val = topo_val.shape
#   
#     print shape_topo_val
#   
# #     res = stamod.predict_model([topo_val.flatten()]*2, df_gfs_verif.loc["2015-11-11 03:00:00":"2015-11-11 03:00:00",:])
#     res = stamod.predict_model([topo_val.flatten()]*2, df_gfs_verif.iloc[10:11,:])
#   
#   
#     data = res['predicted'].sum(axis=2)
#   
# #     np.savetxt('/home/thomas/data.txt', data,delimiter=',')
#        
#     print 'Plot'*80
#     # map visualisation
#     data = data.reshape(shape_topo_val)
#     plt.contourf(data, levels = np.linspace(data.min(), data.max(), 100))
#     plt.colorbar()
#     plt.show()
#   
#   
# #===============================================================================
# # Create ADAS input 
# #===============================================================================
#     date = datetime.datetime.strptime("2015-11-11", '%Y-%m-%d')# date of the file
#     hour=datetime.datetime.strptime("03:00:00", '%H:%M:%S')# hour of the file
#     outpath = '/home/thomas/surfass.lso'
#     
#     
#     topo_val = topo_val.flatten()
#     lat = lat.flatten()
#     lon = lon.flatten()
#     data = data.flatten()
#     print data
#     print data.shape
#      
#     stamod.to_adas(data, lat, lon, topo_val, date, hour, outpath)
# #     
# #===============================================================================
# # model verification temperature directly from the stepwise regression
# #===============================================================================
# 
# #     res = stamod.predict_model(stamod.topo_index, df_gfs_verif)
# #     MAE =  stamod.skill_model(df_verif,res , metrics = metrics.mean_absolute_error, plot=True)
# #     MSE=  stamod.skill_model(df_verif,res , metrics = metrics.mean_squared_error, plot=False)
# #     
# #     print "X"* 20
# #     print "Results"
# #     print "Spatially averaged MSE: " + str (MSE.mean())
# #     print "Spatially averaged MAE: " + str (MAE.mean())
# #     print "X"* 20
# 
# 
# #===============================================================================
# # GFS temperature Error
# #===============================================================================
# # # gfs
# #     df_rec = pd.DataFrame(index= df_verif.index)
# #     for sta in df_verif.columns:
# #         df_rec = pd.concat([df_rec, df_gfs["TMP_2maboveground"]], axis=1, join="outer") 
# #     df_rec.columns = df_verif.columns
# #     df_rec = df_rec-273.15
# # # #     error = df_verif - df_gfs_verif
# #     from sklearn.metrics import mean_absolute_error, mean_squared_error
# # #      
# # #      
# #     print "X"* 20
# #     print "Results"
# #     print "Spatially averaged MSE: " + str (mean_squared_error(df_verif, df_rec))
# #     print "Spatially averaged MAE: " + str (mean_absolute_error(df_verif, df_rec))
# #     print "X"* 20
# 
# 
# #===============================================================================
# # Plot loadings on a map
# #===============================================================================
# # #     # plot map
# #     topo_val  = np.loadtxt('/home/thomas/ASTGTM2_S23W047_dem@PERMANENT',delimiter=',')
# #     topo_val = topo_val[::-1,:]
# #     plt.contourf(topo_val, levels = np.linspace(topo_val.min(), topo_val.max(), 100))
# #     topo_lat  = np.loadtxt('/home/thomas/latitude.txt',delimiter=',')
# #     topo_lon  = np.loadtxt('/home/thomas/longitude.txt',delimiter=',')
# # #    
# #     eigenvectors = stamod.eigenvectors.loc[1,:]
# #     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=True)
# #  
# #     eigenvectors = stamod.eigenvectors.loc[2,:]
# #     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=True)
# # # #         
# #     eigenvectors = stamod.eigenvectors.loc[3,:]
# #     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=False)
#     
# #     eigenvectors = stamod.eigenvectors.loc[4,:]
# #     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=False)
# # 
# 
# #===============================================================================
# # Drop stations based on the frequency of nan
# #===============================================================================
# #     net_iac.dropstanan(perc=15, From=From, To = To)
# #     net_inmet.dropstanan(perc=15, From=From, To = To)
# #     net_sinda.dropstanan(perc=15, From=From, To = To, by='3H')
# 
# #===============================================================================
# # GET COLD POOL EVENTS
# #===============================================================================
# #     print stamod.scores.loc[:,2]
# #     s = pd.Series(index = stamod.scores.index[stamod.scores.loc[:,2] > 15])
# # #     s.index = s.index - pd.Timedelta(days=1) + pd.Timedelta(hours=6)# 9am
# #     s.index = s.index - pd.Timedelta(hours=6) # 9pm
# #     print s
# #     s.to_csv('/home/thomas/cold_poolevents.csv')
# 
# 
# #===============================================================================
# # Dem against measured altitude
# #===============================================================================
#     
# # # ## Plot altitude measured vs estimated
# #     Alt = AttSta.getatt(stanames, 'Alt') 
# #     ZP = AttSta.getatt(stanames, 'Alt_dem')
# #        
# #     plt.scatter(Alt,ZP)
# #     plt.xlabel('Measured altitude')
# #     plt.ylabel('Dem altitude')
# #     plt.plot(Alt,Alt,'r')
# #        
# #     for i, txt in enumerate(stanames):
# #         plt.annotate(txt, (Alt[i],ZP[i]))
# #     plt.show()