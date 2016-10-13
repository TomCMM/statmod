#===============================================================================
# DESCRIPTION
#    Downscale SWAT input climatic variable from the 
#    Global Forecasting System in the Ribeirao Das Posses
# Author
#    Thomas Martin
#        PhD atmospheric sciences
#===============================================================================

from statmod_lib import *
from LCBnet_lib import *


if __name__=='__main__':

    #===============================================================================
    # Set up
    #===============================================================================
    InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full_sortcorr/' # Path data
#     Path_att = '/home/thomas/params_topo.csv' # Path attribut
#     AttSta = att_sta(Path_att=Path_att)
    AttSta = att_sta()
    AttSta.setInPaths(InPath)
    From = "2015-03-01 00:00:00"
    To = "2016-01-01 00:00:00"
    
    params = pd.read_csv('/home/thomas/params_topo.csv', index_col=0) # topographic parameters
    
    path_gfs = "/home/thomas/"        
    df_gfs = pd.read_csv(path_gfs+'gfs_data.csv', index_col =0, parse_dates=True ) # GFS data
    df_gfs = df_gfs[From:To]
    print df_gfs
# 
#===========================================================================
# Temperature
#===========================================================================
#     var = "Ta C"
#     station_names =AttSta.stations(['Ribeirao'])
# #     [station_names.remove(k) for k in ['C13','C11','C12','C17','C06']] # Tmin
#     [station_names.remove(k) for k in ['C11','C12','C06']] # Tmin  
# #     [station_names.remove(k) for k in ['C17']] # Tmin  
#     Files =AttSta.getatt(station_names,'InPath')
#     net_LCB=LCB_net()
#     net_LCB.AddFilesSta(Files)
#     X_LCB = net_LCB.getvarallsta(var=var, by='H', how='mean', From=From, To=To)
# #     X_LCB = net_LCB.getvarallsta(var=var, by='H', how='sum', From=From, To=To)
# #     X_LCB = X_LCB.between_time('03:00','03:00')
# #     X_LCB.plot()
#                 
#     X = X_LCB.dropna(axis=0,how='any')
#      
# #     df_verif = X[-len(X)/3:]
# #     df_train = X[:-len(X)/3]
# #     print X_LCB.index
# #     print df_verif.index
# #     print df_train.index
# #     print df_gfs.index
#      
#     df_train= X
#  
#     stamod = StaMod(df_train, AttSta)
#     stamod.pca_transform(nb_PC=4, standard=False)
#                    
#     stamod.plot_scores_ts()
#     stamod.plot_exp_var()
#     plt.show()
# #               
# #     # FIT LOADINGS
#     params_loadings =  stamod.fit_loadings(params=["Alt","Alt","Alt","Alt"], fit=[pol,pol,pol,pol])
#     stamod.plot_loading(params_fit = params_loadings[0], params_topo= ["Alt","Alt","Alt","Alt"], fit=[pol,pol,pol,pol])
#           
#     #fit scores
# #     df_gfs_r = df_gfs.between_time('03:00','03:00')
# #     print df_gfs_r
# #     df_gfs_r = df_gfs.resample("D").min()
#     df_gfs_r  = df_gfs
#         
#     T = df_gfs_r['TMP_2maboveground']-273.15
#                
#     t1 = df_gfs_r['TMP_2maboveground']-273.15
#     t2 = df_gfs_r['TMP_80maboveground']-273.15
#           
# #     t1 = df_gfs_r['TMP_950mb']-273.15
# #     t2 = df_gfs_r['TMP_800mb']-273.15
#  
# #     p1 = 950
# #     p2 = 800
#           
#     p1 = df_gfs_r['PRES_surface']*10**-2
#     p2 = df_gfs_r['PRES_80maboveground']*10**-2
# #     p2 = 850
#     u_mean = (df_gfs_r['UGRD_10maboveground'] + df_gfs_r['UGRD_80maboveground'])/2
#     v_mean = (df_gfs_r['VGRD_10maboveground'] + df_gfs_r['VGRD_80maboveground'])/2
#             
#     mean_speed , mean_dir = cart2pol(u_mean, v_mean)
#     theta1 = theta(t1, p1)
#     theta2 = theta(t2, p2)
#                   
# #     br_ = br(theta1, theta2, 2, 80)  
# #     fr = froude(br_, mean_speed, 80.)
#     br_ = br(theta1, theta2, 2, 80)  
#     fr = froude(br_, mean_speed, 80.)
#     fr = -np.log(fr)
# #     print T
# #     print fr
#    
#     params_scores = stamod.fit_scores([T,fr,T,T], fit=[lin,lin,lin,lin])
# #     print params_scores
#     stamod.plot_scores([T,fr,T,T], params_fit = params_scores[0], fit=[lin,lin,lin,lin])


#===========================================================================
# Specific humidity
#===========================================================================
#     var = "Ua g/kg"
#     station_names =AttSta.stations(['Ribeirao'])
# #     [station_names.remove(k) for k in ['C11', 'C09','C17','C18','C19','C16']] # Tmin
# #     [station_names.remove(k) for k in ['C13','C11','C09']] # Tmin
# #     [station_names.remove(k) for k in ['C13','C11','C12']] 
#     [station_names.remove(k) for k in ['C13','C16','C11','C12']]
#          
#     Files =AttSta.getatt(station_names,'InPath')
#     net_LCB=LCB_net()
#     net_LCB.AddFilesSta(Files)
#     X_LCB = net_LCB.getvarallsta(var=var, by='H',how='mean', From=From, To=To)
# #     X_LCB = X_LCB.between_time('03:00','03:00')
#     X_LCB.plot()
#             
#     X = X_LCB.dropna(axis=0,how='any')
# #     df_verif = X[:50]
# #     df_train = X[50:]
#     df_train= X
#     stamod = StaMod(df_train, AttSta)
#     stamod.pca_transform(nb_PC=4, standard=False)
#                
#     stamod.plot_scores_ts()
#     stamod.plot_exp_var()
#     plt.show()
#              
#     # FIT LOADINGS
#     params_loadings =  stamod.fit_loadings(params=["Alt","Alt","Alt","Alt"], curvfit=[pol,pol,pol,pol])
#     stamod.plot_loading(params = params_loadings[0], params_topo= ["Alt","Alt","Alt","Alt"], curvfit=[pol,pol,pol,pol])
          
# #     #fit scores
# #     df_gfs_r = df_gfs.resample("D").mean()
# #     df_gfs_r = df_gfs.between_time('03:00','03:00')
# #     df_gfs_r = df_gfs
#     
#     Rh = df_gfs_r['RH_850mb']
#           
#     t1 = df_gfs_r['TMP_2maboveground']-273.15
#     t2 = df_gfs_r['TMP_80maboveground']-273.15
#     p1 = df_gfs_r['PRES_surface']*10**-2
#     p2 = df_gfs_r['PRES_80maboveground']*10**-2
# #     p2 = 850
#     u_mean = (df_gfs_r['UGRD_10maboveground'] + df_gfs_r['UGRD_80maboveground'])/2
#     v_mean = (df_gfs_r['VGRD_10maboveground'] + df_gfs_r['VGRD_80maboveground'])/2
#        
#     mean_speed , mean_dir = cart2pol(u_mean, v_mean)
#     theta1 = theta(t1, p1)
#     theta2 = theta(t2, p2)
#              
#     br_ = br(theta1, theta2, 2, 80)  
#     fr = froude(br_, mean_speed, 80.)
# #     fr = np.log(fr)
#     print fr
# #      
#     params_scores = stamod.fit_scores([Rh, fr])
#     stamod.plot_scores([Rh, fr], params = params_scores)
# #   
    
#===========================================================================
# Wind speed
#===========================================================================
    var = "Dm G"
    station_names =AttSta.stations(['Head'])
#     [station_names.remove(k) for k in [ 'C09','C06','C13','C16']] # Sm m/s # 
#     [station_names.remove(k) for k in ['C06','C16']] # V m/s # 
#     [station_names.remove(k) for k in ['C13']] # U m/s # 
    [station_names.remove(k) for k in ['C06']] # head SM m/s #     
       
    print station_names
    Files =AttSta.getatt(station_names,'InPath')
    net_LCB=LCB_net()
    net_LCB.AddFilesSta(Files)
    Path_att = '/home/thomas/params_topo.csv' # Path attribut
    AttSta = att_sta(Path_att=Path_att)
       
       
    X_LCB = net_LCB.getvarallsta(var=var, by='3H', From=From, To=To)
#     X_LCB = X_LCB.between_time('03:00','03:00')
         
#     X_LCB.plot()
             
    X = X_LCB.dropna(axis=0,how='any')
    df_verif = X[-len(X)/3:]
    df_train = X[:-len(X)/3]
#     df_train= X
     
    stamod = StaMod(df_train, AttSta)
    stamod.pca_transform(nb_PC=4, standard=False)
                
    stamod.plot_scores_ts()
    stamod.plot_exp_var()
#     plt.show()
#     # FIT LOADINGS
    params_loadings =  stamod.fit_loadings(params=["Alt","Alt",'Alt',"Alt"], fit=[lin,lin,lin,lin])
    stamod.plot_loading(params_fit = params_loadings[0], params_topo= ["Alt","Alt",'Alt',"Alt"], fit=[lin,lin,lin,lin])
          
#     params_loadings =  stamod.fit_loadings(params=["topex","topex_w",'Alt'], fit=[lin,lin,lin])
#     stamod.plot_loading(params_fit = params_loadings[0], params_topo= ["topex","topex_w",'Alt'], fit=[lin,lin,lin])
              
        
    #fit scores
#     df_gfs_r = df_gfs.resample("3H").mean()
    df_gfs_r = df_gfs
       
    U = df_gfs_r['UGRD_850mb']
    V =  df_gfs_r['VGRD_850mb']
            
    mean_speed , mean_dir = cart2pol( U, V )
    U_rot, V_rot = PolarToCartesian(mean_speed,mean_dir, rot=-45)
       
    params_scores = stamod.fit_scores([mean_speed, V_rot, df_gfs_r['UGRD_10maboveground']],fit=[lin,lin,lin])
     
#     [T,fr], params_fit = params_scores[0], fit=[lin,lin]
     
    stamod.plot_scores([mean_speed, V_rot, df_gfs_r['UGRD_10maboveground']], params_fit = params_scores[0],fit=[lin,lin,lin])
       
#===============================================================================
# Model verification Temeperature
#===============================================================================
#     res = stamod.predict(stamod.topo_index, [T, fr,T])
#     print res['loadings'].shape
#     print res['scores'].shape
#     print res['predicted'].shape
#     MAE =  stamod.skill(df_verif,[T, fr,T] , metrics = metrics.mean_absolute_error)
#     print MAE

#===============================================================================
# Model verification Wind speed
#===============================================================================
#     res = stamod.predict(stamod.topo_index, [mean_speed, V_rot, df_gfs_r['UGRD_10maboveground']])
#     print res['loadings'].shape
#     print res['scores'].shape
#     print res['predicted'].shape
#     MAE =  stamod.skill(df_verif,[mean_speed, V_rot, df_gfs_r['UGRD_10maboveground']] , metrics = metrics.mean_absolute_error)
#     print MAE
#     MAE =  stamod.skill(df_verif,[mean_speed, V_rot, df_gfs_r['UGRD_10maboveground']] , metrics = metrics.mean_squared_error)
#     print MAE







    