#===============================================================================
# DESCRIPTION
#    Downscale climatic variable in the Serra Da Mantiquera, Serra do Mar from the 
#    Global Forecasting System to make input for the meteorological model ARPS
#===============================================================================
from statmod_lib import *
from LCBnet_lib import *
from mapstations import plot_local_stations
from numpy.testing.utils import measure

if __name__=='__main__':

    AttSta = att_sta()
    AttSta.addatt(path_df = '/home/thomas/arps_coldpool.csv')
    print AttSta.showatt()
    var = 'Ta C'
    From = "2015-02-01 00:00:00"
    To = "2015-11-01 00:00:00"
    
#     path_gfs = "/home/thomas/"        
#     df_gfs = pd.read_csv(path_gfs+'gfs_data.csv', index_col =0, parse_dates=True ) # GFS data

    #===============================================================================
    # Set up
    #===============================================================================
#     Lat = [-23.5,-21.5]
#     Lon = [-47.5, -45]
#     Alt = [400,5000]
# # #     
    Lat = [-25,-21]
    Lon = [-49, -44]
    Alt = [0,5000]

    
#         Cantareira region
#     Lat = [-23.5,-21.5]
#     Lon = [-47.5, -45.5]  
#     Alt = [400,5000]

#     Lat = [-23,-22]
#     Lon = [-47, -46]
#     Alt = [400,5000]


    net_sinda = LCB_net()
    net_inmet = LCB_net()
    net_iac = LCB_net()
    net_LCB = LCB_net()
            
    # AttSta.setInPaths(InPath)
    
       
    Path_Sinda = '/home/thomas/PhD/obs-lcb/staClim/Sinda/obs_clean/Sinda/'
    Path_INMET ='/home/thomas/PhD/obs-lcb/staClim/INMET-Master/obs_clean/INMET/'
    Path_IAC ='/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs_clean/IAC/'
    Path_IAC ='/home/thomas/PhD/obs-lcb/staClim/IAC-Monica/obs_clean/IAC/'
    Path_LCB='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/' # Path data

    print Path_LCB
    
    AttSta_IAC = att_sta()
    AttSta_Inmet = att_sta()
    AttSta_Sinda = att_sta()
    AttSta_LCB = att_sta()
    
    
    AttSta_IAC.setInPaths(Path_IAC)
    AttSta_Inmet.setInPaths(Path_INMET)
    AttSta_Sinda.setInPaths(Path_Sinda)
    AttSta_LCB.setInPaths(Path_LCB)
# 
    stanames_IAC =  AttSta.stations(values=['IAC'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt})
    stanames_Inmet = AttSta.stations(values=['Innmet'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt} )
    stanames_Sinda = AttSta.stations(values=['Sinda'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt} )
    stanames_LCB = AttSta_LCB.stations(values = ['Head'], params={'Lat':Lat, 'Lon':Lon, 'Alt':Alt})



# # ## GET STATIONS WORKING IN DOMAIN
    
#     lats = AttSta.getatt(stanames, 'Lat')
#     lons = AttSta.getatt(stanames, 'Lon')
#     stanames = pd.Series(stanames)
#     lats = pd.Series(lats)
#     lons = pd.Series(lons)
#     print lats
#     stats = pd.concat([lats,lons, stanames], axis=1)
#     stats.columns = ['lats', 'lons', 'stanames']
#     stats.to_csv('/home/thomas/stations_in_domain.csv')

    

# ##    Position
#     measure = AttSta.getatt(stanames_LCB, 'Alt')
#     arps = AttSta.getatt(stanames_LCB,'ZP' )
#     plt.scatter(measure, arps, c='b')
#     plt.scatter(measure, measure, c='r')
#     plt.show()

#     stanames_IAC = ['ma46','su81','bj13','iu37','ph100','pr141']
 
#     stanames_IAC =  AttSta.stations(values=['IAC'])
#     stanames_Inmet = AttSta.stations(values=['Innmet'])
#     stanames_Sinda = AttSta.stations(values=['Sinda'])
#     stanames_LCB = AttSta_LCB.stations(values = ['Head'])
#     

#     [stanames_IAC.remove(x) for x in ['cn15','cb17'] if x in stanames_IAC ] # T % weird values
    [stanames_IAC.remove(x) for x in ['pc58','sb69','ma46','ii31'] if x in stanames_IAC ] # T % weird values
#     [stanames_IAC.remove(x) for x in ['jn38','bj13'] if x in stanames_IAC ] # Ua % weird values
    [stanames_LCB.remove(x) for x in ['C18'] if x in stanames_LCB ] # T % weird values
#     [stanames_LCB.remove(x) for x in ['C11','C12'] if x in stanames_LCB ] # T % weird values
#     [stanames_Inmet.remove(x) for x in ['A555'] if x in stanames_Inmet ]


#     [stanames_IAC.remove(x) for x in ['no50','iu229', 'ce218', 'pb108', 'ma46','mb207','dv136','sg140','jp22','mr230'] if x in stanames_IAC ] # INCORRECT HEIGHT
#     [stanames_Inmet.remove(x) for x in ['A706', 'A531','A740','A530'] if x in stanames_Inmet ]

#     [stanames_IAC.remove(x) for x in ['dv136','su81','sj70','ia32','bj13','va89', 'at7','ep22','pc58','br14','cs3'] if x in stanames_IAC ] # wind weird 
#     [stanames_LCB.remove(x) for x in ['C09'] if x in stanames_LCB ] # windweird values

    stanames =  stanames_IAC + stanames_Inmet + stanames_LCB

    Files_IAC =AttSta_IAC.getatt(stanames_IAC,'InPath')
    Files_Inmet =AttSta_Inmet.getatt(stanames_Inmet,'InPath')
    Files_Sinda =AttSta_Sinda.getatt(stanames_Sinda,'InPath')
    Files_LCB =AttSta_LCB.getatt(stanames_LCB,'InPath')
      
    net_sinda.AddFilesSta(Files_Sinda, net='Sinda')
    net_inmet.AddFilesSta(Files_Inmet, net='INMET')
    net_iac.AddFilesSta(Files_IAC, net='IAC')
    net_LCB.AddFilesSta(Files_LCB)
      
      
    net_iac.dropstanan(perc=15, From=From, To = To)
    net_inmet.dropstanan(perc=15, From=From, To = To)
    net_sinda.dropstanan(perc=15, From=From, To = To, by='3H')
  
      
    X_iac = net_iac.getvarallsta(var=var,by='H',From=From, To = To)
    X_inmet = net_inmet.getvarallsta(var=var,by='H',From=From, To = To)
#     X_sinda = net_sinda.getvarallsta(var=var,by='H',From=From, To = To)
    X_LCB = net_LCB.getvarallsta(var=var, by='H', From=From, To = To )

    X = pd.concat([ X_iac, X_LCB], axis=1)
    X = X.between_time('12:00','12:00')
#     X.plot(subplots=True)
#     plt.show()   
    X = X.fillna(X.mean(), axis=0)
#     X.plot( style='o-')
#     plt.show()
#  
    X = X.dropna(axis=0,how='any') 
    print X.shape
     
# ## Plot altitude measured vs estimated
#     Alt = AttSta.getatt(stanames, 'Alt') 
#     ZP = AttSta.getatt(stanames, 'ZP')
#      
#     plt.scatter(Alt,ZP)
#     plt.plot(Alt,Alt,'r')
#      
#     for i, txt in enumerate(stanames):
#         plt.annotate(txt, (Alt[i],ZP[i]))

#     #===========================================================================
#     # PCA
#     #===========================================================================
#     
    stamod = StaMod(X, AttSta)
    stamod.pca_transform(nb_PC=4, standard=False)

    stamod.plot_exp_var()
#     plt.show()
    stamod.plot_scores_ts()

## GET COLD POOL EVENTS
#     print stamod.scores.loc[:,2]
#     s = pd.Series(index = stamod.scores.index[stamod.scores.loc[:,2] > 15])
# #     s.index = s.index - pd.Timedelta(days=1) + pd.Timedelta(hours=6)# 9am
#     s.index = s.index - pd.Timedelta(hours=6) # 9pm
#     print s
#     s.to_csv('/home/thomas/cold_poolevents.csv')


    
#     # FIT LOADINGS
    params_loadings = stamod.fit_loadings(params=["Lat","Lat",'Lon','Lon',], curvfit=[lin,lin,lin,lin])
    stamod.plot_loading(params = params_loadings[0],params_topo= ["Lat","Lat",'Lon','Lon'], curvfit=[lin,lin,lin,lin])
  
# #     # plot map
#     topo_val  = np.loadtxt('/home/thomas/map_lon44_49_lat20_25_r@PERMANENT',delimiter=',')
#     topo_val = topo_val[::-1,:]
#     topo_lat  = np.loadtxt('/home/thomas/latitude.txt',delimiter=',')
#     topo_lon  = np.loadtxt('/home/thomas/longitude.txt',delimiter=',')
# #    
#     eigenvectors = stamod.eigenvectors.loc[1,:]
#     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=True)
#  
#     eigenvectors = stamod.eigenvectors.loc[2,:]
#     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=True)
# # #         
#     eigenvectors = stamod.eigenvectors.loc[3,:]
#     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=False)
    
#     eigenvectors = stamod.eigenvectors.loc[4,:]
#     plot_local_stations(AttSta, topo_lat, topo_lon, topo_val, eigenvectors=eigenvectors,pos_sta_net=False, annotate=False)
# 
# #   
# #   
# # #     # FIT SCORES    
# #    
#     df_gfs_r = df_gfs.between_time('03:00','03:00')
#             
#     T = df_gfs_r['TMP_2maboveground']-273.15
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
#     fr = np.log(fr)
#      
#          
#     print T
#     print fr
#     params_scores = stamod.fit_scores([T, fr,df_gfs_r['VGRD_850mb'] *df_gfs_r['TMP_850mb']  ,df_gfs_r['RH_850mb']])
#     print params_scores
#     stamod.plot_scores([T, fr, df_gfs_r['VGRD_850mb']*df_gfs_r['TMP_850mb'], df_gfs_r['RH_850mb']], params = params_scores)
#    
# #      
#       
#      
#      
    
    
    
    
    