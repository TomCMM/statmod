#===============================================================================
# DESCRIPTION
#    As I have a lot of data to perform the PCA, it take time to prepare the data
#    So this script aims to create a dataframe wit all the necessary variables
#===============================================================================
from LCBnet_lib import *
from Irradiance import Irradiance_sim_obs
import glob
LCB_Irr = Irradiance_sim_obs.LCB_Irr

 
   
   
#===============================================================================
# Variables from Meteorological station
#===============================================================================
InPath='/home/thomas/PhD/obs-lcb/LCBData/obs/Full/'
Files=glob.glob(InPath+"*")
      
net=LCB_net()
AttSta = att_sta()
AttSta.setInPaths(InPath)
AttSta.showatt()
      
stanames = AttSta.stations(['Ribeirao'])
      
staPaths = AttSta.getatt(stanames , 'InPath')
net.AddFilesSta(staPaths)
df_stations = net.getvarallsta(var='Ta C', all= True, by='H')
     
 
#===============================================================================
# Irradiance from R.sun accounting for the observed ratio
# Altitude from DEM
#===============================================================================
# Note
# I am using the elevation from the DEm for ease.
# However in the future, I should put the observed altitude 
inpath = "/home/thomas/"
stanames = ['C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19']
alt = pd.read_csv(inpath+"Irradiance_rsun_lin2_lag-0.2_elev_df.csv", index_col = 0, parse_dates=True)
   
irr = LCB_Irr()
inpath_obs = '/home/thomas/PhD/obs-lcb/LCBData/obs/Irradiance/data/'
files_obs = glob.glob(inpath_obs+"*")
irr.read_obs(files_obs)
  
irr.read_sim(inpath+"Irradiance_rsun_lin2_lag-0.2_glob_df.csv")
# this should be implemented in the Grass code directly
irr.data_sim.columns = stanames
alt.columns = stanames
  
# rng = pd.date_range(irr.data_sim.index[0], irr.data_sim.index[-1],freq='1H')
# df_0 = pd.DataFrame(0,index =rng, columns=stanames)
# 
# 
# irr.data_sim = irr.data_sim.reindex(rng)
# alt = alt.reindex(rng)
# irr.data_sim = pd.merge(irr.data_sim, df_0, how='right')
# alt = pd.merge(alt, df_0, how='right')
  
# pas sur que faire le ratio avec le simuler clear sky soit 
# la meilleur forme surtout in the morning and evening transitioning
ratio = irr.ratio()
  
  
# ratio[ratio.between_time('16:00','08:00').index] = 100
   
glob_irr = irr.data_sim.multiply(ratio, axis=0)
  
glob_irr = glob_irr.reindex(pd.date_range(glob_irr.index[0], glob_irr.index[-1],freq='1H'))
glob_irr = glob_irr.replace([np.inf, -np.inf], np.nan)
glob_irr  = glob_irr.fillna(0)
alt = alt.reindex(pd.date_range(alt.index[0], alt.index[-1],freq='1H'), method='pad')
alt  = alt.fillna(0)
  
   
   
#===============================================================================
# Gfs variables 
#===============================================================================
inpath = "/home/thomas/"
     
df_gfs = pd.read_csv(inpath+'gfs_serie.csv', index_col =0, parse_dates=True )

# df_gfs = df_gfs['TMP_950mb'] - 273.15

cols_var = df_gfs.iloc[:,6:].columns

# Data = pd.DataFrame(df_gfs.index)

for c_v in cols_var:
 
    newdf = pd.DataFrame(df_gfs[c_v] )
    for staname in stanames: 
        newdf[staname] = df_gfs[c_v]
 
    del newdf[c_v]
    if c_v == cols_var[0]:
        data_gfs = pd.DataFrame(newdf.stack(), columns = [c_v])
        print data_gfs
    else:
        data_gfs[c_v] = newdf.stack()
        
  
#===============================================================================
# Merge dataframes
#===============================================================================
   
alt_stacked = alt.stack().reset_index()
alt_stacked.columns = ['Date', 'stations', 'Alt']
glob_irr_stacked = glob_irr.stack().reset_index()
glob_irr_stacked.columns = ['Date', 'stations', 'irr']
df_station_stacked = df_stations.stack().reset_index()
df_station_stacked.columns = ['Date', 'stations', 'Ta C']
data_gfs = data_gfs.reset_index()
print cols_var
print data_gfs.columns
print data_gfs
data_gfs.columns = ['Date', 'stations']+list(cols_var)
 
   
result = pd.merge(alt_stacked,glob_irr_stacked,on=['Date','stations'], how='inner')
result = pd.merge(result,df_station_stacked,on=['Date','stations'], how='inner')
result = pd.merge(result,data_gfs,on=['Date','stations'], how='inner')
print result
result.to_csv('/home/thomas/df_PCA.csv')
  
def ala(df):
    for i in df['Date']:
        print i
