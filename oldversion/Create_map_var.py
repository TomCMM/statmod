#===============================================================================
# DESCRIPTION
# import the eigen value and eigenvector and the parameters of the multiple regression
# and reconstruct a map of the temperature
#===============================================================================
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#===============================================================================
# Get map irradiance and elev
#===============================================================================
dayofyear =100
hour = 03

# dayofyear = 336
# hour = 9

date = pd.to_datetime(datetime.datetime(2015, 1, 1) + datetime.timedelta(dayofyear - 1) + datetime.timedelta(hours=hour))

# date = pd.to_datetime(datetime.datetime(2013, 1, 1) + datetime.timedelta(dayofyear - 1) + datetime.timedelta(hours=hour))
irr  = np.loadtxt('/home/thomas/100_15rasterglob.txt',delimiter=',')
elev  = np.loadtxt('/home/thomas/100_15rasterelev.txt',delimiter=',')

# irr  = np.loadtxt('/home/thomas/336_9rasterglob.txt',delimiter=',')
# elev  = np.loadtxt('/home/thomas/336_9rasterelev.txt',delimiter=',')

#===============================================================================
# Get gfs data and create an array of them 
#===============================================================================

# df_gfs = pd.read_csv('/home/thomas/gfs_serie.csv', index_col=0, parse_dates=True)
df_gfs = pd.read_csv('/home/thomas/gfs_data_GFS_131202.csv', index_col=0, parse_dates=True)
data_gfs = df_gfs.ix[date]

vars = {}
# vars['irr'] = irr # Only day 

plt.contourf(irr[::-1,:])
plt.colorbar()
plt.show()

vars['Alt'] =  elev

CLWMR_500mb =np.empty(irr.shape)
CLWMR_500mb.fill(data_gfs['CLWMR_500mb'])
vars['CLWMR_500mb'] = CLWMR_500mb

HGT_950mb =np.empty(irr.shape)
HGT_950mb.fill(data_gfs['HGT_950mb'])
vars['HGT_950mb'] = HGT_950mb


RH_950mb =np.empty(irr.shape)
RH_950mb.fill(data_gfs['RH_950mb'])
vars['RH_950mb'] = RH_950mb

TMP_950mb =np.empty(irr.shape)
TMP_950mb.fill(data_gfs['TMP_950mb'])
vars['TMP_950mb'] = TMP_950mb

VVEL_950mb =np.empty(irr.shape)
VVEL_950mb.fill(data_gfs['VVEL_950mb'])
vars['VVEL_950mb'] = VVEL_950mb




#===============================================================================
# Retrieve parameter and eig_pairs to reconstruct the variable map
#===============================================================================


df_params = pd.read_csv('/home/thomas/params_9h.csv',index_col=0, header=None)
df_eigpairs = pd.read_csv('/home/thomas/eig_pairs_9h.csv',index_col=0)
df_params_std = pd.read_csv('/home/thomas/params_std_9h.csv',index_col=0)

Y = np.zeros(irr.shape)
for CPs in range(len(df_eigpairs)-1):
    print "Principal component ->  "+ str(CPs)
    CP = np.zeros(irr.shape)
    for var in  vars.keys():
        print var
        X_std = vars[var]
        X_std = (vars[var] - df_params_std.loc[0][var] ) / df_params_std.loc[1][var]
        CP = CP +  X_std * df_eigpairs[var][CPs]
    Y = Y+ df_params.loc[CPs+1].values * CP

Y0 = np.zeros(irr.shape)
Y0 .fill(df_params.loc[0].values.item(0) )

Y = Y+Y0

print Y
print Y.shape


CS = plt.contour(elev[::-1,::],colors='k')
plt.contourf(Y[::-1,:], levels = np.linspace(Y.min(), Y.max(), 100))
plt.clabel(CS, inline=1, fontsize=10)
plt.colorbar()
plt.show()


np.savetxt("/home/thomas/variable.txt", Y, delimiter=',')




# Another way to standardize using python
# X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# df_params_std = pd.concat([pd.Series(X.mean(axis=0)), pd.Series(X.std(axis=0))], axis=1)
# df_params_std = df_params_std.T
# df_params_std.columns = predi_name
# df_params_std.to_csv('/home/thomas/params_std.csv')









