#===============================================================================
# Library
#===============================================================================
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

import statsmodels.api as sm

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

#------------------------------------------------------------------------------ 
# Loading dataframe
df = pd.read_csv("/home/thomas/df_PCA.csv", index_col=1, parse_dates=True)

df = df.between_time('09:00','10:00')
print df
df['newindex'] = np.arange(len(df))
df.index = df['newindex'] 
del df['stations']
del df['Unnamed: 0']
del df['newindex']

df = df.dropna(axis=0, how='any')

print df
#------------------------------------------------------------------------------ 
# split data table into data X and class labels y

esti_name = "Ta C"
predi_name = ['CLWMR_500mb', 'HGT_950mb',"RH_950mb","TMP_950mb","VVEL_950mb",'irr', 'Alt'] # Day 
# predi_name = ['CLWMR_500mb', 'HGT_950mb',"RH_950mb","TMP_950mb","VVEL_950mb", 'Alt'] # Night


X = df[predi_name].values
y = df[esti_name].values


#------------------------------------------------------------------------------ 
#Standardizing
# from sklearn.preprocessing import StandardScaler
# X_std = StandardScaler().fit_transform(X)

# Another way to standardize using python
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

df_params_std = pd.concat([pd.Series(X.mean(axis=0)), pd.Series(X.std(axis=0))], axis=1)
df_params_std = df_params_std.T
df_params_std.columns = predi_name
df_params_std.to_csv('/home/thomas/params_std.csv')

#===============================================================================
# 1- Eigendecomposition - Computing Eigenvectors and Eigenvalues
#===============================================================================
#------------------------------------------------------------------------------ 
# #Covariance Matrix
# import numpy as np
# mean_vec = np.mean(X_std, axis=0)
# cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
# print('Covariance matrix \n%s' %cov_mat)


#------------------------------------------------------------------------------ 
#eidgen decomposition after standardizing 
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
 
eig_vecs.dot(eig_vals.T)
 
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# 
# # OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR 
# 
# #------------------------------------------------------------------------------ 
# # Correlation Matrix 
 
# cor_mat1 = np.corrcoef(X_std.T)
#  
# # eidgen decomposition after standardizing 
# eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
  
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)
# 
# cor_mat2 = np.corrcoef(X.T)
# 
# eig_vals, eig_vecs = np.linalg.eig(cor_mat2)
# 
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)
# 
# #We can clearly see that all three approaches yield the same eigenvectors and eigenvalue pairs:
# #    Eigendecomposition of the covariance matrix after standardizing the data.
# #    Eigendecomposition of the correlation matrix.
# #    Eigendecomposition of the correlation matrix after standardizing the data.
# # OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR  OR 
# 
# 
# #------------------------------------------------------------------------------ 
# # Singular Vector Decomposition
# 
# u,s,v = np.linalg.svd(X_std.T)


#===============================================================================
# 2- Selecting Principal Components
#===============================================================================
#------------------------------------------------------------------------------ 
#Sorting Eigenpairs

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

df_eig_val = pd.DataFrame(eig_vals, columns=['eigen_value'])
df_eig_vec = pd.DataFrame(eig_vecs, columns= predi_name )

df_eig_pairs = pd.concat([df_eig_val, df_eig_vec], axis=1 )
df_eig_pairs = df_eig_pairs.sort(columns='eigen_value', ascending=False)
df_eig_pairs = df_eig_pairs.reindex(index = range(len(df_eig_pairs.index)))
df_eig_pairs.to_csv('/home/thomas/eig_pairs.csv')




# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


#===============================================================================
# Plot explained variances
#===============================================================================
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

X = sm.add_constant(X)
plt.figure(figsize=(6, 4))

plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(len(var_exp)), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



#===============================================================================
# Plot type Glauber
#===============================================================================
N = len(var_exp)
ind = np.arange(N)
width = 0.1

fig, ax = plt.subplots()

rects1 = ax.bar(ind, eig_pairs[0][1], width, color='0.05')
rects2 = ax.bar(ind + width, eig_pairs[1][1], width, color='0.2')
rects3 = ax.bar(ind + width*2, eig_pairs[2][1], width, color='0.4')
rects4 = ax.bar(ind + width*3, eig_pairs[3][1], width, color='0.6')
rects5 = ax.bar(ind + width*4, eig_pairs[4][1], width, color='0.8')
rects6 = ax.bar(ind + width*5, eig_pairs[5][1], width, color='0.9')
rects7 = ax.bar(ind + width*6, eig_pairs[6][1], width, color='0.95')


# add some text for labels, title and axes ticks
ax.set_ylabel('EOF components')
ax.set_title('PCA analysis')
ax.set_xticks(ind + width)
ax.set_xticklabels(predi_name)

ax.legend((rects1[0], rects2[0], rects3[0],rects4[0], rects5[0], rects6[0] ,rects7[0] ), ('e1', 'e2','e3', 'e4', 'e5', 'e6', 'e7'))


# def autolabel(rects):
#     # attach some text labels
#     for rect in rects:
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 '%d' % int(height),
#                 ha='center', va='bottom')
# 
# autolabel(rects1)
# autolabel(rects2)

plt.show()





#------------------------------------------------------------------------------ 
# Projection Matrix
# Choosing k eigenvectors with the largest eigenvalues
# Day
# matrix_w = np.hstack((eig_pairs[0][1].reshape(len(var_exp),1),
#                       eig_pairs[1][1].reshape(len(var_exp),1),
#                     eig_pairs[2][1].reshape(len(var_exp),1),
#                     eig_pairs[3][1].reshape(len(var_exp),1),
#                     eig_pairs[4][1].reshape(len(var_exp),1),
#                     eig_pairs[5][1].reshape(len(var_exp),1),
#                     eig_pairs[6][1].reshape(len(var_exp),1),
#                       ))
 #Night
matrix_w = np.hstack((eig_pairs[0][1].reshape(len(var_exp),1),
                      eig_pairs[1][1].reshape(len(var_exp),1),
                    eig_pairs[2][1].reshape(len(var_exp),1),
                    eig_pairs[3][1].reshape(len(var_exp),1),
                    eig_pairs[4][1].reshape(len(var_exp),1),
                    eig_pairs[5][1].reshape(len(var_exp),1),
                      ))
# 
# matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),
# 
#                       ))

print('Matrix W:\n', matrix_w)

#===============================================================================
# 6. Transforming the samples onto the new subspace
#===============================================================================
X_reduced = X_std.dot(matrix_w)

# assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."
#  
# plt.plot(transformed[0,0:20], transformed[1,0:20],\
#      'o', markersize=7, color='blue', alpha=0.5, label='class1')
# plt.plot(transformed[0,20:40], transformed[1,20:40],
#      '^', markersize=7, color='red', alpha=0.5, label='class2')
# plt.xlim([-4,4])
# plt.ylim([-4,4])
# plt.xlabel('x_values')
# plt.ylabel('y_values')
# plt.legend()
# plt.title('Transformed samples with class labels')
#  
# plt.draw()
# plt.show()


#===============================================================================
# Multiple regression of the component
#===============================================================================
X_reduced = sm.add_constant(X_reduced)
est = sm.OLS(y, X_reduced).fit()
ypred_CPA = est.predict(X_reduced)

# est = sm.OLS(y, X).fit()
# ypred = est.predict(X)
# 
# plt.plot(ypred,'r')
plt.plot(y,'k')
plt.plot(ypred_CPA,'b')
# plt.show()

params = pd.Series(est.params, name='Parameters')
params.to_csv('/home/thomas/params.csv')

#===============================================================================
# Using Scikit-learn
#===============================================================================

pca = PCA()
X_reduced = pca.fit_transform(scale(X))
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

mse = []

n = len(X_reduced)
kf_10 = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=2)

for i in np.arange(1, 3):
    pls = PLSRegression(n_components=i, scale=False)
    pls.fit(scale(X_reduced),y)
    Y_pred = pls.predict(X_reduced)
    score = cross_validation.cross_val_score(pls, X_reduced, y, cv=kf_10, scoring='mean_squared_error').mean()
    mse.append(-score)

plt.plot(Y_pred,'g')
plt.plot(y,'k')
plt.show()


plt.plot(np.arange(1, 3), np.array(mse), '-v')
plt.xlabel('Number of principal components in PLS regression')
plt.ylabel('MSE')
plt.xlim((-0.2, 5.2))








