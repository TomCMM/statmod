
import pandas as pd
import matplotlib.pyplot as plt 

if __name__ =='__main__':
#     path='/home/thomas/PhD/obs-lcb/staClim/svg/SVG_2013_2016_Thomas_30m.csv'
#     df = pd.read_csv(path, index_col=0)
#     print df
#     print df.columns
#     df['T_AR_HMP_FINAL'].plot()
#     plt.show()

    path='/home/thomas/PhD/obs-lcb/staClim/peg/Th_peg_tar30m.csv'
    df = pd.read_csv(path, index_col=1, header=None)
    df.columns = ['i','Ta C']
    del df['i']
    print df
    print df.columns
#     df['T_AR_HMP_FINAL'].plot()
#     plt.show()