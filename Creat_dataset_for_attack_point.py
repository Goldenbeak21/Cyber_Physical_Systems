import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler

data_df = pd.read_csv('test_update_feature_df_200_800_2.csv')
data_df['attack']=0
#print(data_df.shape)
X_train = data_df[(0<data_df.delay)].drop(['period', 'powerSetPoint', 'sigma', 'delay'], axis=1)
normal_seq_p=X_train.ix[:,0:151]
normal_seq_pe=X_train.ix[:,301:452]
normal_seq_t=X_train.ix[:,602:753]
frames_p_pe_t = [normal_seq_p, normal_seq_pe,normal_seq_t]
normal_seq = pd.concat(frames_p_pe_t, axis=1,ignore_index=True)
normal_seq['attack']=0
#print(normal_seq.ix[0,:])
#print(normal_seq_pe.head(1))
#print(normal_seq_t.head(1))
for i in range(1,5):
   attack_seq_p = X_train.ix[:, i:(151+i)]
   attack_seq_pe = X_train.ix[:, (301+i):(452 + i)]
   attack_seq_t = X_train.ix[:, (602+i):(753 + i)]
   frame_p_pe_t = [attack_seq_p, attack_seq_pe, attack_seq_t]
   attack_seq = pd.concat(frame_p_pe_t, axis=1,ignore_index=True)
   attack_seq['attack'] = 1

   frames = [attack_seq, normal_seq]

   result = pd.concat(frames)
   normal_seq=result.copy()

normal_seq.to_csv("overall_attack_testdataset.csv",index=None)
