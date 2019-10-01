import numpy as np
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import  Input,multiply, Lambda
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv('trainsetfinal.csv')
test_df = pd.read_csv('testsetfinal.csv')
'''from sklearn.utils import shuffle
test_df = shuffle(test_df)
train_df = shuffle(train_df)'''

class DataLoader():
    def __init__(self, X, y, batch_size, seq_length, input_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        #-1 MEANS LAST INDEX
        X_shape = list(X.shape)
        X_shape[-1] = int(X_shape[-1] / input_size)
        step = int(X_shape[-1] / seq_length)
        lengh = step * seq_length
        # like image we have 3 dimention of T,P,PE
        X = X.reshape((X_shape[0], input_size, -1))[:, :, :lengh]#here is hust want to ingnore extra data than for example 200
        #print(X.shape)

        self.X = X.reshape((X_shape[0], seq_length, -1))##number of records * seq_lengh*3*15
        #print(self.X.shape)

        self.y = y

    def dataset(self):
        return (self.X, self.y)


params = {
    "epochs": 300,
    "batch_size": 64,
    "seq_length": 20,
    "dropout_keep_prob": 0.1,
    "hidden_unit": 500,
    "validation_split": 0.1,
    "input_size": 3
}
#for single branch like delay
X_test = test_df.drop(['period', 'powerSetPoint', 'sigma', 'delay'], axis=1)
y_test = test_df[['delay']]

X_train = train_df.drop(['period', 'powerSetPoint', 'sigma', 'delay'], axis=1)
y_train = train_df[['delay']]
#for start point
'''X_test = test_df.drop(['period','end','powerSetPoint','sigma','delay','start','split'],axis=1)
y_test = test_df[['delay']]

X_train = train_df.drop(['period','end','powerSetPoint','sigma','delay','start','split'],axis=1)
y_train = train_df[['delay']]'''
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(np.concatenate([X_test, X_train], axis=0))
scaler_y.fit(np.concatenate([y_test, y_train], axis=0))

data_loader = DataLoader(scaler_X.transform(X_test), scaler_y.transform(y_test), params["batch_size"],
                         params["seq_length"], params["input_size"])
X_test, y_test = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_train), scaler_y.transform(y_train), params["batch_size"],
                         params["seq_length"], params["input_size"])
X_train, y_train = data_loader.dataset()
#print(X_train.shape)

import tensorflow as tf
from tensorflow.keras.backend.tensorflow_backend import set_session




config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import regularizers

###############attation

SINGLE_ATTENTION_VECTOR = False
TIME_STEPS=params["seq_length"]
import time
import tensorflow.keras
class TimeHistory(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
time_callback = TimeHistory()

def attention_3d_block(inputs,layer_name):
    # inputs.shape = (batch_size, time_steps, input_dim)
    #layer_name = 'attention_layer1'
    #inputs = inp
    name = layer_name
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    #print(a.shape)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1),name=name)(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
"""Build RNN (LSTM) model on top of tensorflow.keras and Tensorflow"""
layers = lstm_layer
maxlen = params["seq_length"]

inp = Input(shape=(layers[0], layers[1]))

input_attention = attention_3d_block(inp, "input_attention")

layer_1=GRU(units=layers[2],batch_size= params["batch_size"], return_sequences=True, activation='relu')(input_attention)
dropout_1=Dropout(params['dropout_keep_prob'])(layer_1)
#attention_input=concatenate([attention, layer_1])
#print(attention.shape)
layer_2=GRU(units=layers[2],batch_size= params["batch_size"], return_sequences=True,activation='relu')(dropout_1)
#print(Bidirectional.shape)
dropout_2 =Dropout(params['dropout_keep_prob'])(layer_2)


layer_3 = GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True,activation='relu')(dropout_2)

dropout_3 = Dropout(params['dropout_keep_prob'])(layer_3)

layer_4 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True,activation='relu'))(dropout_3)

dropout_4 = Dropout(params['dropout_keep_prob'])(layer_4)
layer_5 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True,activation='relu'))(dropout_4)

dropout_5 = Dropout(params['dropout_keep_prob'])(layer_5)
#layer_6 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=False,activation='relu'))(dropout_5)

#dropout_6 = Dropout(params['dropout_keep_prob'])(layer_6)
output_attention = attention_3d_block(dropout_5,"output_attention")
#print(attention_3.shape)'''

#dense=Dense(units=layers[2], activation='relu')(dropout_6)
#print(dense.shape)
#dropout_4 = Dropout(params['dropout_keep_prob'])(dense)
attention_mul = Flatten()(output_attention)
dense_2=Dense(units=layers[3],activation='relu')(attention_mul)
#print(dense_2.shape)
# optimizer = Adam(clipvalue=0.5)
adam = Adam(clipvalue=0.5, lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
model = Model(inputs=inp, outputs=dense_2)
inter = Model(inputs = inp, outputs = model.get_layer("input_attention").output)
model.compile(loss="mean_squared_error", optimizer=adam)
history = model.fit(X_train, y_train, epochs=50)
results = inter.predict(X_test)

plt.scatter(t, results[10,15,:], color='red')
plt.scatter(t, X_test[10,15,:], color = 'blue')
plt.show()

t = np.arange(0,45,1)
t1 = np.arange(0,301,1)
sum = np.zeros(45,)
for i in range(45):
    sum[i] = (results[0,5,i]/X_test[0,5,i])
import matplotlib.pyplot as plt
test_df = np.array(test_df)
for i in range(50):
    print(i)
    plt.bar(t[0:15],results[4,16,0:15], color = 'blue')
    plt.bar(t[15:30],results[4,16,15:30], color = 'red')
    plt.bar(t[30:45],results[4,16,30:45], color = 'green')
    plt.show()
    
cusm = np.zeros((149,3))
for i in range(149):
    for j in range(3):
        cusm[i,j] = np.sum(results[i, 16, j*15:(j+1)*15])

t2 = np.arange(0,3,1)
m = [4,20,26,27,56,64,75,115,129,136]
m= np.array(m)

for i in range(149):
  '''  print(i)
    plt.bar(t2, cusm[i,:])
    plt.show()
    '''   
    
ts0 = np.sum(cusm[:,0])
ts1 = np.sum(cusm[:,1])
ts2 = np.sum(cusm[:,2])


t1 = np.arange(0,30,1)
for i in range(m.shape[0]):
    print(m[i])
    plt.plot(t1, test_df[m[i],240+2:242+30]-mu[0])
    plt.show()
    plt.plot(t1, test_df[m[i],303+240:303+240+30]-mu[1])
    plt.show()
    plt.plot(t1, test_df[m[i],604+240:604+240+30]-mu[2])
    plt.show()


form = pd.read_csv('modelica_trainset.csv').values
form = np.array(form)
mu = np.zeros((3,))
for i in range(3):
    mu[i] = np.mean(form[:,i])

'''
                    validation_split=params['validation_split'],
                    callbacks=[ModelCheckpoint(filepath="models/" + saved_model, monitor='loss', verbose=1,
                                               save_best_only=True), \
                               ModelCheckpoint(filepath="models/" + saved_model + "_val", monitor='val_loss', verbose=1,
                                               mode='min', save_best_only=True), time_callback]
                    )
times = time_callback.times

pd.DataFrame({"time": times}).to_csv("time/%s.csv" % (saved_model),index=True)'''

#pd.DataFrame(history.history).to_csv("history/%s.csv" % (saved_model))'''

# In[6]:
'''if df_his is None:
    df = pd.DataFrame(history.history)
    df.to_csv("history_%s.csv" % (saved_model), header=True)
else:
    df = pd.concat([df_his, pd.DataFrame(history.history)]).reset_index()
    df.to_csv("history_%s.csv" % (saved_model), header=True)'''

from tensorflow.keras.models import load_model
import time

model = load_model("models/%s" % (saved_model))#, custom_objects={'Attention': Attention(params["seq_length"])}
print(model.count_params())
model.summary()
start_time = time.time()
predict = scaler_y.inverse_transform(model.predict(X_test))
print("--- %s seconds ---" % (time.time() - start_time))
#exit()
y_true = scaler_y.inverse_transform(y_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


def NRMSD(y_true, y_pred):
    rmsd = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    y_min = min(y_true)
    y_max = max(y_true)

    return rmsd / (y_max - y_min)


def MAPE(y_true, y_pred):
    y_true_select = (y_true != 0)

    y_true = y_true[y_true_select]
    y_pred = y_pred[y_true_select]

    errors = y_true - y_pred
    return sum(abs(errors / y_true)) * 100.0 / len(y_true)


# In[13]:


nrmsd = NRMSD(y_true, predict)
mape = MAPE(y_true, predict)
mae = mean_absolute_error(y_true, predict)
rmse = np.sqrt(mean_squared_error(y_true, predict))
print("NRMSD", nrmsd)
print("MAPE", mape)
print("neg_mean_absolute_error", mae)
print("Root mean squared error", rmse)

pd.DataFrame({"predict": predict.flatten(), "y_true": y_true.flatten()}).to_csv("results/%s.csv" % (saved_model), header=True)

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
from tensorflow.keras.utils.vis_utils import plot_model

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


'''def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/tensorflow.keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations
attention_vectors = []
for i in range(2973):
    #testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    x_test =  np.expand_dims(X_test[i,:,:], axis=0)
    #print(X_test[i,:,:].shape)
    attention_vector = np.mean(get_activations(model,x_test,
                                               print_shape_only=True,
                                               layer_name='attention')[0], axis=2).squeeze()
    #print('attention =', attention_vector)
    #assert (np.sum(attention_vector) - 1.0) < 1e-5
    attention_vectors.append(attention_vector)

attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
print(attention_vector_final.shape)
# plot part.
import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                     title='Attention Mechanism as '
                                                                           'a function of input'
                                                                           ' dimensions.')
plt.show()'''