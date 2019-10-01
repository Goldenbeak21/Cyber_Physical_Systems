import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
from keras.layers.merge import concatenate
from keras.layers import Input, multiply, Lambda
from keras.layers.core import *
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler

test_df = pd.read_csv("overall_attack_testdataset.csv")
train_df = pd.read_csv("overall_attack_traindataset.csv")
from sklearn.utils import shuffle
test_df = shuffle(test_df)
train_df = shuffle(train_df)


class DataLoader():
    def __init__(self, X, y, batch_size, seq_length, input_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        # -1 MEANS LAST INDEX
        X_shape = list(X.shape)
        X_shape[-1] = int(X_shape[-1] / input_size)
        step = int(X_shape[-1] / seq_length)
        lengh = step * seq_length
        # like image we have 3 dimention of T,P,PE
        X = X.reshape((X_shape[0], input_size, -1))[:, :,
            :lengh]  # here is hust want to ingnore extra data than for example 200
        #print(X[:,0,:].shape)

        self.X = X[:,1,:].reshape((X_shape[0], seq_length, -1))  ##number of records * seq_lengh*3*15
        #print(self.X.shape)
        #exit()


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
# for single branch like delay
X_test = test_df.drop(['attack'], axis=1)
y_test = test_df[['attack']]

X_train = train_df.drop(['attack'], axis=1)
y_train_1 = train_df['attack']
y_train = train_df[['attack']]
print(y_train_1.to_numpy())
#print(np.unique(y_train))
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_1),y_train_1.to_numpy())
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)

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
#exit()
print(X_train.shape)
print(X_test.shape)


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.layers import Bidirectional
from keras import regularizers

###############attation

SINGLE_ATTENTION_VECTOR = False
TIME_STEPS = params["seq_length"]
import time
import keras


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_callback = TimeHistory()


def attention_3d_block(inputs, layer_name):
    # inputs.shape = (batch_size, time_steps, input_dim)

    name = layer_name
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    # print(a.shape)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name=name)(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


def rnn_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""
    maxlen = params["seq_length"]

    inp = Input(shape=(layers[0], layers[1]))

    attention = attention_3d_block(inp, "input_attention")
    layer_1 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True, activation='relu'))(attention)
    dropout_1 = Dropout(params['dropout_keep_prob'])(layer_1)
    # attention_input=concatenate([attention, layer_1])
    # print(attention.shape)
    layer_2 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True, activation='relu'))(
        dropout_1)
    # print(Bidirectional.shape)
    #dropout_2 = Dropout(params['dropout_keep_prob'])(layer_2)
    attention_2 = attention_3d_block(layer_2,"attention")

    '''layer_3 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=False, activation='relu'))(
    dropout_2)

dropout_3 = Dropout(params['dropout_keep_prob'])(layer_3)

layer_4 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True, activation='relu'))(dropout_3)

dropout_4 = Dropout(params['dropout_keep_prob'])(layer_4)
layer_5 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True, activation='relu'))(
    dropout_4)

dropout_5 = Dropout(params['dropout_keep_prob'])(layer_5)
layer_6 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=False))(dropout_5)

dropout_6 = Dropout(params['dropout_keep_prob'])(layer_6)
# attention_3 = attention_3d_block(layer_3,'attention')
# print(attention_3.shape)'''

    #dense = Dense(units=layers[2], activation='relu')(dropout_2)
    # print(dense.shape)
    #dropout_4 = Dropout(params['dropout_keep_prob'])(dense)
    attention_mul = Flatten()(attention_2)
    dense_2 = Dense(units=layers[3], activation='sigmoid')(attention_mul)
    # print(dense_2.shape)
    # optimizer = Adam(clipvalue=0.5)
    adam = Adam(clipvalue=0.5, lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model = Model(inputs=inp, outputs=dense_2)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model


lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]

saved_model = "SECOND_FEATURE"
model = rnn_lstm(lstm_layer, params)

df_his = None

# Train RNN (LSTM) model with train set
history = model.fit(X_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_split=params['validation_split'],
                    class_weight=class_weight_dict,
                    callbacks=[ModelCheckpoint(filepath="models/" + saved_model, monitor='loss', verbose=1,
                                               save_best_only=True), \
                               ModelCheckpoint(filepath="models/" + saved_model + "_val", monitor='val_loss', verbose=1,
                                               mode='min', save_best_only=True), time_callback]
                    )

# pd.DataFrame(history.history).to_csv("history/%s.csv" % (saved_model))


from keras.models import load_model

model = load_model("models/%s" % (saved_model))  # , custom_objects={'Attention': Attention(params["seq_length"])}
print(model.count_params())
print(y_test)

predict_1 = model.predict(X_test)
rounded= [round(x[0]) for x in predict_1]
predict= np.array(rounded,dtype='int64')
#print(predict)
#predict = (np.argmax(predict_1,axis=1))
#print(predict.shape)
#print(y_test)
#print(np.array(y_test).squeeze().shape)
scores = model.evaluate(X_test, y_test, verbose=0)
#predict=round(scores)
#print(predict_1)
print("Accuracy: %.2f%%" % (scores[1]*100))
#y_true = scaler_y.inverse_transform(y_test)
df = pd.DataFrame({"predict": predict, "y_true": np.array(y_test).squeeze()})
#df_1 = pd.DataFrame({"predict": np.array(y_test)})
#df_1.to_csv('result-%s.csv' % (saved_model), index=True, header=True)
df.to_csv("results/%s.csv" % (saved_model) , index=False, header=True)
#exit()



from sklearn.metrics import average_precision_score, recall_score, precision_score,f1_score, roc_curve, confusion_matrix
#print(predict)
#print(np.array(y_test))

average_precision=average_precision_score(np.array(y_test), predict_1)
print("average_precision:",average_precision)

recall=recall_score(np.array(y_test), np.array(predict),average='binary')
print("recall:", recall)

precision=precision_score(np.array(y_test), np.array(predict),average='binary')
print("precision:", precision)

f1=f1_score(np.array(y_test), np.array(predict),average='binary')
print("f1: ",f1)

tn, fp, fn, tp = confusion_matrix(np.array(y_test), np.array(predict)).ravel()# In[13]:
print(fp)
print("fpr: ",(fp/(fp+tn)))


#df = pd.DataFrame({"predict": predict, "y_true": y_test})
#df.to_csv('startpomit_result-%s.csv' % (saved_model), index=True, header=True)

'''from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
#from collections import Counter
#Counter(y_test) # y_true must be your labels
print(np.array(y_test,dtype="int64").squeeze())
print(np.array(predict))
precision, recall, _ = precision_recall_curve(np.array(y_test,dtype="int64").squeeze(), np.array(predict))

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Diabetic status Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

import keras.backend as K
import numpy as np
from keras.utils.vis_utils import plot_model

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)'''

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
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
for i in range(1000):
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

import seaborn as sns
from pandas import DataFrame
#pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',title='Attention Mechanism as '
'''attention_vector_final = np.array([0.09881116, 0.04011875, 0.09874146, 0.03741224, 0.18375048, 0.05859727,
                                   0.02719666, 0.0294565, 0.0356444, 0.04843325, 0.03953683, 0.0257762,
                                   0.0267469, 0.03855285, 0.04842967, 0.03316928, 0.04160227, 0.02852467,
                                   0.02667178, 0.03282746])'''# pe
# P [0.01150572 0.01622499 0.01471417 0.01967236 0.0486087  0.0231822 0.02308206 0.26022872 0.02202909 0.04149413 0.01079566 0.0297078  0.03181939 0.03871002 0.0838391  0.05695531 0.04732064 0.0734517  0.04593809 0.10071981]
# 'a function of input'
                                                                          # ' dimensions.')
df=DataFrame(attention_vector_final, columns=['attention (%)'])
sns.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)
df.style.background_gradient(cmap='summer')
import matplotlib.pyplot as plt
print(attention_vector_final)

pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                     title='Attention Mechanism as '
                                                                           'a function of input'
                                                                           ' dimensions.')
plt.show()
# [0.04480592 0.01578229 0.01535583 0.04554126 0.0567533  0.03128133
#  0.0671864  0.05798084 0.04749841 0.04616091 0.04579237 0.05080504
#  0.03808663 0.05026736 0.07026131 0.05901461 0.05111502 0.06866731
#  0.02790152 0.10974246] three featurs
