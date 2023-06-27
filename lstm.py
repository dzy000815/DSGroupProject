import numpy as np
import pandas as pd
import tensorflow
import keras
from keras import optimizers

Dense = tensorflow.keras.layers.Dense
LSTM = tensorflow.keras.layers.LSTM
Sequential = tensorflow.keras.Sequential
Dropout = tensorflow.keras.layers.Dropout
SGD = tensorflow.keras.optimizers.SGD
Adam = tensorflow.keras.optimizers.Adam
Earlystopping = tensorflow.keras.callbacks.EarlyStopping
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from random import randint

# %%-----------------------------------Read the files from datasets--------------------------------
# Read the training data from the file
x = pd.read_csv('train1.csv')
x = x.values

# Append the random number of 0 or 1 to override the pandemic data
xNoP = pd.read_csv('train1_noP.csv')
for i in range(3):
    temp = []
    for j in range(len(xNoP)):
        temp.append(randint(0, 1))
    xNoP.insert(xNoP.shape[1], str(i), temp)
xNoP = xNoP.values


def data_process(x, y, window_size, predict_size):
    data_in = []
    data_out = []
    for i in range(x.shape[0] - window_size - predict_size):
        data_in.append(x[i:i + window_size])
        data_out.append(y[i + window_size:i + window_size + predict_size])
    data_in = np.array(data_in)
    data_out = np.array(data_out)
    data_process = {'datain': data_in, 'dataout': data_out}
    return data_process


# Define the LSTM model. It is a two-level model with 128 neural units in total. And two full connection layers to generate the output. And the plt function will plot the training loss. It also has a training mode or predicting mode. The models are saved in the file


def lstm_model(train_X, test_X, train_y, test_y, timestep, batch_size, epoch, predict_size, train, diff, pandemic=True):
    if (train):
        model = Sequential()
        early_stopping = Earlystopping(monitor='val_loss', patience=3)
        model.add(LSTM(units=64, input_shape=(timestep, train_X.shape[2]), return_sequences=True, activation='relu'))
        model.add(LSTM(units=64, return_sequences=False, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(predict_size, activation='sigmoid'))
        sgd = optimizers.Adam(lr=0.0002)
        model.add(Dropout(0.1))
        model.compile(loss='mean_squared_error', optimizer=sgd)
        model.summary()

        history = model.fit(train_X, train_y, batch_size=batch_size, epochs=epoch, validation_data=(test_X, test_y))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        if not pandemic:
            model.save('newMicroPCA_diff_nopandemic.h5')
        else:
            if diff:
                model.save('newMicroPCA_diff.h5')
            else:
                model.save('newMicroPCA.h5')

        preds = model.predict(np.vstack((train_X, test_X)))
        preds_test = model.predict(test_X)
    else:
        if not pandemic:
            model = keras.models.load_model('newMicroPCA_diff_nopandemic.h5')
        else:
            if diff:
                model = keras.models.load_model('newMicroPCA_diff.h5')
            else:
                model = keras.models.load_model('newMicroPCA.h5')
        preds = model.predict(np.vstack((train_X, test_X)))
        preds_test = model.predict(test_X)

    return preds, preds_test


def split(data_prepro):
    X_train1 = data_prepro['datain'][:600]
    y_train1 = data_prepro['dataout'][:600]
    X_test1 = data_prepro['datain'][600:700]
    y_test1 = data_prepro['dataout'][600:700]
    X_train2 = data_prepro['datain'][700:-100]
    y_train2 = data_prepro['dataout'][700:-100]
    X_test2 = data_prepro['datain'][-100:]
    y_test2 = data_prepro['dataout'][-100:]
    X_train = np.vstack((X_train1, X_train2))
    y_train = np.vstack((y_train1, y_train2))
    X_test = np.vstack((X_test1, X_test2))
    y_test = np.vstack((y_test1, y_test2))
    return X_train, X_test, y_train, y_test


def truth(preds_test_dediff, ytestTrueAdd):
    PredsDiffPrice = []
    for i in range(len(preds_test_dediff)):
        temp = ytestTrueAdd[i]
        tempList = []
        tempList.append(temp)
        for j in range(len(preds_test_dediff[i])):
            tempList.append(tempList[j] + preds_test_dediff[i][j])
        PredsDiffPrice.append(tempList)
    PredsDiffPrice = np.array(PredsDiffPrice)
    PredsDiffPrice = np.delete(PredsDiffPrice, 0, axis=1)
    return PredsDiffPrice


def readMax(input):
    max = input.max()
    min = input.min()
    gap = max - min
    gap = float(gap)
    return float(max), float(min), gap


def de_normalize(min, gap, input_train, input_test):
    output_train = input_train * gap + min
    output_test = input_test * gap + min
    return output_train, output_test


origin_y = pd.read_csv('originy.csv')
origin_y_diff = origin_y['Close Pfizer'].diff()

ydiff = pd.read_csv('inputs_diff.csv')
y1diff = ydiff.values.squeeze().astype('float64')
maxydiff, minydiff, gapdiff = readMax(origin_y_diff)

y = pd.read_csv('inputs.csv')
y1 = y.values.squeeze().astype('float64')
maxy, miny, gap = readMax(origin_y['Close Pfizer'])

Dates = origin_y['Date']

data_preprodiff = data_process(x, y1diff, 30, 5)
X_traindiff, X_testdiff, y_traindiff, y_testdiff = split(data_preprodiff)

# predict the difference
predsdiff, preds_testdiff = lstm_model(X_traindiff, X_testdiff, y_traindiff, y_testdiff, 30, 64, 1000, 5, train=False,
                                       diff=True)

data_prepro = data_process(x, y1, 30, 5)
X_train, X_test, y_train, y_test = split(data_prepro)
# predict the origin
preds, preds_test = lstm_model(X_train, X_test, y_train, y_test, 30, 64, 1000, 5, train=False, diff=False)

data_preprodiff_noP = data_process(xNoP, y1diff, 30, 5)
X_traindiff_nop, X_testdiff_nop, y_traindiff_nop, y_testdiff_nop = split(data_preprodiff_noP)
# predict the difference
predsdiff_nop, preds_testdiff_nop = lstm_model(X_traindiff_nop, X_testdiff_nop, y_traindiff_nop, y_testdiff_nop, 30, 64,
                                               1000, 5, train=False, diff=True, pandemic=False)





predsde, preds_testde = de_normalize(miny, gap, preds, preds_test)
preds_dediff, preds_test_dediff = de_normalize(minydiff, gapdiff, predsdiff, preds_testdiff)
predsdiff_nop_de,preds_testdiff_nop_de = de_normalize(minydiff, gapdiff, predsdiff_nop, preds_testdiff_nop)
y_testdiff = np.array(y_testdiff.squeeze())
y_test_diff = y_testdiff.astype(np.float64)
y_testde ,y_test_dediff= de_normalize(miny, gap, y_test, y_testdiff)



ytestTrue = np.array(origin_y['Close Pfizer'])
ytestTrue1 = ytestTrue[631:731]
ytestTrue2 = ytestTrue[-105:-5]
ytestTrueTest = np.append(ytestTrue1, ytestTrue2)

# DiffPrice = []
# for i in range(len(preds_test_dediff)):
#     tempList = []
#     for j in range(5):
#         tempList.append(ytestTrueTest[i + j])
#     DiffPrice.append(tempList)
# DiffPrice = np.array(DiffPrice)

DiffPrice = y_testde

true_test = truth(preds_test_dediff, ytestTrueTest)
true_test_nop = truth(preds_testdiff_nop_de, ytestTrueTest)

ytestTrueAll = ytestTrue[30:]
true_all = truth(preds_dediff, ytestTrueAll)
true_all_nop = truth(predsdiff_nop_de, ytestTrueAll)

true_test = (true_test - miny) / gap
DiffPrice = (DiffPrice - miny) / gap
true_test_nop = (true_test_nop - miny) / gap

mse_testDiff = metrics.mean_squared_error(true_test, DiffPrice)
# rmse = math.sqrt(mse_test)
maeDiff = metrics.mean_absolute_error(true_test, DiffPrice)
rvalueDiff = metrics.r2_score(true_test, DiffPrice)
mse_testDifforigin = metrics.mean_squared_error(preds_testdiff, y_testdiff)

mse_test = metrics.mean_squared_error(preds_testde, y_testde)
mae = metrics.mean_absolute_error(preds_test, y_test)
rvalue = metrics.r2_score(preds_test, y_test)
mse_testorigin = metrics.mean_squared_error(preds_test, y_test)

mse_test_nop = metrics.mean_squared_error(true_test_nop, DiffPrice)
mae_nop = metrics.mean_absolute_error(true_test_nop, DiffPrice)
rvalue_nop = metrics.r2_score(true_test_nop, DiffPrice)
mse_testorigin_nop = metrics.mean_squared_error(preds_testdiff_nop, y_testdiff_nop)

# Print these parameters.


print("Mean Squared Error:", mse_testDiff)
print("Mean Absolute Error:", maeDiff)
print("R-squared Value:", rvalueDiff)
print("MSE before denormalization:", mse_testDifforigin)
print()
print("MSE:", mse_test)
print("Mean Absolute Error:", mae)
print("R-squared Value:", rvalue)
print("MSE before denormalization:", mse_testorigin)
print()
print("MSE:", mse_test_nop)
print("Mean Absolute Error:", mae_nop)
print("R-squared Value:", rvalue_nop)
print("MSE before denormalization:", mse_testorigin_nop)
# Plot the prediction of differential values and direct stock price.


plt.title('LSTM prediction of testing data (difference value)')
plt.plot(range(len(true_test)), true_test[:, 0], label='prediction(diff)')
plt.plot(range(len(ytestTrueAdd)), ytestTrueAdd, label='truth')
# plt.plot(range(0,len(trueValue),5),pots,'x',color ='red',label='correction point')
plt.legend()
plt.xlabel('Data units of testing data')
plt.ylabel('Stock price (USD)')

plt.show()

Dates_plt = Dates[30:-6]
plt.figure(figsize=(15, 5), dpi=500)
plt.title('Prediction results of different datasets by LSTM')
plt.plot(Dates_plt, true_all[:, 0], label='prediction(diff)', linewidth=0.5, color='blue')
plt.plot(Dates_plt, true_all_nop[:, 0], label='prediction without pandemic (diff)', linewidth=0.5, color='orange')
plt.plot(Dates_plt, predsde[:, 0], label='prediction(direct)', linewidth=0.5, color='red')
plt.plot(Dates_plt, ytestTrue[:-6], label='truth', linewidth=0.5, color='green')
plt.fill_between(Dates_plt[630:730], 0, max(ytestTrue[:-6]) + 10, facecolor='blue', alpha=0.3)
plt.fill_between(Dates_plt[-104:], 0, max(ytestTrue[:-6]) + 10, facecolor='blue', alpha=0.3)
plt.xlim(Dates_plt.iloc[0], Dates_plt.iloc[-1])
plt.xticks(size=2)
plt.yticks(size=2)
plt.legend()
plt.xlabel('Years')
plt.ylabel('Stock price (USD)')

# plt.savefig('5days correlation.png')

plt.show()
