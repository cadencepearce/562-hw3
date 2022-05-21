import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift
from sklearn import svm
import pickle
import time

def main():
    # # make_data()
    # with open('pk_dct.p', 'rb') as handle:
    #     pk_dct = pickle.load(handle)
    # # print(pk_dct)
    # X_norms = [list(pk_dct[n]) for n in pk_dct]
    # print(X_norms)



    

    frequency = 19000.

    rec = record_data(freq=frequency)
    data = get_fft(data=rec)


    with open('awayData20More.pickle', 'rb') as handle:
        aw = pickle.load(handle)
    X_awymore = np.squeeze([get_fft(aw[a]) for a in aw])

    with open('towardData20More.pickle', 'rb') as handle:
        t = pickle.load(handle)
    X_twdmore = np.squeeze([get_fft(t[a]) for a in t])

    with open('twd.pickle', 'rb') as handle:
        twd = pickle.load(handle)
    X_twds = np.squeeze([get_fft(twd[a]) for a in twd])

    with open('away.pickle', 'rb') as handle:
        away = pickle.load(handle)
    X_aways = np.squeeze([get_fft(away[a]) for a in away])

    X_away = np.concatenate([X_awymore, X_aways])
    X_twd = np.concatenate([X_twdmore, X_twds])

    # labels
    y = ['twds'] * len(X_twd) + ['away'] * len(X_away)
    x = np.concatenate([X_twd, X_away])

    # sklearn
    clf = svm.SVC(probability=True)
    clf.fit(x, y)


    # predict
    prob = clf.predict_proba([data])
    pred = clf.predict([data])
    print(pred)

    NFFT=1024
    fs = 44100.
    fVals=np.arange(start = -NFFT/2,stop = NFFT/2)*fs/NFFT
    fVals = list(np.array(fVals, int)[930:977])

    fig4, ax = plt.subplots(nrows=1, ncols=1) #create figure handle
    ax.plot(fVals, data, 'b')
    ax.set_title(pred)
    ax.set_xlabel('Frequency (Hz)')         
    ax.set_ylabel('|DFT Values|')
    plt.show()


def record_data(freq):
    print('wait')
    time.sleep(1)
    print('record')
    frequency = freq
    fs = 44100.
    seconds = 2

    t = np.linspace(0,seconds,round(seconds*fs))
    y = np.sin(frequency*t*2*np.pi)

    myRecordForward = sd.playrec(y,fs,channels=1)
    sd.wait()

    return myRecordForward


def get_fft(data):
    # eliminate noise at the beginning of the recording
    dataInputMove = data[40000:,0]

    # make FFT
    delta = 1000
    NFFT = 1024
    X = fftshift(fft(dataInputMove, NFFT))
    X[int(NFFT/2)-1:int(NFFT/2)+1] = 0

    return np.abs(X)[930:977] # FFT window around frequency peak for 18kHz
    

def learn(no_gesture, away, twd):
    clf = svm.SVC()
    clf.fit(no_gesture, away)


def make_data():
    pk_dct = dict()
    for i in range(10):
        frequency = 19000.
        rec = record_data(freq=frequency)
        data = get_fft(data=rec)
        pk_dct[i] = data

        pickle.dump(pk_dct, open( "pk_dct.p", "wb" ) )


def plot_data(file):
    with open(file, 'rb') as handle:
        pk_dct = pickle.load(handle)
    # print(pk_dct)
    X_norms = [list(pk_dct[n]) for n in pk_dct]
    print(X_norms)
    

    NFFT=1024
    fs = 44100.
    fVals=np.arange(start = -NFFT/2,stop = NFFT/2)*fs/NFFT
    fVals = list(np.array(fVals, int)[930:977])
    # print(fVals)
    n = 0
    for norm in X_norms:
        fig4, ax = plt.subplots(nrows=1, ncols=1) #create figure handle
        ax.plot(fVals, norm, 'b')
        ax.set_title(str(n))
        plt.show()
        n += 1

main()
