import numpy as np
import scipy as sp

def computeCO(weight, height, sex):
    bsa = np.sqrt(weight * height / 3600)
    if sex == 'F':
        Vtot = (3.47 * bsa - 1.954)*1000
    else:
        Vtot = (3.47 * bsa - 1.229)*1000
        # Vtot = (2.63 * bsa + 0.146) / 1000
    CO = Vtot / 60

    return CO

def computeMAP(bpSignal, fs):
    # find systolic (high pressure) peaks
    print(bpSignal.shape, fs)
    sysPksIdces = sp.signal.find_peaks(bpSignal, height=25, distance=0.75*fs)[0].astype(int) # come back and fix this

    map = np.zeros(sysPksIdces.size-1)
    for i in range(sysPksIdces.size-1):
        systolicBp = bpSignal[sysPksIdces[i]]
        diastolicBp = np.min(bpSignal[sysPksIdces[i]:sysPksIdces[i+1]])
        map[i] = diastolicBp + (systolicBp - diastolicBp)/ 3.0
    
    return (map, sysPksIdces)

def computeRR(sysPkIdces, fs):
    rr = np.diff(sysPkIdces) / fs
    return rr