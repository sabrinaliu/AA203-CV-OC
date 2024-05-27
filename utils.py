import numpy as np

def computeCO(weight, height, sex):
    bsa = np.sqrt(weight * height / 3600)
    if sex == 'F':
        Vtot = (3.47 * bsa - 1.954) / 1000
    else:
        Vtot = (3.47 * bsa - 1.229) / 1000
        # Vtot = (2.63 * bsa + 0.146) / 1000
    CO = Vtot / 60

    return CO