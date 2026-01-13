import numpy as np


def load_pkslst(file_name, wavelength):
    q2_peaks = []
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith('['):
                if line.startswith('[np.float64('):
                    theta2_peak = float(line.split('[np.float64(')[1].split('), ')[0])
                else:
                    theta2_peak = float(line.split('[')[1].split(', ')[0])
                q2_peak = (2*np.sin(np.pi/180 * theta2_peak/2) / wavelength)**2
                q2_peaks.append(q2_peak)
    return np.array(q2_peaks)
