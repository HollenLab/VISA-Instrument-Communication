import numpy as np
from scipy.fft import fft, fftfreq, ifft, fftshift
from scipy.interpolate import CubicSpline
from skrf import Network

def read_transfer_func(filename: str):
    '''
    Purpose: This function read a two-channel touchstone file containing information 
    about how pulses are distorted in a system and returns arrays representing the 
    transfer function of the system.

    Arguments:
    file_name: The name of the touchstone file

    Returns: 
    s21: The transfer function
    swept_fs: The frequencies to which the values of s21 correspond
    '''
    
    net = Network(filename)
    s21 = net.s[:,1,0]
    swept_fs = net.f
    return s21, swept_fs

def make_transfer_func(s21, swept_fs):
    '''
    Purpose: This function reads the transfer function of a system for a range of positive 
    frequencies and creates a function that interpolates between these measurements and 
    for negative frequency inputs returns the complex conjugate of the value at the 
    corresponding positive frequency.  For inputs outside of the range of the input arrays 
    this function returns zero.

    Arguments:
    s21: The transfer function
    swept_fs: The frequencies to which the values of s21 correspond

    Returns:
    full_transfer_func: A function that will return the value of s21 at the input frequency.
    upper_limit: The highest frequency for which the measurements are valid
    lower_limit: The lowest frequency for which the measurements are valid

    IMPORTANT: For frequencies with magnitudes greater than the upper_limit or less than 
    the lower_limit, the 
    function will return zero.
    '''

    transfer_func = CubicSpline(swept_fs, s21)
    upper_limit = swept_fs[-1]
    lower_limit = swept_fs[0]

    def full_transfer_func(arr):
        answer = transfer_func(arr)
        negative = arr < 0
        out_of_bounds = np.logical_or(arr < -1 * upper_limit, arr > upper_limit)
        out_of_bounds = np.logical_or(out_of_bounds, np.logical_and(arr < lower_limit, arr > -1*lower_limit) )
        answer = np.where(negative, np.conjugate(transfer_func(-1 * arr)), answer)
        answer = np.where(out_of_bounds, np.complex128(np.zeros_like(answer)), answer)
        return answer
    
    return full_transfer_func, upper_limit, lower_limit

def shape_pulse(pulse: np.array, time_int: np.array, transfer_func, \
                scaling_limit = 0.1) -> np.array:
    '''
    Purpose: This function takes in a pulse and shapes it into a different pulse 
    that will be distorted into the desired pulse after going through the system with 
    the given transfer function.  

    IMPORTANT: If the magnitude of a certain frequency 
    needs to be scaled by a quantity greater than the amount given by scaling_limit
    then the program will simply cut that frequency out of the signal rather than 
    trying to force enormous voltages through what may essentially be a low/high pass
    filter.

    Arguments:
    pulse: The desired pulse
    transfer_func: The transfer function of the system
    time_int: The time interval of the pulse (must be same shape as pulse)
    scaling limit: The minimum value of the magnitude of the transfer function below 
    which the function will decide that the system will not pass the frequency in question 
    and cuts it out of the signal

    Returns:
    shaped_pulse: A numpy array of the modified pulse over the same time interval with 
    the same time interval as the original
    '''

    dt = time_int[1] - time_int[0]
    freqs = fftfreq(len(pulse), dt)
    transfer_array = transfer_func(freqs)
    shaped_pulse = ifft( np.divide(fft(pulse), transfer_array, \
                                   where = (np.absolute(transfer_array) > scaling_limit), \
                                    out = np.complex128(np.zeros_like(transfer_array))) )
    return shaped_pulse