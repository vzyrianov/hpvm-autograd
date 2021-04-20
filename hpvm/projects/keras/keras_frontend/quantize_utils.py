

from scipy import stats
from numpy import linalg
import numpy as np
import sys


# NOTE: enable/disable smart quantization of weights and activations
smart_quantization = False


def quantize_arr(input_arr, min_val, max_val):

    quantize_range = 256.0
    input_range = max_val - min_val
    mul_factor = input_range / quantize_range

    v1 = np.subtract(input_arr, min_val)
    v2 = np.divide(v1, mul_factor)
    v3 = v2.astype(int)
    v4 = np.multiply(v3, mul_factor)
    v5 = np.add(v4, min_val)
    v6 = np.clip(v5, min_val, max_val)

    return v6


def compute_norm(a1, a2):

    norm_inp = np.subtract(a1, a2)
    #norm = linalg.norm(norm_inp, ord = 1)
    norm = np.sum(np.abs(norm_inp))
    print ("*** norm = ", norm)
        
    return norm
    

def get_best_quant_range(input_arr):

    # For disabled smart quantization, skip expensive quant range computation
    if smart_quantization == False:
        min_val = np.percentile(input_arr, 0.1)
        max_val = np.percentile(input_arr, 99.9)
        return (min_val, max_val)


    # Trying different threshold values for INT8 quantization    
    min_percentiles = [0.0]
    max_percentiles = [99.9, 99.8, 99.7, 99.5]
     
    min_norm = 100000000
    min_pair = (0, 100)
    range_vals = (0, 0)
    for i in min_percentiles:
      for j in max_percentiles:
        print (" i = ", i, " j = ", j, " \n")
        min_val = np.percentile(input_arr, i)
        max_val = np.percentile(input_arr, j)

        res = quantize_arr(input_arr, min_val, max_val)    
        norm = compute_norm(res, input_arr)

        if norm < min_norm:
          min_norm = norm
          min_pair = (i, j)
          range_vals = (min_val, max_val)

    print ("--- min_norm = ", min_norm, " , min_pair = ", min_pair ,  "  range_vals = ", range_vals)


    return range_vals                
    
    


def dumpQuantizeRanges(weights_dir, input_min, input_max, w_max, w_min, \
                       b_max, b_min, output_min, output_max):


    outfile_path = weights_dir + "/quant_ranges.txt"  

    f = open(outfile_path, "a+")

    f.write(str(input_min) + " " + str(input_max) + " " + str(w_min) + " " + str(w_max) + " " + \
            str(b_min) + " " + str(b_max) + " " + str(output_min) + " " + str(output_max) + "\n")

    f.close()
    

    
    




     
if __name__ == "__main__":


    vals = np.zeros((2,3))
    vals[0][0] = 1.2
    vals[0][1] = 0.48
    vals[0][2] = 0.5
    vals[1][0] = -0.3
    vals[1][1] = 0.25
    vals[1][2] = 0.46

    input_arr = np.array(vals)

    res = quantize_arr(input_arr, -0.3, 1.4)

    print (res, "\n")

    divergence = compute_norm(res, input_arr) 

    print ("divergence = ", divergence, "\n")
