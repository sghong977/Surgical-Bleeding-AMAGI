import numpy as np
import pandas as pd

from itertools import groupby

import copy

from skimage.morphology import binary_erosion, binary_dilation, opening, closing, erosion, dilation


class FilterBank:
    def __init__(self, seq_list:list, seq_fps=1):

        self.EXCEPTION_NUM = -100
        
        self.seq_list = seq_list # seq_list should be 1D list, list should be composite in 0 or 1. example:[1,0,1,1,0,0,1,1,0...]
        self.seq_fps = seq_fps # default fps(vihub's 1.2 inference fps) = 1

    def list_to_numpy(self, list):
        return np.array(list, dtype=np.int32)

    def numpy_to_list(self, npy):
        return npy.tolist()

    def apply_filter(self, seq_list, filter_type:str, kernel_size:int) :
        """
        appling post processing from support filter [median, opening, closing]

        Args:
            seq_list: 1D input list, [1,0,0,0,1,1,0,1 ...]
            filter_type: str, ['median', 'opening', 'closing']
            kernel_size: int, filter's kernel size

        Returns:
            results: seq_list after applying pp filter. example [1,0,0,1,1,..]
            results error : -100 (EXCEPTION_NUM)
        """

        assert filter_type in ['median', 'opening', 'closing'], "NOT SUPPORT FILTER"
        
        seq_numpy = self.list_to_numpy(seq_list) # converting list to numpy

        #print('\n\n\t\t ===== APPLYING FILTER | type : {} | kernel_size = {} =====\n\n'.format(filter_type, kernel_size))

        results = self.EXCEPTION_NUM # init error value
        
        if filter_type == 'median' :
            #assert kernel_size in [1,3,5,7,9,11], "NOT SUPPORT FILTER SIZE"

            results=self.medfilter(seq_numpy, kernel_size) # 1D numpy
        
        elif filter_type == 'opening' :
            results = self.openingfilter(seq_numpy, kernel_size) # 1D numpy

        elif filter_type == 'closing' :
            results = self.closingfilter(seq_numpy, kernel_size) # 1D numpy


        results = results.astype(seq_numpy.dtype) # convert to original dtype
        results = self.numpy_to_list(results) # convert to numpy to list

        return results # numpy

    # not yet used
    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...] 

    # not yet used
    def decode_list(self, run_length): # run_length -> [0,1,1,1,1,0 ...]
        decode_list = []

        for length, group in run_length : 
            decode_list += [group] * length

        return decode_list

    
    def time_to_filtersize(self, filter_sec, fps):
        """
        calculate filter kernel size from applying filter second

        Args:
            filter_time:float or int, applying filter time you want to convert
            fps: float or int, seq_list's fps, if predict seq's fps is 1fps, you set param fps=1

        Returns:
            filter_size: converted filter kernel size
            results error : -100 (EXCEPTION_NUM)

        example:
            time_to_filtersize(1, 1) => 1
            time_to_filtersize(1, 6) => 6
            time_to_filtersize(3, 30) => 90
        """
        filter_size = self.EXCEPTION_NUM
        
        filter_size = filter_sec * fps

        return int(filter_size)
	

    def medfilter (self, x, k):
        """
        Apply a length-k median filter to a 1D array x. Boundaries are extended by repeating endpoints.

        Args:
            x:1D numpy
            k:kernel_size

        Returns:
            1D numpy
        """
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."

        k2 = (k - 1) // 2
        y = np.zeros ((len (x), k), dtype=x.dtype)

        '''
        print('==> prepare')
        print(y)
        '''

        y[:,k2] = x
        
        '''
        print('\n==> arrange')
        print(y)
        '''
        for i in range (k2):
            j = k2 - i
            y[j:,i] = x[:-j]
            y[:j,i] = x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]

        '''
        print('\n==> margin padding')
        print(y)
        '''

        return np.median (y, axis=1)

    def openingfilter(self, x, iter_count):
        """
        Applying opening mophology calc from 1D sequence

        Args:
            x:1D numpy
            iter_count:  iteration count of opening mophology calculation

        Returns:
            1D numpy
        """
        out = np.copy(x)

        # opening
        for i in range(iter_count):	
            out = erosion(out)
        
        for j in range(iter_count):
            out = dilation(out)

        return out

    def closingfilter(self, x, iter_count):
        """
        Applying closing mophology calc from 1D sequence

        Args:
            x:1D numpy
            iter_count:  iteration count of closing mophology calculation

        Returns:
            1D numpy
        """

        out = np.copy(x)

        # closing
        for i in range(iter_count):	
            out = dilation(out)
        
        for j in range(iter_count):
            out = erosion(out)

        return out

    def apply_best_filter(self):
        """
        Applying apply best pp filter from self.seq_list

        Returns:
            best_pp_results
        """

        pp_results = self.EXCEPTION_NUM

        # apply filter sequence
        pp_results = self.apply_filter(self.seq_list, "opening", self.time_to_filtersize(1, self.seq_fps))
        pp_results = self.apply_filter(pp_results, "closing", self.time_to_filtersize(1, self.seq_fps))

        return pp_results

if __name__ == '__main__':
    # example data
    seq_list = [1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1] # predict sequence
    seq_fps = 1 # sequence's fps

    ##### example 1. when you want to apply best pp filter
    fb = FilterBank(seq_list, seq_fps)
    best_pp_seq_list = fb.apply_best_filter()
    
    print(seq_list)
    print(best_pp_seq_list) # pp results

    ##### example 2. when you want to apply custimizing sequence of pp filter
    # fb2 = FilterBank(seq_list, seq_fps) # seq_fps
    custimize_pp_seq_list = fb.apply_filter(seq_list, "median", kernel_size=3) 
    custimize_pp_seq_list = fb.apply_filter(custimize_pp_seq_list, "opening", kernel_size=1)

    print(seq_list)
    print(custimize_pp_seq_list) # pp results