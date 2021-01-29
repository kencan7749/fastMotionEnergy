#!/usr/bin/python
#-*- coding: utf-8 -*- 
"""
fastMotionEnergyModel class
    python package for calculating motion energy of gray scale video 
"""

import os
import sys

import numpy as np

sys.path.append('./utils')
from make3dgabor_frames import make3dgabor_frames

sys.path.append('./')
from preprocWavelets_grid import preprocWavelets_grid, getGaborParameters


from preprocWavelets_grid_GetMetaParams import preprocWavelets_gird_GetMetaParams

#sys.path.append('/home/shirakawa/kspy/bdpy3')
#from fig import makefigure, draw_footnote, box_off

class fastMotionEnergyModel():
    """[summary]
    """

    def __init__(self, stimsize, params, Weight = None):
        """[summary]

        Args:
            stimsize ([type]): determine the height and width of video
            params ([type]): param class 
        """

        self.stimxytsize = stimsize
        if Weight is not None:
            self.W = Weight
        else:
            self.W = None
        

        if self.W is None:
            print('loading weight')
            self.W  = self._load_weight(stimsize, params)

        
    def calculate(self, S):
        """ calculate motion energy from S

        Args:
            S (np.array): (h, w, frame) <- (gray scale video)


        Output:
            motion energy map whose shape (frame, #of filter)
        """

        # reshape 2D (hxw, fr)
        S_2D = S.reshape(np.int(self.xypixels), self.stimxytsize[2])
        assert S_2D.shape != S.reshape(-1, self.stimxytsize[-1])

        #Obtain gabor weight
        gabor0_list = self.W[0]
        gabor90_list = self.W[1]

        scw0_list = self.W[2]
        scw90_list = self.W[3]

        #spatial filter
        gout0_arr, gout90_arr = self._spatial_filters(S_2D, gabor0_list, gabor90_list)
        

        #temporal filter
        gout0_h1_arr, gout0_h2_arr  = self._temporal_filters(gout0_arr, scw0_list, scw90_list)
        gout90_h1_arr, gout90_h2_arr = self._temporal_filters(gout90_arr, scw0_list, scw90_list)

        #oriented_liser  (right part of fig 18)
        sum_response = gout0_h1_arr + gout90_h2_arr
        sub_response = gout0_h2_arr - gout90_h1_arr

        #squared sum (motion energy)
        motionenergy = sum_response ** 2 + sub_response ** 2

        return motionenergy

    def _load_weight(self, stimsize, params):
        """[summary]

        Args:
            stimsize ([type]): [description]
            params ([type]): [description]
        """
        setval = self._set_variable(stimsize, params)
        masklimit = setval[0]
        maskenv_below = setval[1]
        wcount = setval[2]
        stimsize = setval[3]
        aspect_ratio = setval[4]
        patchxytsize = setval[5]
        xypixels = setval[6]
        #it is redundant
        self.stimsize = stimsize
        self.xypixels = xypixels
        self.patchxytsize = patchxytsize
        #load gaussian parmeters
        gparams = getGaborParameters(params,aspect_ratio)[0]

        waveletchannelnum = gparams.shape[1]

        W =  self._load_gaussian_weight(params, gparams)
        return W


    def _set_variable(self, stimsize, params):
        """[summary]

        Args:
            stimsize ([type]): [description]
            params ([type]): [description]

        Returns:
            [type]: [description]
        """
        masklimit = 0.001 ## pixel value < masklimit
        maskenv_below = 0.1 # spatial envelope < maskenv_below x stimulus size

        #prepare for the weight
        wcount = 0

        # Stimulus aspect ratio; always X/åY only support aspect ratio is 1 
        aspect_ratio = stimsize[1]/ stimsize[0]

        patchxytsize = list(stimsize[:2]) + [params.tsize]
        xypixels = np.prod(patchxytsize[:2])

        return (masklimit, maskenv_below, wcount,
                stimsize, aspect_ratio, patchxytsize, xypixels)

    def _load_gaussian_weight(self,params, gparams):
        """[summary]

        Args:
            params ([type]): [description]
            gparams ([type]): [description]

        Returns:
            [type]: [description]
        """

        waveletchannelnum = gparams.shape[1]
        gabor0_list = []
        gabor90_list = []
        scw0_list = []
        scw90_list  = []
        #obtain each weight according to gparams
        for ii in range(waveletchannelnum):
            thisgparam = gparams[:,ii].reshape([9,1])

            gabor0, gabor90, gtw = self._get_each_3dgabor_weight(thisgparam, params)
            #store = weight
            gabor0_list.append(gabor0.flatten())
            gabor90_list.append(gabor90.flatten())

            scw0_list.append(gtw[0])
            scw90_list.append(gtw[1])

        #numpy array
        gabor0_list = np.array(gabor0_list).astype(np.float32)
        gabor90_list = np.array(gabor90_list).astype(np.float32)

        scw0_list = np.array(scw0_list).astype(np.float32)
        scw90_list = np.array(scw90_list).astype(np.float32)
        
        return [gabor0_list,  gabor90_list, scw0_list, scw90_list]


    def _get_each_3dgabor_weight(self, gparam , params):
        """[summary]

        Args:
            gparam ([type]): [description]
            params ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        lastgparam = np.zeros([9,1])

        if any(gparam[[0,1,2,3,4,5,8]] != lastgparam[[0,1,2,3,4,5,8]]):
            diff = True
        else:
            diff = False

        if diff: #this strucutre is bad
            if params.gaborcachemode ==2:
                gabors = params.gaborcache[:,:,ii]
                gtw = params.gtwcache[:,:,ii]
                # need to fix
            else:

                [gabor0, gabor90, gtw] = make3dgabor_frames(self.patchxytsize, np.vstack([gparam[:7],[[0]]
                                                                                    ,gparam[8]]))

        return gabor0, gabor90, gtw

    def _spatial_filters(self, S, gabor0, gabor90):
        """[summary]

        Args:
            S ([type]): [description]
            gabor0 ([type]): [description]
            gabor90 ([type]): [description]

        Returns:
            [type]: [description]
        """

        gout0 = np.dot(gabor0, S)
        gout90 = np.dot(gabor90, S)

        return gout0, gout90

    def _temporal_filters(self, gout, scw0, scw90):
        """[summary]

        Args:
            gout ([type]): [description]
            scw0 ([type]): [description]
            scw90 ([type]): [description]

        Returns:
            [type]: [description]
        """
        gout_h1 = np.einsum("ij,ik->ijk", gout, scw0)
        gout_h2 = np.einsum("ij,ik->ijk", gout, scw90)
        return gout_h1, gout_h2 

if __name__ ==  '__main__':
    from preprocWavelets_grid_GetMetaParams import preprocWavelets_gird_GetMetaParams
    params = preprocWavelets_gird_GetMetaParams(2)
    me = fastMotionEnergyModel((224, 224, 16), params)

    rand_vid = np.random.rand(1,16, 224, 224, 3)
    #convert gray
    rand_gray_vid = np.mean([rand_vid[...,0],rand_vid[...,1],rand_vid[...,2], 0])
    rand_gray_vid_transpose = rand_gray_vid[0].transpose(1,2,0)
    print('start calculating')
    output= me.calculate(rand_gray_vid_transpose)

    print(np.mean(output))



