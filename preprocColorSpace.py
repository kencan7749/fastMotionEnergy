import numpy as np
import cv2
from PIL import Image

def gammacorrect(im, g):
    im = im.astype(np.float) / 255
    if g!=1.0:
        im = im ** g;

    return im

# not completely match to matlab version maybe cv2 cvt CoLOR is needed to transform 0-255 not 0-1
def preprocColorSpace(S, params):
    # Usage: [Spreproc, params] = preprocColorSpace(S, params)
    #
    #  A script for switching between color spaces for color stimuli; relies on
    # color functions in Matlab's ImageProcessing toolbox.
    #
    # Created by SN 2009
    # Modified / commented by ML 2013.03.21
    # reimplement in python by KS 2019.10.26

    # Default parameters


    default_dict = {'colorconv': 'rgb2lab',
                    'colorchannels': 0,
                    'gamma': 1.0,
                    'verbose': True,

                    }

    for key in default_dict.keys():
        if hasattr(params, key) is False:
            print('add ' + key)
            setattr(params, key, default_dict[key])


    # assure image is in 0 - 255 range before  gamma - correcting
    if np.max(S) < 1.1:
        S = S * 255;

    # Assure functions are available
    if params.colorconv == None:
        raise('Selected color conversion function {%s} is not available in your matlab install!  Try using params.colorconv = ''rgb2gray'' or use params = preprocColorSpace_GetMetaParams(2)\n (Or you may simply not have the image processing toolbox...)'+ params.colorconv)

    # color - space conversion
    if params.verbose:
        print('Converting color space ... [' + params.colorconv  +']');

    cstim = np.zeros([S.shape[0],S.shape[1],S.shape[3]],dtype=np.float); # change from original since it is 0
    framenum=S.shape[3];




    for ii in range(framenum):
        tim = np.squeeze(S[:,:,:,ii])
        tim = gammacorrect(tim,params.gamma)
        if params.colorconv == 'rgb2lab':
            #opencv desire to 0-255
            tim = tim * 255
            cim = cv2.cvtColor(tim.astype(np.uint8), cv2.COLOR_RGB2LAB)
            cim = cim.astype(np.float)


        elif params.colorconv == 'rgb2gray':
            tim = tim * 255
            cim = cv2.cvtColor(tim.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cim = cim.astype(np.float)

        cstim[:,:,ii] = cim[:,:,params.colorchannels]

    del S

    if params.verbose:
        print('Done.')

    print(type(cstim), type(params))
    return cstim, params




