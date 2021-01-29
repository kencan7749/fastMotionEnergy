import numpy as np

def preprocDownsample(S, params):
    # Usage: [Spp, params] = preprocDownsample(S, params)
    #
    # Downsamples stimuli to(fMRI or other) sampling rate; e.g., takes frames
    # of a movie presented at 15 Hz and downsamples them to an fMRI sampling
    # rate(TR) of .5 Hz(2 seconds / measurement).
    #
    # Inputs:
    #  S = preprocessed stimulus matrix, [time x channels]
    # params = parameter struct, with fields:
    #       .dsType = string specifying type of downsampling: 'box'[default],
    #                'gauss', 'max', or 'none'
    #       .gaussParams = 2 - element array specifying[,] (??).Only necessary
    #               if params.dsType = 'gauss'
    #       .imHz = frame rate of stimulus in Hz
    #       .sampleSec = length in seconds of 1 sample of data(for fMRI, this
    #                     is the TR or repetition time)
    #       .frameshifts = amount to shift frames; empty implies no shift
    #       .gaussParams = standard deviation and temporal offset for Gaussian
    #                        downsampling window
    # Output:
    #    Spp

    default_dict = {'dsType': 'box',
                    'imHz': 15,
                    'sampleSec': 2,
                    'frameshifts': [],
                    'gaussParams': []
                    }

    for key in default_dict.keys():
        if hasattr(params, key) is False:
            print('add ' + key)
            setattr(params, key, default_dict[key])

    fr_per_sample = params.sampleSec * params.imHz
    # downsample the preprocessed stimuli
    if params.dsType == 'box':
        if hasattr(params, 'frameshifts'):
            if params.frameshifts != []:
                print('shifting %d frames...\n', params.frameshifts);
                S = np.roll(S, params.frameshifts[0])
            tframes = int(np.floor(S.shape[0]/ fr_per_sample) * fr_per_sample)
            S = S[:tframes]
            S = S.reshape([fr_per_sample, -1, S.shape[1]], order='C') #???
            S = np.mean(S,0)  #???
    elif params.dsType == 'none':
        pass
    elif params.dsType == 'max':
        #omit now
        pass
    elif params.dsType == 'gauss':
        #omit now
        pass

    return S, params

