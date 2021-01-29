import numpy as np

def preprocNonLinearOut(S,params):
    # Usage: [Spreproc, params] = preprocNonLinearOut(S, params)
    #
    # Output nonlinearity on each channel of a model.Generally done BEFORE
    # (zscore or other) normalization.
    #
    # Inputs:
    # S: Stimulus or PreprocessedStimulus(STRFlab classes), or simple
    # numerical matrix(assumed to be nFrames x nChannels)
    # params: a struct array of parameters, with fields:
    #   .nonLinOutExp: exponent to which to raise each channel value,
    #                   thus: abs(S). ^ x. * sign(S) Multiple alues(e.g.
    #                   [.5, 2]) raise each column to each different
    #                   exponent, and concatenate the results(here,
    #                   doubling the number of channels).
    #                   Alternately, this can be 'log', w / another
    #                   parameter(see next)
    #   .nonLinOutParam = []; % This need filling in for certain options
    #                        of nonLinOutExp(e.g. for 'log', it specifies a
    #                       small delta to add to avoid log(0) = -inf
    #   .verbose = T / F, verbose printing
    #
    #   Modified from SN code by ML 2013.03 .21
    #   Ported to python by KS 2019.10.27

    # Default parameters

    default_dict = {'class_name': 'preprocNonLinearOut',
                    'nonLinOutExp': 0.5,
                    'verbose': False}


    for key in default_dict.keys():
        if hasattr(params, key) is False:
            print('add '+ key)
            setattr(params,key,default_dict[key])

    if hasattr(params, 'verbose'):
        if params.verbose:
            print('Processing static output nonlinearity...');

    #.nonLinOutParam has a differen meaning for each different output nonlinearity.

    if type(params.nonLinOutExp) == 'str':
        if params.nonLinOutExp == 'log':
            #For log, d in a small value to add to assure no - Inf channels
            d  = params.nonLinOutParam
            for ii in range(S.shape[1]):
                S[:,ii] = np.log(d + S[:,ii])
        elif params.nonLinOutExp == 'logstd':
            if hasattr(params,'nonLinOutStd'):
                stds = params.nonLinOutStd
            else:
                stds = np.nanstd(S,0)
                params.nonLinOutStd = stds
            # For logstd d is also small value to avoid -INf
            d = params.nonLinOutParam
            for ii in range(S.shape[1]):
                S[:,ii] = np.log(d + S[:,ii]/stds[ii])
        elif params.nonLinOutExp == 'logmean':
            #omit now
            pass
        elif params.nonLinOutExp == 'linear':
            print('Linearity requested Do nothing')

    else:
        if len(params.nonLinOutExp) == 1:
            S = np.abs(S) ** params.nonLinOutExp ** np.sign(S)

        else:
            #omit now
            pass

    params.nChan = S.shape[1]

    return S, params




