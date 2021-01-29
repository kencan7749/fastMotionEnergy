import numpy as np

def preprocNormalise(S, params):
    # Usage: varargout = preprocNormalize(S, params)
    #
    # Normalize channels of a model.Most commonly, z - score each channel, but
    # other methods are available.
    #
    # Inputs:
    # S: stimulus / preprocessed stimulus.Better be 2 D(time x channels)
    # params: parameter struct array, with fields:
    #   .normalize: string, one of the following: 'zscore'[default],
    #               'gaussianize', 'uniform', '0to1', '-1to1'
    #   .reduceChannels: if scalar < 1, keep all channels with stds >
    #                params.reduceChannels * max std; if scalar > 1, keep n
    #                channels;  if 1 or true, use following parameter as index to
    #                keep some channels.Default = [] = do nothing.
    #    [.reduceChannelsValidChannels]: index of channels to keep
    #                (optional, only use if.reduceChannels == 1)
    #    .crop: 2 - element vector[min, max] - crop values above / below
    #                max / min to max / min.Default = [] = do nothing.
    #
    # Outputs:
    # Spreproc: normalized stimulus
    # [params]: preprocessing params, potentially w /.means,.stds added( if
    # params.normalize = 'zscore')
    #
    # ML 2013.03 .20

    # KS 191027

    default_dict = {'class_name': 'preprocNormalize',
                    'reducedChannels': [],
                    'crop': [],
                    'normalize': 'zscore', #'gaussianize', 'kameUniform', '0to1', '-1to1'

                    }

    for key in default_dict.keys():
        if hasattr(params, key) is False:
            print('add ' + key)
            setattr(params, key, default_dict[key])


    if params.normalize == 'zscore':
        if hasattr(params, 'means'):
            Spreproc = (S - params.mean) / params.stds
        else:
            means = np.mean(S,0)
            stds = np.std(S,0)
            params.means = means
            params.stds = stds
            Spreproc = (S - params.means) / params.stds
    elif params.normalise == 'zscore':
        #omit now
        pass
    elif params.normalize == 'gaussianize':
        #omitnow
        pass
    elif params.normalize == 'uniform':
        #omit now
        pass
    elif params.normalize == '0to1':
        #omit now
        pass
    elif params.normalise == '-1to1':
        #omit now
        pass

    # reduce the number of channes based on the standard deviation of hannes, or
    # a pre-defined index

    if hasattr(params, 'reduceChannels') == False:
        print('Reducing channels is not well tested in re-implementation of code yet!')
        print('Skip now')
        pass

    #Crop values to fit withn specified range
    if hasattr(params, 'crop'):
        #Spreproc = np.max([Spreproc, crop[0]])
        #Spreproc = np.min([Spreproc, crop[1]])
        #???? maybe wrong need to be checked matlab
        pass
    return Spreproc, params


