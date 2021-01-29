class preprocNormalize_GetMetaParams(object):

    def __init__(self, argNum=1):

        self.class_name = 'preprocNormalize'

        if argNum == 1:
            # Original arguments recmomended by SN
            self.valid_w_index = []; # specific index of channels to keep(overrides.reduceChannels)
            self.reduceChannels = []; # number of channels / pct of channels to keep
            self.normalize = 'zscore'; # normalization method
            self.crop = []; # min / max to which to crop; empty does nothing

        elif argNum ==2:
            self.valid_w_index = [];  # specific index of channels to keep(overrides.reduceChannels)
            self.reduceChannels = [];  # number of channels / pct of channels to keep
            self.normalize = 'gaussianize';  # normalization method
            self.crop = [];  # min / max to which to crop; empty does nothing

        elif argNum==3:
            # Original arguments recmomended by SN; crops to[-3.5, 3.5]
            self.valid_w_index = []; # specific index of channels to keep(overrides.reduceChannels)
            self.reduceChannels = []; # number of channels / pct of channels to keep
            self.normalize = 'zscore'; # normalization method
            self.crop = [-3.5, 3.5]; # min / max to which to crop empty does nothing

        elif argNum == 4:
            raise('blank for now')

        elif argNum == 5:
            # Original arguments recmomended by SN; crops to[-3.5, 3.5]
            self.valid_w_index = [];  # specific index of channels to keep(overrides.reduceChannels)
            self.reduceChannels = [];  # number of channels / pct of channels to keep
            self.normalize = 'zscore';  # normalization method
            self.crop = [-5, 5];  # min / max to which to crop empty does nothing
            self.useTrnParams = True
