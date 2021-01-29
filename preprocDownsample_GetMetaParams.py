class preprocDownsample_GetMetaParams(object):

    def __init__(self, argNum=1):
        if argNum == 1:
            # Simple box average, for TR=1, imhz=15
            self.dsType = 'box';
            self.imHz = 15; # movie / image sequence frame rate
            self.sampleSec = 1; # TR
            self.frameshifts = []; # empty = no shift
            self.gaussParams = []; # [1, 2]; % sigma, mean

        elif argNum == 2:
            # Simple box average, for TR=2, imhz=15
            self.dsType = 'box';
            self.imHz = 15; # movie / image sequence frame rate
            self.sampleSec = 2; # TR
            self.frameshifts = []; # empty = no shift
            self.gaussParams = []; # [1, 2]; % sigma, mean
        elif argNum == 3:
            # Gaussian downsampling, for TR=2, imhz=15
            self.dsType = 'gauss';
            self.imHz = 15; # movie / image sequence frame rate
            self.sampleSec = 2; # TR
            self.frameshifts = []; # empty = no shift
            self.gaussParams = [1, 2]; # mean, standarddeviation
        elif argNum == 4:
            # Max downsampling, for TR=2, imhz=15
            self.dsType = 'max';
            self.imHz = 15; # movie / image sequence frame rate
            self.sampleSec = 2; # TR
            self.frameshifts = []; # empty = no shift
            self.gaussParams = []; # [1, 2]; # sigma, mean
        elif argNum ==5:
            #Simple box average, for TR=2, imhz=24
            self.dsType = 'box';
            self.imHz = 24; # movie / image sequence frame rate
            self.sampleSec = 2; # TR
            self.frameshifts = []; # empty = no shift
            self.gaussParams = []; # [1, 2]; % sigma, mean
        elif argNum == 6:
            # Simple box average, for TR=1, imhz=24
            self.dsType = 'box';
            self.imHz = 24; # movie / image sequence frame rate
            self.sampleSec = 1; # TR
            self.frameshifts = []; # empty = no shift
            self.gaussParams = []; # [1, 2]; % sigma, mean
        else:
            raise('Unknown parameter configuration!');

