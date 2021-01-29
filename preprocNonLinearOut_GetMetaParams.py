class preprocNonLinearOut_GetMetaParams(object):

    def __init__(self, argNum=1):
        self.class_name = 'preprocNonLinearOut'
        if argNum == 1:
            # Original params recommended by SN
            self.gainControl = []; # Broken: gain control for each channel based on luminance / color.
            self.gainControlOut = []; # Broken: gain control for each channel based on luminance / color.
            self.nonLinOutExp = 'log'; # Output nonlinearity
            self.nonLinOutParam = 1.0000e-05; # delta to add t channel values to prevent log(0) = -inf

        elif argNum == 2:
            # Original params recommended by SN
            self.gainControl = []; # Broken: gain control for each channel based on luminance / color.
            self.gainControlOut = []; # Broken: gain control for each channel based on luminance / color.
            self.nonLinOutExp = .5; # Output nonlinearity
            self.nonLinOutParam = 1.0000e-05; # delta to add to channel values to prevent log(0) = -inf

        else:
            raise ('Unknown parameter configuration!');
