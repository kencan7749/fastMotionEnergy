import numpy as np



class preprocColorSpace_GetMetaParams(object):

    def __init__(self, argNum =1):
        if argNum == 1:
            # Convert to L * A * B colorspace, keep luminance channel

            self.class_name = 'preprocColorSpace';

            self.colorconv = 'rgb2lab';
            self.colorchannels = 1;
            self.gamma = 1.0;
            self.verbose = True;
        elif argNum == 2:
            # Convert to grayscale using rgb2gray(inferior, but present on
            # older matlab versions
            self.class_name = 'preprocColorSpace';

            self.colorconv = 'rgb2gray';
            self.colorchannels = 1;
            self.gamma = 1.0;
            self.verbose = True;
        else:
            raise('Unknown argument!')


class preprocWavelets_gird_GetMetaParams(object):
    # Usage: params = preprocWavelets_grid_GetMetaParams(Arg)
    # Returns parameter set for a preprocWavelets_grid.The parameters
    # specified by setting Arg to 2 are the parameters used to compute
    # motion energy in Nishimoto et al(2011); Arg = 1 is a similar model
    # with fewer channels that provides comparable results for modeling
    # fMRI data.
    #
    # See code for other parameter presets.

    def __init__(self, argNum=2):
        if argNum == 1:
            # Motion energy model with fewer features
            # Housekeeping
            self.class_name = 'preprocWavelets_grid';

            self.show_or_preprocess = 1; # True to preprocess; false to return gabor channels
            self.verbose = 1;
            self.gaborcachemode = 0; # whether or not to cache calculated results(could become faster)
            self.valid_w_index = np.nan; # Select particular gabor channels by number
            # Temporal frequency params
            self.tfdivisions = 3; # Number of temporal frequencies; [tfmin...tfmax] or [0, tfmin...tfmax] if zerorf = 1
            self.tfmax = 2.66667; # = 4 hz @ 15 fps([tfmax] cycles per[tsize] frames at 15 fps; 2.66667 / 10 * 15 = 4 Hz)
            self.tfmin = 1.33333; # = 2 hz @ 15 fps(1.33333 / 10 * 15 = 2 Hz)
            self.tsize = 10; # The size of temporal window(frames)
            self.tf_gaussratio = 10; # temporal frequency to gauss envelope ratio of Gabor bigger number = more waves(larger envelope)
            self.tenv_max = 0.3000; # the maximum gaussian envelope size(relative to tsize)
            self.zerotf = 1; # Include 0 Hz(static) energy channels
            self.f_gaussratio = .5; # frequency to gauss ratio of Gabor; obsolete
            # Orientation / direction params
            self.dirdivisions = 8;
            self.local_dc = 1; # T / F.Include circular gaussians(w / no spat.freq.)
            self.directionSelective = 1;
            # Spatial extent params
            self.sfdivisions = 5;
            self.sfmax = 24; #
            self.sfmin = 1.5; #
            self.f_step_log = 1; # Applies to both SF and TF?
            self.std_step = 4; # Governs how closely spaced channels are; a reasonable range is 2.5 - 4
            self.sf_gaussratio = 0.6000; # 81 channels @ maxsf = 24; 9 x9; 13 x13 @ maxsf = 32
            self.fenv_mode = 0; # use same env for spatial & temporal gabors
            self.senv_max = 0.3000;
            self.wrap_all = 0; # whether or not the filters cover the very edge of images
            # Handling phase
            self.phasemode = 0; # Determines how to do phase(square & sum quadrature pairs(i.e., energy), linear, rectified, etc)
            self.phasemode_sfmax = np.nan; # Calculate only energy channels(e.g., no linear channels) if sf exceeds this number.
            self.zeromean = 1;

        elif argNum == 2:
            # larger motion energy model
            # ( as in Nishimoto et al 2011)
            # STRFlab conventions, housekeeping
            self.class_name = 'preprocWavelets_grid';

            self.show_or_preprocess = 1;  # True to preprocess; false to return gabor channels
            self.wrap_all = 0;
            self.verbose = 1;
            self.gaborcachemode = 0;  # whether or not to cache calculated results(could become faster)
            self.valid_w_index = np.nan;  # Select particular gabor channels by number
            # Temporal frequency params
            self.tfdivisions = 3;  # Number of temporal frequencies; [tfmin...tfmax] or [0, tfmin...tfmax] if zerorf = 1
            self.tfmax = 2.66667;  # = 4 hz @ 15 fps([tfmax] cycles per[tsize] frames at 15 fps; 2.66667 / 10 * 15 = 4 Hz)
            self.tfmin = 1.33333;  # = 2 hz @ 15 fps(1.33333 / 10 * 15 = 2 Hz)
            self.tsize = 10;  # The size of temporal window(frames)
            self.tf_gaussratio = 10;  # temporal frequency to gauss envelope ratio of Gabor bigger number = more waves(larger envelope)
            self.tenv_max = 0.3000;  # the maximum gaussian envelope size(relative to tsize)
            self.zerotf = 1;  # Include 0 Hz(static) energy channels

            # Orientation / direction params
            self.dirdivisions = 8;
            self.local_dc = 1;  # T / F.Include circular gaussians(w / no spat.freq.)
            self.directionSelective = 1;
            # Spatial extent params
            self.sfdivisions = 5;
            self.sfmax = 32;  #
            self.sfmin = 2;  #
            self.f_step_log = 1;  # Applies to both SF and TF?
            self.std_step = 3.5;  # Governs how closely spaced channels are; a reasonable range is 2.5 - 4
            self.sf_gaussratio = 0.6000;  # 81 channels @ maxsf = 24; 9 x9; 13 x13 @ maxsf = 32
            self.fenv_mode = 0;  # use same env for spatial & temporal gabors
            self.senv_max = 0.3000;
            self.wrap_all = 0;  # whether or not the filters cover the very edge of images
            # Handling phase
            self.phasemode = 0;  # Determines how to do phase(square & sum quadrature pairs(i.e., energy), linear, rectified, etc)
            self.phasemode_sfmax = np.nan;  # Calculate only energy channels(e.g., no linear channels) if sf exceeds this number.
            self.zeromean = 1;

        elif argNum == 3:
            # Same as 1 but Np Pyramid (high spatial frequency channels only)
            # smaller motion energy model
            # StRFlab conventions, housekeeping

            self.class_name = 'preprocWavelets_grid';

            self.show_or_preprocess = 1;  # True to preprocess; false to return gabor channels
            self.wrap_all = 0;
            self.verbose = 1;
            self.gaborcachemode = 0;  # whether or not to cache calculated results(could become faster)

            # Temporal frequency params
            self.tfdivisions = 3;  # Number of temporal frequencies; [tfmin...tfmax] or [0, tfmin...tfmax] if zerorf = 1
            self.tfmax = 2.66667;  # = 4 hz @ 15 fps([tfmax] cycles per[tsize] frames at 15 fps; 2.66667 / 10 * 15 = 4 Hz)
            self.tfmin = 1.33333;  # = 2 hz @ 15 fps(1.33333 / 10 * 15 = 2 Hz)
            self.tsize = 10;  # The size of temporal window(frames)
            self.tf_gaussratio = 10;  # temporal frequency to gauss envelope ratio of Gabor bigger number = more waves(larger envelope)
            self.tenv_max = 0.3000;  # the maximum gaussian envelope size(relative to tsize)
            self.zerotf = 1;  # Include 0 Hz(static) energy channels
            self.f_gaussratio = .5;

            # Orientation / direction params
            self.dirdivisions = 8;
            self.local_dc = 1;  # T / F.Include circular gaussians(w / no spat.freq.)
            self.directionSelective = 1;
            # Spatial extent params
            self.sfdivisions = 1;
            self.sfmax = 24;  #
            self.sfmin = 24;  #
            self.f_step_log = 1;  # Applies to both SF and TF?
            self.std_step = 4;  # Governs how closely spaced channels are; a reasonable range is 2.5 - 4
            self.sf_gaussratio = 0.6000;  # 81 channels @ maxsf = 24; 9 x9; 13 x13 @ maxsf = 32
            self.fenv_mode = 0;  # use same env for spatial & temporal gabors
            self.senv_max = 0.3000;
            self.wrap_all = 0;  # whether or not the filters cover the very edge of images
            # Handling phase
            self.phasemode = 0;  # Determines how to do phase(square & sum quadrature pairs(i.e., energy), linear, rectified, etc)
            self.phasemode_sfmax = np.nan;  # Calculate only energy channels(e.g., no linear channels) if sf exceeds this number.
            self.zeromean = 1;

        elif argNum == 4:
            # Same as 2 but No pyramid (high spatial frequency channels only)
            # STRF lab conventions, housekeeping
            self.class_name = 'preprocWavelets_grid';

            self.show_or_preprocess = 1;  # True to preprocess; false to return gabor channels
            self.wrap_all = 0;
            self.verbose = 1;
            self.gaborcachemode = 0;  # whether or not to cache calculated results(could become faster)
            self.valid_w_index = np.nan;  # Select particular gabor channels by number
            # Temporal frequency params
            self.tfdivisions = 3;  # Number of temporal frequencies; [tfmin...tfmax] or [0, tfmin...tfmax] if zerorf = 1
            self.tfmax = 2.66667;  # = 4 hz @ 15 fps([tfmax] cycles per[tsize] frames at 15 fps; 2.66667 / 10 * 15 = 4 Hz)
            self.tfmin = 1.33333;  # = 2 hz @ 15 fps(1.33333 / 10 * 15 = 2 Hz)
            self.tsize = 10;  # The size of temporal window(frames)
            self.tf_gaussratio = 10;  # temporal frequency to gauss envelope ratio of Gabor bigger number = more waves(larger envelope)
            self.tenv_max = 0.3000;  # the maximum gaussian envelope size(relative to tsize)
            self.zerotf = 1;  # Include 0 Hz(static) energy channels

            # Orientation / direction params
            self.dirdivisions = 8;
            self.local_dc = 1;  # T / F.Include circular gaussians(w / no spat.freq.)
            self.directionSelective = 1;
            # Spatial extent params
            self.sfdivisions = 5;
            self.sfmax = 32;  #
            self.sfmin = 32;  #
            self.f_step_log = 1;  # Applies to both SF and TF?
            self.std_step = 3.5;  # Governs how closely spaced channels are; a reasonable range is 2.5 - 4
            self.sf_gaussratio = 0.6000;  # 81 channels @ maxsf = 24; 9 x9; 13 x13 @ maxsf = 32
            self.fenv_mode = 0;  # use same env for spatial & temporal gabors
            self.senv_max = 0.3000;

            # Handling phase
            self.phasemode = 0;  # Determines how to do phase(square & sum quadrature pairs(i.e., energy), linear, rectified, etc)
            self.phasemode_sfmax = np.nan;  # Calculate only energy channels(e.g., no linear channels) if sf exceeds this number.
            self.zeromean = 1;

        elif argNum == 5:

            # same as 1 w/ NO TEMPORAL CHANNESL (static Gabor wavelet model)
            # STRFlab conventions, housekeeping
            self.class_name = 'preprocWavelets_grid';

            self.show_or_preprocess = 1;  # True to preprocess; false to return gabor channels
            self.verbose = 1;
            self.gaborcachemode = 0;  # whether or not to cache calculated results(could become faster)
            self.valid_w_index = np.nan;  # Select particular gabor channels by number
            # Temporal frequency params
            self.tfdivisions = 3;  # Number of temporal frequencies; [tfmin...tfmax] or [0, tfmin...tfmax] if zerorf = 1
            self.tfmax = 0;  # = 4 hz @ 15 fps([tfmax] cycles per[tsize] frames at 15 fps; 2.66667 / 10 * 15 = 4 Hz)
            self.tfmin = 0;  # = 2 hz @ 15 fps(1.33333 / 10 * 15 = 2 Hz)
            self.tsize = 1;  # The size of temporal window(frames)
            self.tf_gaussratio = 1;  # temporal frequency to gauss envelope ratio of Gabor bigger number = more waves(larger envelope)
            self.tenv_max = 0.3000;  # the maximum gaussian envelope size(relative to tsize)
            self.zerotf = 1;  # Include 0 Hz(static) energy channels
            self.f_gaussratio = .5

            # Orientation / direction params
            self.dirdivisions = 8;
            self.local_dc = 1;  # T / F.Include circular gaussians(w / no spat.freq.)
            self.directionSelective = 1;
            # Spatial extent params
            self.sfdivisions = 5;
            self.sfmax = 24;  #
            self.sfmin = 1.5;  #
            self.f_step_log = 1;  # Applies to both SF and TF?
            self.std_step = 4;  # Governs how closely spaced channels are; a reasonable range is 2.5 - 4
            self.sf_gaussratio = 0.6000;  # 81 channels @ maxsf = 24; 9 x9; 13 x13 @ maxsf = 32
            self.fenv_mode = 0;  # use same env for spatial & temporal gabors
            self.senv_max = 0.3000;
            self.wrap_all = 0;  # whether or not the filters cover the very edge of images
            # Handling phase
            self.phasemode = 0;  # Determines how to do phase(square & sum quadrature pairs(i.e., energy), linear, rectified, etc)
            self.phasemode_sfmax = np.nan;  # Calculate only energy channels(e.g., no linear channels) if sf exceeds this number.
            self.zeromean = 1;


        elif argNum == 6:
            # same as 2 w/ NO TEMPORAL CHANNELS (static Gabor wavelet model)
            # (Approximately as in Kay et al 2008, Naselaris et al 2009)
            # STRFlab conventions, housekeeping
            self.class_name = 'preprocWavelets_grid';

            self.show_or_preprocess = 1;  # True to preprocess; false to return gabor channels
            self.wrap_all = 0;
            self.verbose = 1;
            self.gaborcachemode = 0;  # whether or not to cache calculated results(could become faster)
            self.valid_w_index = np.nan;  # Select particular gabor channels by number
            # Temporal frequency params
            self.tfdivisions = 1;  # Number of temporal frequencies; [tfmin...tfmax] or [0, tfmin...tfmax] if zerorf = 1
            self.tfmax = 0;  # = 4 hz @ 15 fps([tfmax] cycles per[tsize] frames at 15 fps; 2.66667 / 10 * 15 = 4 Hz)
            self.tfmin = 0;  # = 2 hz @ 15 fps(1.33333 / 10 * 15 = 2 Hz)
            self.tsize = 1;  # The size of temporal window(frames)
            self.tf_gaussratio = 1;  # temporal frequency to gauss envelope ratio of Gabor bigger number = more waves(larger envelope)
            self.tenv_max = 0.3000;  # the maximum gaussian envelope size(relative to tsize)
            self.zerotf = 1;  # Include 0 Hz(static) energy channels

            # Orientation / direction params
            self.dirdivisions = 8;
            self.local_dc = 1;  # T / F.Include circular gaussians(w / no spat.freq.)
            self.directionSelective = 1;
            # Spatial extent params
            self.sfdivisions = 5;
            self.sfmax = 32;  #
            self.sfmin = 2;  #
            self.f_step_log = 1;  # Applies to both SF and TF?
            self.std_step = 3.5;  # Governs how closely spaced channels are; a reasonable range is 2.5 - 4
            self.sf_gaussratio = 0.6000;  # 81 channels @ maxsf = 24; 9 x9; 13 x13 @ maxsf = 32
            self.fenv_mode = 0;  # use same env for spatial & temporal gabors
            self.senv_max = 0.3000;

            # Handling phase
            self.phasemode = 0;  # Determines how to do phase(square & sum quadrature pairs(i.e., energy), linear, rectified, etc)
            self.phasemode_sfmax = np.nan;  # Calculate only energy channels(e.g., no linear channels) if sf exceeds this number.
            self.zeromean = 1;

        elif argNum == 7:
            # Motion energy model for HCP data
            # STRF conventions housekeeping
            self.class_name = 'preprocWavelets_grid';

            self.show_or_preprocess = 1;  # True to preprocess; false to return gabor channels
            self.wrap_all = 0;
            self.verbose = 1;
            self.gaborcachemode = 0;  # whether or not to cache calculated results(could become faster)
            self.valid_w_index = np.nan;  # Select particular gabor channels by number
            # Temporal frequency params
            self.tfdivisions = 3;  # Number of temporal frequencies; [tfmin...tfmax] or [0, tfmin...tfmax] if zerorf = 1
            self.tfmax = 1.66667;  # = 4 hz @ 24 fps([tfmax] cycles per[tsize] frames at 15 fps; 2.66667 / 10 * 15 = 4 Hz)
            self.tfmin = 0.83333;  # = 2 hz @ 24 fps(1.33333 / 10 * 15 = 2 Hz)
            self.tsize = 10;  # The size of temporal window(frames)
            self.tf_gaussratio = 10;  # temporal frequency to gauss envelope ratio of Gabor bigger number = more waves(larger envelope)
            self.tenv_max = 0.3000;  # the maximum gaussian envelope size(relative to tsize)
            self.zerotf = 1;  # Include 0 Hz(static) energy channels

            # Orientation / direction params
            self.dirdivisions = 8;
            self.local_dc = 1;  # T / F.Include circular gaussians(w / no spat.freq.)
            self.directionSelective = 1;
            # Spatial extent params
            self.sfdivisions = 5;
            self.sfmax = 32;  #
            self.sfmin = 2;  #
            self.f_step_log = 1;  # Applies to both SF and TF?
            self.std_step = 3.5;  # Governs how closely spaced channels are; a reasonable range is 2.5 - 4
            self.sf_gaussratio = 0.6000;  # 81 channels @ maxsf = 24; 9 x9; 13 x13 @ maxsf = 32
            self.fenv_mode = 0;  # use same env for spatial & temporal gabors
            self.senv_max = 0.3000;
            self.wrap_all = 0;  # whether or not the filters cover the very edge of images
            # Handling phase
            self.phasemode = 0;  # Determines how to do phase(square & sum quadrature pairs(i.e., energy), linear, rectified, etc)
            self.phasemode_sfmax = np.nan;  # Calculate only energy channels(e.g., no linear channels) if sf exceeds this number.
            self.zeromean = 1;

        else:
            raise ('Unknown argument!')





