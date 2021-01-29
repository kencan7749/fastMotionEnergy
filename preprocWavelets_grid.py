import time
import numpy as np
import sys
#sys.path.append('../utils')
sys.path.append('./utils')
from make3dgabor_frames import make3dgabor_frames
from dotdelay_frames import dotdelay_frames


def preprocWavelets_grid(S, params):
    # Usage: [Spreproc, params] = preprocWavelets_grid(S, params);
    #
    # A script for preprocessing of stimuli using a Gabor wavelet basis set
    #
    # INPUT:  [S] = A X - by - Y - by - T matrix containing stimuli(movie);
    # range of images will affect filter output values
    # (0 - 1 vs 0-255), but that difference will go away
    # after normalization of channels( if performed in
    # subsequent preprocessing steps)
    # [params] = structure that contains parameters for
    #  preprocessing, with fields:
    # --- STRFlab housekeeping - --
    #.show_or_preprocess = If this is set to 0, the function returns wavelets
    # of size(S) * number of channels, instead of preprocessed
    # stimuli.This may be used for visualization purpose.
    # If.valid_w_index is also set, this eturns only a subset of
    # wavelets specified by.valid_w_index.(default: 1)
    # --- Misc - --
    #
    # .f_step_log = A flag to specify linear(false) or log(true) step of frequency(default: 0)
    #.valid_w_index = This is used to specify a subset of wavelets to obtain.
    # (See.show_or_preprocess)
    #.verbose = a flag for verbose mode(default: 1)
    #.zeromean = a flag for whether to make stimulus zero mean
    # before preprocessing(default: 0)
    # fenv_mode = a flag to set fenv_mode(no diff btw.spatial and
    # temporal envelopes??) (default: 0)
    # gaborcachemode = (??)(default: 0)
    #
    # --- Orientation / direction parameters - --
    #.dirdivisions = Number of directions for wavelets(default: 8)
    #.directionSelective = a flag for whether the model should be selective
    # for different directions of motion at the same orientation
    #
    # --- Spatial freqeuncy / location parameters - --
    #.sfdivisions = Number of spatial frequencies(default: 5)
    #.sfmax = The maximum spatial frequency / stimulus size at zero velocity(default: 9)
    #.sfmin = The minimum spatial frequency / stimulus size at zero velocity(default: 2)
    #.local_dc = A flag to add localized dc(i.e., 0 spatial freq.) channels(default: 0)
    #.sf_gaussratio = The ratio between the Gaussian window and spatial frequency(default: 0.5)
    #.std_step = Spatial separation of each wavelet in terms of sigma of
    # the Gaussian window(default: 2.5)
    #.senv_max = The maximum spatial envelope(default: 0.3)
    #.wrap_all = Whether to cluster channels tightly at the center
    # of the screen(1) or not (0).(default: 0)
    # --- Temporal freqeuncy parameters - --
    #.tfdivisions = Number of velocities(default: 5)
    #.tsize = Number of frames to calculate wavelets(default: 10)
    #.tfmax = The maximum temporal frequency / stimulus size(default: 3.5)
    #.tf_gaussratio = The ratio between the Gaussian window and temporal frequency(default: 0.4)
    #.tenv_max = The maximum temporal envelope(default: 0.3)
    #
    # --- Phase parameters - --
    #.phasemode = A parameter to specify how to deal with phase information
    # (i.e., how to combine[or not] Gabor channels with different phases)
    # 0: spectral amplitude(default)
    # 1: linear sin and cos phase ampliture(2 x number of wavelets)
    # 2: half - rectified sin and cos phase amplitude(4 x number of wavelets)
    # 3: 0 + 1(3 x number of wavelets)
    # 4: 0 + 2(5 x number of wavelets)
    # 5: phase, atan2(chout90, chout0)
    # 6: 0 + 5(2 x number of wavelets)
    # 7: phase, atan2(chout90, chout0), half - rectified
    # 8: 0 + 7(3 x number of wavelets)
    #.phasemode_sfmax = The maximum spatial frequency to use phaseinformation
    # For higher frequency wavelets over this value, only
    # spectral amplitude information are used.(default: Inf)
    #
    # OUTPUT:
    # [Spreproc] = Preprocessed stimuli that can be used for STRF fitting.
    #              NxD matrix, N = sample size, D = dimensions(model channels)
    # [params] = structure that contains parameters for preprocessing, with additional fields:
    #  .nChan = Number of preprocessed channels
    # (dimensionality of each data vector, AKA: D=dimensions)
    # .gaborparams = A set of parameters for each Gabor wavelet.
    #                This is a p - by - D matrix where p is number of parameters(8)
    #                and D is the number of wavelet channels
    #                Each field in gaborparams represents:
    #                [pos_x pos_y direction s_freq t_freq s_size t_size phasevalue]
    #                phasevalue can be 0 to 6, where
    #                0: spectra
    #                1: linear sin transform
    #                2: linear cos transform
    #                3: half - rectified sin transform(positive values)
    #                4: half - rectified sin transform(negative values)
    #                5: half - rectified cos transform(positive values)
    #                6: half - rectified cos transform(negative values)
    #                7: dPhase / dt
    #                8: dPhase / dt(positive values)
    #                9: dPhase / dt(negative values)
    # .zeromean_value = Offset value of total movie.This is set if.zeromean is non - zero.
    #
    # EXAMPLE:
    # params = preprocWavelets;
    # returns the default set of parameters.
    #
    # [Spreproc, params] = preprocWavelets_grid(S, PARAMS)  returns preprocessed stimuli( or wavelets) and parameters
    #
    # SEE ALSO: make3dgabor, preprocSpectra
    # == == == == == == == == == ==


    # Default parameters
    # STRFlab housekeeping

    default_dict = {'show_or_preprocess': 1,
                    'f_step_log': 0,
                    'fenv_mode': 0,
                    'gaborcachmode': 0,
                    'valid_w_index': np.nan,
                    'zeromean': 1,
                    'verbose': 1,
                     'dirdivisions': 8,
                    'directionSelective': 1,
                    'sfdivisions':5,
                    'sfmax': 9.0,
                    'sfmin': 2.0,
                    'local_dc': 0,
                    'sf_gaussratio': 0.5,
                    'senv_max': 0.3,
                    'std_step': 2.5,
                    'wrap_all': 0,
                    'tfmax': 3.0,
                    'tfmin': 1.333,
                    'tfdivisions': 5,
                    'tsize': 9,
                    'zerotf': 1,
                    'tf_gaussratio': 0.4,
                    'tenv_max': 0.3,
                    'phasemode':0,
                    'phasemode_sfmax': np.nan,
                    'class_name': 'preprocWavelets_grid'
                    }

    for key in default_dict.keys():
        if hasattr(params, key) is False:
            print('add '+ key)
            setattr(params,key,default_dict[key])
    """
    params.show_or_preprocess = 1;
    # Misc
    params.f_step_log = 1;  #params.f_step_log = 0;
    params.fenv_mode = 0; # ??
    params.gaborcachemode = 0;
    params.valid_w_index = np.nan;
    params.zeromean = 1;
    params.verbose = 1;
    # Orientation / direction
    params.dirdivisions = 8;
    params.directionSelective = 1;
    # Spatial frequency / location
    params.sfdivisions = 5;
    params.sfmax = 32; #params.sfmax = 9.0;
    params.sfmin = 2.0;
    params.local_dc = 1; #params.local_dc = 0;
    params.sf_gaussratio = 0.6; #params.sf_gaussratio = 0.5;
    params.senv_max = 0.3;
    params.std_step = 3.5; #params.std_step = 2.5;
    params.wrap_all = 0;
    # Temporal frequency
    params.tfmax = 2.667; #params.tfmax = 3.0;
    params.tfmin = 1.0; #params.tfmin = 1.333;
    params.tfdivisions = 3; #params.tfdivisions = 5;
    params.tsize = 10;  #params.tsize = 9;
    params.zerotf = 1;
    params.tf_gaussratio = 10; #params.tf_gaussratio = 0.4;
    params.tenv_max = 0.3;
    # Phase
    params.phasemode = 0;
    params.phasemode_sfmax = np.nan; # linke to.phasemode
    # Fill in default parameters

    params.class_name = 'preprocWavelets_grid';
    """
    print(params.dirdivisions)
    # Timing
    start_t = time.time()

    # stimulus check
    stimxytsize = S.shape

    # Stimulus aspect ratio; always X/Y
    aspect_ratio = stimxytsize[1]/ stimxytsize[0]

    if len(stimxytsize) == 2:
        stimxytsize = list(stimxytsize) + [1]; # make sure 3 dimensions
        stimxytsize = tuple(stimxytsize)
    elif len(stimxytsize)==4:
        if params.verbose:
            print('Processing color channels separately...')

        #### Should be raise error
        raise('Not appropriate not')
        """
        nColorChannels = S.shape[3];
        # Color stimuli; recursive call to process color channels separately
        Spreproc = [];
        for ci in rangelen(nColorChannels)
            [tstim p] = feval(params.class_name, S(:,:,:,ci), params);
            Spreproc = cat(2,Spreproc, tstim);
        end
        params = p;
        varargout{1} = Spreproc;
        if nargout >1
            varargout{2} = params;
        end
        % Done!
        return
        """

    patchxytsize = list(stimxytsize[:2]) + [params.tsize];
    xypixels = np.prod(patchxytsize[:2]);
    verbose = params.verbose;

    S = S.astype(np.float32)

    S = S.reshape(np.int(xypixels), stimxytsize[2], order='C') # correct to match the matlab ver comparing order 'F'
    #S = S.reshape(np.int(xypixels), stimxytsize[2], order='F')

    if params.show_or_preprocess:
        if params.zeromean:
            if verbose:
                print('[[zero mean stimuli]]\n')
            if hasattr(params, 'zeromean_value'):
                S = S - params.zeromean_value
                #S = bsxfun( @ minus, S, params.zeromean_value);
            else:
                thismean = np.mean(S)  # the diff is 10^-4 comparing to matlab maybe due to efficient number
                #S = bsxfun( @ minus, S, thismean);
                S -= thismean
                params.zeromean_value = thismean;

    # Make a list of gabor parameters
    if hasattr(params,'gaborparams') == False or params.phasemode == 5 or params.phasemode == 7:
        if verbose:
            print('Making a list of gabor parameters... ')
        # Added aspect ratio as necessary influence on Gabor parameters
        [gparams] = getGaborParameters(params,aspect_ratio)
    else:
        gparams = params.gaborparams;

    waveletchannelnum = gparams.shape[1];

    if verbose:
        print('channel num: %d\n', waveletchannelnum);

    if verbose and any([params.valid_w_index == 0]):
        print('Valid channel num: %d\n', len(params.valid_w_index));


    # Set up a matirx to fill in
    if params.show_or_preprocess:
        if verbose:
            print('Preprocessing...')
        Spreproc = np.zeros([stimxytsize[2], waveletchannelnum], dtype=np.float)
    else:
        if verbose:
            print('Making wavelets ...')
        if hasattr(params, 'valie_w_index') == False:
            gnum = int(waveletchannelnum)#len(waveletchannelnum)
        else:
            gnum = len(params.valid_w_index)
        gaborbank = np.zeros(patchxytsize +  [gnum], dtype=np.float)
    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------
    # ignore wavelet pixel for speed - up where:
    masklimit = 0.001 ## pixel value < masklimit
    maskenv_below = 0.1 # spatial envelope < maskenv_below x stimulus size

    if params.gaborcachemode == 1:
        gaborcache = np.zeros([2, np.prod(patchxytsize[:2]), waveletchannelnum], dtype=np.float)
        gtwcache = np.zeros([2, params.tsize, waveletchannelnum], dtype=np.float)

    lastgparam = np.zeros([9,1])
    wcount = 0

    for ii in range(waveletchannelnum):

        if hasattr(params, 'valid_w_index'):
            if ii == params.valid_w_index:
                continue
        thisgparam = gparams[:,ii].reshape([9,1])
        thesame = 1;



        if any(thisgparam[[0,1,2,3,4,5,8]] != lastgparam[[0,1,2,3,4,5,8]]):
            thesame = 0

        if not thesame:

            if params.gaborcachemode ==2:
                gabors = params.gaborcache[:,:,ii]
                gtw = params.gtwcache[:,:,ii]
            else:

                [gabor0, gabor90, gtw] = make3dgabor_frames(patchxytsize, np.vstack([thisgparam[:7],[[0]]
                                                                                    ,thisgparam[8]])) # average diff is 0.006  and the correlation is 0.9993 to matlab
                gabors = np.vstack([gabor0.flatten(), gabor90.flatten()])

            if params.gaborcachemode == 1:
                gaborcache[:,:,ii]  = gabors
                gtwcache[:,:,ii] = gtw

            lastgparam = thisgparam
        phaseparam = thisgparam[7]
        if params.show_or_preprocess:
            if not thesame:
                senv = thisgparam[5]
                if senv < maskenv_below:
                    smask = np.where(np.sum(np.abs(gabors), 0) > masklimit)[0]
                    [chout0, chout90] = dotdelay_frames(gabors[:,smask], gtw, S[smask,:])
                else:
                    [chout0, chout90] = dotdelay_frames(gabors, gtw, S)

            if phaseparam == 0:
                chout = np.sqrt(chout0 **2 + chout90 **2)
                Spreproc[:,ii] = chout
            elif phaseparam == 1:
                chout = chout0
                Spreproc[:,ii] = chout
            elif phaseparam == 2:
                chout = chout90
                Spreproc[:,ii] = chout
            elif phaseparam == 3:
                chout = chout0
                chout[chout < 0] = 0
                Spreproc[:,ii] = chout
            elif phaseparam == 4:
                chout = chout0
                chout[chout>0] = 0
                Sprecproc[:,ii] = -chout
            elif phaseparam == 5:
                chout = chout90
                chout[chout < 0] = 0
                Sprecproc[:, ii] = chout
            elif phaseparam == 6:
                chout = chout90
                chout[chout > 0] = 0
                Sprecproc[:, ii] = - chout
            elif phaseparam == 7:
                chout = np.arctan2(chout90, chout0)
                dtphase = np.arange(np.diff(chout,1,0))
                dtpahse = dtphase + -2 * np.pi*np.sign(dtphase) * np.round(np.abs(dtphase)/ (2*np.pi))
                Spreproc[:,ii] = dtphase
            elif phaseparam == 8:
                chout = np.arctan2(chout90, chout0)
                dtphase = np.arange(np.diff(chout, 1, 0))
                dtphase = dtphase + -2 * np.pi*np.sign(dtphase) * np.round(np.abs(dtphase)/ (2*np.pi))
                dtphase[dtphase<0] = 0
                Spreproc[:,ii] = dypahse
            elif phaseparams == 9:
                chout = np.arctan2(chout90, chout0)
                dtphase = np.arange(np.diff(chout, 1, 0))
                dtphase = dtphase + -2 * np.pi * np.sign(dtphase) * np.round(np.abs(dtphase) / (2 * np.pi))
                dtphase[dtphase > 0] = 0
                Spreproc[:, ii] = -dypahse

        else:

            if phaseparam in [1, 3,4]:
                # reconstruct space-time Gabor
                rgs = np.matrix(gabors[0,:]).T.dot(np.matrix(gtw[1,:])) + np.matrix(gabors[1,:]).T.dot(np.matrix(gtw[0,:]))  ###?????
                rgs = np.array(rgs).reshape(patchxytsize, order = 'C')
                gaborbank[:,:,:,wcount] = rgs
            else:
                # reconstruct space-time Gabor
                rgc = np.matrix(-gabors[0,:]).T.dot(np.matrix(gtw[0,:])) + np.matrix(gabors[1,:]).T.dot(np.matrix(gtw[1,:])) ###?????
                rgc = np.array(rgc).reshape(patchxytsize, order ='C')
                gaborbank[:,:,:,wcount] = rgc
            wcount += 1


    if params.gaborcachemode ==1:
        params.gaborcache = gaborcache
        params.gtwcache = gtwcache
        params.gaborcachemode = 2

    if verbose:
        print('Wavelet preprocessing done in %.1f min (cputime).', (time.time() - start_t)/60)
        if params.show_or_preprocess:
            print('%d channels, %d samples', Spreproc.shape[1], Spreproc.shape[0])
        else:
            print('%d channels',  gaborbank.shape[3])

    if params.show_or_preprocess:
        if params.phasemode in [5,6,7,8]:
            pind = np.find(gparams[7,:] == 7 | gparams[7,:] == 8 | gparams[7,:] == 9)
            print('thresholding phase channels ...')
            for p in range(pind):
                phasech = Spreproc[:, pind[p]]
                if gparms[7, pind[p]-1] == 0: # look for the
                    # corresponding amplitude channel
                    ampch = Spreproc[:,pind[p-1]]
                else:
                    ampch = Spreproc[:,pind[p]-2]
                a_thresh = np.nanstd(ampch) * paramss.a_thresh
                avalind = ampch > a_thresh
                avalind = avalind and np.arange(avalind[:end-1])
                phasech[avalind == 0]  = 0
                Spreproc[:,pind[p]] = phasech

            if params.phasemode in [5,7]:
                Spreproc = Spreproc[:,pind]
                gparams = gparams[:,pind]
                print('Using only dPhase/dt channels: %d', Spreproc.shape[1])

    else:
           Spreproc = gaborbank

    params.gaborparams = gparams
    params.nChan = gparams.shape[1]
    return Spreproc, params
























#
# Making a list of gabor parameters
#

def getGaborParameters(params,aspect_ratio):

    if params.f_step_log:
        sf_array = np.logspace(np.log10(params.sfmin), np.log10(params.sfmax), params.sfdivisions)
        if params.zerotf:
            tf_array = np.logspace(np.log10(params.tfmin), np.log10(params.tfmax), params.tfdivisions-1)
            tf_array = np.array([0] + list(tf_array))

        else:
            tf_array = np.logspace(np.log10(params.tfmin), np.log10(params.tfmax), params.tfdivisions)

    else:
        sf_array = np.linspace(params.sfmin, params.sfmax, params.sfdivisions)
        tf_array = np.linspace(params.tfmin, params.tfmax, params.tfdivisions)

    dir_array = np.arange(0, params.dirdivisions).astype(np.float) / params.dirdivisions * 360.


    if params.phasemode == 0:
        #spectral amplitudes
        pmarray = [0]
    elif params.phasemode == 1:
        # linear sin and cos transform amplitude
        pmarray = [1,2]
    elif params.phasemode == 2:
        # holf rectified sin and cos amplitudes
        pmarray = [3,4,5,6]
    elif params.phasemode == 3:
        # 0 + 1
        pmarray = [0,1,2,]
    elif params.phasemode == 4:
        #0 + 2
        pmarray = [0, 3,4,5,6]
    elif params.phasemode == 5:
        # pahse atan2(sin cos)
        pmarray = [0 ,7]
    elif params.phasemode == 6:
        # 0 + 5
        pmarray = [0,7]
    elif params.phasemode == 7:
        # phase, atan2(sin, cos), harf-rectified
        pmarray = [0,8,9]
    elif params.phasemode == 8:
        pmarray = [0,8,9]

    dirstart = 1
    if params.local_dc:
        dirstart = 0 ## add local dc channel

    waveletcount = 0
    gparams = np.zeros([8, 200000], dtype=np.float) #prepare for some amount of memory for gparams
    # Add a row to gparams to account for aspect ratioo
    gparams = np.vstack([gparams, np.ones([1, gparams.shape[1]],dtype=np.float)])

    for ti in range(params.tfdivisions):
        tf = tf_array[ti]
        for fi in range(params.sfdivisions):
            sf = sf_array[fi]

            if params.fenv_mode:
                f = np.sqrt(sf **2 + tf **2)
                fenv = np.min([params.fenv_max, 1/f*params.f_gaussratio])
                tenv = fenv
                senv = fenv
            else:
                senv = params.senv_max
                if sf != 0:
                    senv = np.min([params.senv_max, 1/sf * params.sf_gaussratio])
                tenv = params.tenv_max

                if tf != 0:
                    tenv = np.min([params.tenv_max, 1/tf*params.tf_gaussratio])

            if params.directionSelective == 0:
                tf = tf + i

            # Account for asymmetrical images
            if aspect_ratio == 1.0:
                # Symmetrical images
                numsps2 = np.floor((1-senv * params.std_step) / (params.std_step* senv) /2)
                numsps2 = np.max([numsps2, 0])

                if numsps2>=1 and params.wrap_all:
                   numsps2 = numsps2 +1
                centers = senv * params.std_step*np.arange(-numsps2, numsps2+1) + 0.5
                cx, cy = np.meshgrid(centers, centers)
            else:
                # aspect ratio is x/y thus ar + x == True xOR y/ar == True y
                # Compute
                print('frame is asymmetirc')
                raise('not implemetn now')

            thisnumdirs = len(dir_array)
            if tf == 0 or params.directionSelective == 0:
                thisnumdirs = int(np.ceil(thisnumdirs/2)) # use only ~180 deg

            if sf == 0:
                thisnumdirs = 1

            for xyi in range(len(cx.flatten())):
                xcenter = cx.flatten()[xyi]
                ycenter = cy.flatten()[xyi]
                for diri in range(dirstart, thisnumdirs+1):
                    if diri:
                        dir = dir_array[diri-1]
                        thissf = sf
                    else:
                        if params.local_dc ==1:
                            dir = 0
                            thissf = 0 # local dc channels
                        else:
                            dir = 0
                            thissf = sf * 0.01 # to avoid the extact same channel

                    if thissf >= params.phasemode_sfmax:

                        thisgparam = [x_center, y_center, dir, thissf, tf,senv, tenv,0,1]
                        if aspect_ratio !=1:
                            thisgparam[8] = max([elong, 1])
                        gparams[:,waveletcount] = thisgparam
                        waveletcount = waveletcount + 1

                    else:
                        for pmod in pmarray:

                            thisgparam = [xcenter, ycenter, dir, thissf, tf, senv, tenv, pmod, 1]
                            if aspect_ratio != 1.0:
                                thisgparam[8] = np.max([elong,1])
                            gparams[:,waveletcount] = thisgparam
                            waveletcount = waveletcount + 1


    gparams = gparams[:,:waveletcount]

    return [gparams]

def ME_model_torch(S, params):
    
    # stimulus check
    stimxytsize = S.shape
    # Stimulus aspect ratio; always X/åY
    aspect_ratio = stimxytsize[1]/ stimxytsize[0]

    patchxytsize = list(stimxytsize[:2]) + [params.tsize];
    xypixels = np.prod(patchxytsize[:2]);
    
    S = S.astype(np.float32)

    S = S.reshape(np.int(xypixels), stimxytsize[2])
    
    #subtract mean
    if hasattr(params, 'zeromean_value'):
        S = S - params.zeromean_value
        #S = bsxfun( @ minus, S, params.zeromean_value);
    else:
        thismean = np.mean(S)
        #S = bsxfun( @ minus, S, thismean);
        S -= thismean
        params.zeromean_value = thismean;
    
    
    [gparams] = getGaborParameters(params,aspect_ratio)
    
    waveletchannelnum = gparams.shape[1]
    
    
    
    Spreproc = torch.zeros([stimxytsize[2], waveletchannelnum])
    
    # ignore wavelet pixel for speed - up where:
    masklimit = 0.001 ## pixel value < masklimit
    maskenv_below = 0.1 # spatial envelope < maskenv_below x stimulus size
    
    lastgparam = np.zeros([9,1])
    wcount = 0
    
    for ii in range(waveletchannelnum):

        # extract one filter 
        thisgparam = gparams[:,ii].reshape([9,1])
        thesame = 1;

        if any(thisgparam[[0,1,2,3,4,5,8]] != lastgparam[[0,1,2,3,4,5,8]]):
            
            thesame = 0

        if not thesame:

            if params.gaborcachemode ==2:
                gabors = params.gaborcache[:,:,ii]
                gtw = params.gtwcache[:,:,ii]
            else:

                [gabor0, gabor90, gtw] = make3dgabor_frames(patchxytsize, np.vstack([thisgparam[:7],[[0]]
                                                                                    ,thisgparam[8]]))
                gabors = np.vstack([gabor0.flatten(), gabor90.flatten()])

            if params.gaborcachemode == 1:
                gaborcache[:,:,ii]  = gabors
                gtwcache[:,:,ii] = gtw

            lastgparam = thisgparam
            
        phaseparam = thisgparam[7]
        
        if not thesame:
            senv = thisgparam[5]
            if senv < maskenv_below:
                smask = np.where(np.sum(np.abs(gabors), 0) > masklimit)[0]
                #modify
                [chout0, chout90] = dotdelay_frames_pytorch(gabors[:,smask], gtw, S[smask,:])
            else:
                #modify
                [chout0, chout90] = dotdelay_frames_pytorch(gabors, gtw, S)

        # Squared sum
        
        chout = torch.sqrt(chout0 **2 + chout90 **2)
        Spreproc[:,ii] = chout
        
        
        
    return Spreproc



