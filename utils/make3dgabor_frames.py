import numpy as np

def make3dgabor_frames(xytsize, params):
    # function[gabor, gabor90] = make3dgabor(xytsize, params
    #
    # returns a gabor functions of size X - by - Y - by - T, specified by a vector PARAMS.
    #
    # INPUT:
    # [xytsize] = vector of x, y, and t size, i.e.[64 64 5]
    # [params(1: 2)] = center_x, center_y
    # The spatial center of the Gabor function.The axes are normalized
    # to 0(lower left corner) to 1(upper right corner).
    # e.g., [0.5 0.5] put the Gabor at the center of the matrix.
    # [params(3)] = The direction of the Gabor function in degree(0 - 360).
    # [params(4: 5)] = Spatial frequency and temporal frequency
    # They determine how many cycles in XYTSIZE pixels for each dimension.
    # [params(6: 7)] = Spatial and Temporal envelope size in standard deviation
    # [params(8)] = Phase of the Gabor function(optional, default is 0)
    #
    # OUTPUT:
    # [gabor] = a gabor function of size X - by - Y - by - T, specified by a vector PARAMS.
    # [gabor90] = the quadrature pair Gabor function
    #

    # #

    if np.isreal(params[2]) == False:
        params[2] = np.abs(params[2])
        gabor, gabor90, scweights = make3ddog_frames(xytsize,params)

    
    cx = params[0]
    cy = params[1]
    dir = params[2]
    sf = params[3]
    tf = params[4]
    senv = params[5]
    tenv = params[6]
    if len(params) > 7:
        phase = params[7] * np.pi / 180
    else:
        phase = 0

    if len(params) > 8:
        elong = params[8]
    else:
        elong = 1

    #dx = np.arange(0,1, 1/(xytsize[0]-1))
    #dy = np.arange(0,1, 1/(xytsize[1]-1))
    dx = np.arange(0, 1, 1. / (xytsize[0] ))
    dy = np.arange(0, 1, 1. / (xytsize[1] ))

    if len(xytsize) >= 3 and xytsize[2] >1:
        dt = np.arange(0,1, 1./xytsize[2])
        #dt = dt[:-1]
    else:
        xytsize[2] = 1
        dt = 0.5
    dx = dx.astype(np.float)
    dy = dy.astype(np.float)

    dt = dt.astype(np.float)

    # [iy, ix, it] = ndgrid(dx, dy dt)
    #
    # gauss = np.exp(- ((ix-cx)**2 + (iy-cy0 **2) / (2 * senv**2) - (it- 0.5) **2 / (2*tenv**2))

    fx = -sf * np.cos(float(dir)/180.*np.pi) * 2 * np.pi
    fy = sf * np.sin(float(dir)/180.*np.pi) * 2 * np.pi
    ft = np.real(tf) * 2 * np.pi

    # grat = sin((ix - cx) * fx + (iy - cy) * fy + (it - 0.5) * ft + phase);
    # grat90 = cos((ix - cx) * fx + (iy - cy) * fy + (it - 0.5) * ft + phase);

    # calc a frame and parameters

    [iys, ixs, its] = np.meshgrid(dx,dy,1)

    if elong == 1:
        g_slice = np.exp(- ((ixs -cx) **2 + (iys-cy) **2 )/ (2 * senv**2))
    else:
        g_slice = elonggauss(ixs, iys, cx, cy, senv, elong, float(dir))

    grat_s= np.sin( (ixs - cx) * fx + (iys - cy) * fy + phase)
    grat_90_s = np.cos((ixs - cx) * fx + (iys - cy) * fy + phase)

    gs_slice = g_slice * grat_s
    gc_slice = g_slice * grat_90_s

    env_t = np.exp(- (dt - 0.5) **2. / (2. * tenv **2.))

    if np.imag(tf):
        gs_t = np.sin( (dt -0.5) * ft ) + np.sin( (dt-0.5) * (-ft))
        gc_t = np.cos( (dt - 0.5) * ft) + np.cos( (dt - 0.5 ) * (-ft))
    else:
        gs_t = np.sin( (dt -0.5) * ft)
        gc_t = np.cos( (dt- 0.5) * ft)

    scweights = np.vstack([env_t * gs_t, env_t * gc_t])


    gabor = gs_slice
    gabor90 = gc_slice


    #gabor = gauss * grat
    # gabor90 = gauss.grat90

    # gabr = np.float(gabor)
    # gabor90 = np.float(gabor90)

    return gabor[...,0], gabor90[...,0], scweights#, gtw



def elonggauss(ixs, iys, cx, cy, senv, elong, dir):
    sxy = np.array([1, elong])
    sxy = sxy/np.linalg.norm(sxy)

    sigma_x = sxy[0]
    sigma_y = sxy[1]

    theta = float(dir) * np.pi /180.

    a = np.cos(theta) ** 2. /2. /sigma_x **2. + np.sin(theta) **2. /2. / sigma_y **2
    b = - np.sin(2. * theta)/4./sigma_x **2. + np.sin(2.* theta) / 4. /sigma_y **2
    c = np.sin(theta) **2. /2. /sigma_x**2. + np.cos(theta) **2./2./sigma_y **2

    a = a/(2 * senv **2)
    b = b/(2 * senv **2)
    c = c/(2 * senv **2)

    gslice = np.exp(- (a*(ixs-cs) **2 + 2 * b * (ixs -cx) * (iys -cy) + c * (iys - cy) **2))

    return gslice

