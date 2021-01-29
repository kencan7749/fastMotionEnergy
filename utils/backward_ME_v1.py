import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import h5py
#import cv2
from PIL import Image
from scipy.io import loadmat, savemat
import scipy
import torch
import torch.optim as optim

sys.path.append('../')
#from preprocColorSpace import preprocColorSpace
from preprocWavelets_grid import preprocWavelets_grid_pytorch, getGaborParameters

from make3dgabor_frames import make3dgabor_frames
from dotdelay_frames import dotdelay_frames_pytorch

from preprocWavelets_grid_GetMetaParams import preprocWavelets_gird_GetMetaParams




def calc_back(S, true_feat, params):
    loss_fun = torch.nn.MSELoss(reduction='sum')
    
    # stimulus check
    stimxytsize = S.shape
    # Stimulus aspect ratio; always X/Y
    aspect_ratio = stimxytsize[1]/ stimxytsize[0]

    patchxytsize = list(stimxytsize[:2]) + [params.tsize];
    xypixels = np.prod(patchxytsize[:2]);

    S = S.view(np.int(xypixels), stimxytsize[2])
    S.retain_grad()

    #subtract mean
    if hasattr(params, 'zeromean_value'):
        S = S - params.zeromean_value
        #S = bsxfun( @ minus, S, params.zeromean_value);
    else:
        thismean = torch.mean(S)
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
    
    grad_strage = np.zeros_like(S.detach().numpy())
    loss_list = np.zeros(waveletchannelnum)

    #caluculate loss in each channel
    for iii in range(waveletchannelnum):

        # extract one filter 
        thisgparam = gparams[:,iii].reshape([9,1])
        thesame = 1;

        if any(thisgparam[[0,1,2,3,4,5,8]] != lastgparam[[0,1,2,3,4,5,8]]):
            thesame = 0
        if not thesame:

            if params.gaborcachemode ==2:
                gabors = params.gaborcache[:,:,iii]
                gtw = params.gtwcache[:,:,iii]
            else:
                #obtain parameter
                [gabor0, gabor90, gtw] = make3dgabor_frames(patchxytsize, np.vstack([thisgparam[:7],[[0]]
                                                                                    ,thisgparam[8]]))
                gabors = np.vstack([gabor0.flatten(), gabor90.flatten()])

            if params.gaborcachemode == 1:
                gaborcache[:,:,iii]  = gabors
                gtwcache[:,:,iii] = gtw

            lastgparam = thisgparam

        phaseparam = thisgparam[7]

        if not thesame:
            senv = thisgparam[5]
            if senv < maskenv_below:
                smask = np.where(np.sum(np.abs(gabors), 0) > masklimit)[0]
                
                #dotdelay_frames_pytorch_
                gs, scw, iin = gabors[:,smask], gtw, S[smask,:]
                gs = torch.tensor(gs.astype(np.float32))
                scw = torch.tensor(scw.astype(np.float32))
                iin = torch.tensor(iin, requires_grad = True)

                ktsize = scw.shape[1]
                ktsize2c = int(np.ceil(ktsize/2))
                ktsize2f = int(np.floor(ktsize/2))
                itsize = iin.shape[1]
                #print(type(gs))

                #print(gs.shape)
                #spatial filter
                gout = torch.t(torch.mm(gs, iin))
                #temporal filter 
                outs = torch.mm(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) +  torch.mm(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
                outc = - torch.mm(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + torch.mm(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
                gout.retain_grad()

               
                #the first  harf
                for ii in range(ktsize2c):
                    z = torch.zeros([ktsize2c-ii -1 ,1])
                    outs[:,ii] = torch.cat([outs[ktsize2c -ii - 1:,ii], z.flatten()])
                    outc[:,ii] = torch.cat([outc[ktsize2c-ii - 1:, ii], z.flatten()])

                # the second half

                for ii in range(ktsize2f):
                    ti = ii + ktsize2c
                    z = torch.zeros(ii + 1)
                    end = len(outs)
                    outs[:,ti] = torch.cat([z, outs[:end-ii - 1 ,ti]])
                    outc[:,ti] = torch.cat([z, outc[:end-ii - 1, ti]])

                outs.retain_grad()
                outc.retain_grad()

                chouts = torch.sum(outs,1)
                choutc = torch.sum(outc, 1)
                chouts.retain_grad()
                choutc.retain_grad()

            else:
                #modify
                #[chout0, chout90] = dotdelay_frames_pytorch_(gabors, gtw, S)
                gs, scw, iin = gabors, gtw, S
                
                gs = torch.tensor(gs.astype(np.float32))
                scw = torch.tensor(scw.astype(np.float32))
                iin = torch.tensor(iin, requires_grad = True)

                ktsize = scw.shape[1]
                ktsize2c = int(np.ceil(ktsize/2))
                ktsize2f = int(np.floor(ktsize/2))
                itsize = iin.shape[1]
                #print(type(gs))

                #print(gs.shape)
                #spatial filter
                gout = torch.t(torch.mm(gs, iin))
                #temporal filter 
                outs = torch.mm(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) +  torch.mm(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
                outc = - torch.mm(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + torch.mm(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
                gout.retain_grad()
                
                #the first  harf
                for ii in range(ktsize2c):
                    z = torch.zeros([ktsize2c-ii -1 ,1])
                    outs[:,ii] = torch.cat([outs[ktsize2c -ii - 1:,ii], z.flatten()])
                    outc[:,ii] = torch.cat([outc[ktsize2c-ii - 1:, ii], z.flatten()])

                # the second half

                for ii in range(ktsize2f):
                    ti = ii + ktsize2c
                    z = torch.zeros(ii + 1)
                    end = len(outs)
                    outs[:,ti] = torch.cat([z, outs[:end-ii - 1 ,ti]])
                    outc[:,ti] = torch.cat([z, outc[:end-ii - 1, ti]])



                chouts = torch.sum(outs,1)
                choutc = torch.sum(outc, 1)
                chouts.retain_grad()
                choutc.retain_grad()

        chout0, chout90 = chouts, choutc
        chout0.retain_grad()
        chout90.retain_grad()
        # Squared sum
        chout = torch.sqrt(chout0 **2 + chout90 **2)
        chout.retain_grad()


        loss = loss_fun(chout, torch.tensor(true_feat[:,iii]))
        loss_list[iii] = loss.detach().numpy()

        #loss.backward(retain_graph=True)
        loss.backward()
        #if senv < maskenv_below:
        #    grad_strage[smask,:] += iin.grad.numpy()
        #else:
        #    grad_strage += iin.numpy()
        if senv < maskenv_below:
            grad_strage[smask,:] += iin.grad.numpy() * 1./ waveletchannelnum
        else:
            grad_strage += iin.grad.numpy()* 1./ waveletchannelnum
            
    return np.sum(loss_list), grad_strage


def torch_optim(S, true_feat, params, opt_conf = 'sgd'):
    loss_fun = torch.nn.MSELoss(reduction='sum')
    if S.grad is not None:
        S.grad.data.zero_()
    # 最適化関数
    if opt_conf == "sgd":
        optimizer = optim.SGD({S}, lr=0.1)
    elif opt_conf == "momentum_sgd":
        optimizer = optim.SGD({S}, lr=0.1, momentum=0.9)
    elif opt_conf == "adadelta":
        optimizer = optim.Adadelta({S}, rho=0.95, eps=1e-04)
    elif opt_conf == "adagrad":
        optimizer = optim.Adagrad({S})
    elif opt_conf == "adam":
        optimizer = optim.Adam({S}, lr=1e-1, betas=(0.9, 0.99), eps=1e-09)
    elif opt_conf == "rmsprop":
        optimizer = optim.RMSprop({S})
    optimizer.zero_grad()
    
    # stimulus check
    stimxytsize = S.shape
    # Stimulus aspect ratio; always X/Y
    aspect_ratio = stimxytsize[1]/ stimxytsize[0]

    patchxytsize = list(stimxytsize[:2]) + [params.tsize];
    xypixels = np.prod(patchxytsize[:2]);

    S = S.view(np.int(xypixels), stimxytsize[2])
    S.retain_grad()

    #subtract mean
    if hasattr(params, 'zeromean_value'):
        S = S - params.zeromean_value
        #S = bsxfun( @ minus, S, params.zeromean_value);
    else:
        thismean = torch.mean(S)
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
    
    grad_strage = np.zeros_like(S.detach().numpy())
    loss_list = np.zeros(waveletchannelnum)

    #caluculate loss in each channel
    for iii in range(waveletchannelnum):

        # extract one filter 
        thisgparam = gparams[:,iii].reshape([9,1])
        thesame = 1;

        if any(thisgparam[[0,1,2,3,4,5,8]] != lastgparam[[0,1,2,3,4,5,8]]):
            thesame = 0
        if not thesame:

            if params.gaborcachemode ==2:
                gabors = params.gaborcache[:,:,iii]
                gtw = params.gtwcache[:,:,iii]
            else:
                #obtain parameter
                [gabor0, gabor90, gtw] = make3dgabor_frames(patchxytsize, np.vstack([thisgparam[:7],[[0]]
                                                                                    ,thisgparam[8]]))
                gabors = np.vstack([gabor0.flatten(), gabor90.flatten()])

            if params.gaborcachemode == 1:
                gaborcache[:,:,iii]  = gabors
                gtwcache[:,:,iii] = gtw

            lastgparam = thisgparam

        phaseparam = thisgparam[7]

        if not thesame:
            senv = thisgparam[5]
            if senv < maskenv_below:
                smask = np.where(np.sum(np.abs(gabors), 0) > masklimit)[0]
                
                #dotdelay_frames_pytorch_
                gs, scw, iin = gabors[:,smask], gtw, S[smask,:]
                gs = torch.tensor(gs.astype(np.float32))
                scw = torch.tensor(scw.astype(np.float32))
                iin = torch.tensor(iin, requires_grad = True)

                ktsize = scw.shape[1]
                ktsize2c = int(np.ceil(ktsize/2))
                ktsize2f = int(np.floor(ktsize/2))
                itsize = iin.shape[1]
                #print(type(gs))

                #print(gs.shape)
                #spatial filter
                gout = torch.t(torch.mm(gs, iin))
                #temporal filter 
                outs = torch.mm(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) +  torch.mm(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
                outc = - torch.mm(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + torch.mm(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
                gout.retain_grad()

               
                #the first  harf
                for ii in range(ktsize2c):
                    z = torch.zeros([ktsize2c-ii -1 ,1])
                    outs[:,ii] = torch.cat([outs[ktsize2c -ii - 1:,ii], z.flatten()])
                    outc[:,ii] = torch.cat([outc[ktsize2c-ii - 1:, ii], z.flatten()])

                # the second half

                for ii in range(ktsize2f):
                    ti = ii + ktsize2c
                    z = torch.zeros(ii + 1)
                    end = len(outs)
                    outs[:,ti] = torch.cat([z, outs[:end-ii - 1 ,ti]])
                    outc[:,ti] = torch.cat([z, outc[:end-ii - 1, ti]])

                outs.retain_grad()
                outc.retain_grad()

                chouts = torch.sum(outs,1)
                choutc = torch.sum(outc, 1)
                chouts.retain_grad()
                choutc.retain_grad()

            else:
                #modify
                #[chout0, chout90] = dotdelay_frames_pytorch_(gabors, gtw, S)
                gs, scw, iin = gabors, gtw, S
                
                gs = torch.tensor(gs.astype(np.float32))
                scw = torch.tensor(scw.astype(np.float32))
                iin = torch.tensor(iin, requires_grad = True)

                ktsize = scw.shape[1]
                ktsize2c = int(np.ceil(ktsize/2))
                ktsize2f = int(np.floor(ktsize/2))
                itsize = iin.shape[1]
                #print(type(gs))

                #print(gs.shape)
                #spatial filter
                gout = torch.t(torch.mm(gs, iin))
                #temporal filter 
                outs = torch.mm(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) +  torch.mm(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
                outc = - torch.mm(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + torch.mm(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
                gout.retain_grad()
                
                #the first  harf
                for ii in range(ktsize2c):
                    z = torch.zeros([ktsize2c-ii -1 ,1])
                    outs[:,ii] = torch.cat([outs[ktsize2c -ii - 1:,ii], z.flatten()])
                    outc[:,ii] = torch.cat([outc[ktsize2c-ii - 1:, ii], z.flatten()])

                # the second half

                for ii in range(ktsize2f):
                    ti = ii + ktsize2c
                    z = torch.zeros(ii + 1)
                    end = len(outs)
                    outs[:,ti] = torch.cat([z, outs[:end-ii - 1 ,ti]])
                    outc[:,ti] = torch.cat([z, outc[:end-ii - 1, ti]])



                chouts = torch.sum(outs,1)
                choutc = torch.sum(outc, 1)
                chouts.retain_grad()
                choutc.retain_grad()

        chout0, chout90 = chouts, choutc
        chout0.retain_grad()
        chout90.retain_grad()
        # Squared sum
        chout = torch.sqrt(chout0 **2 + chout90 **2)
        chout.retain_grad()


        #loss = loss_fun(chout, torch.tensor(true_feat[:,iii]))
        loss = loss_fun(chout, torch.from_numpy(true_feat[:,iii]).type(torch.FloatTensor))
        loss_list[iii] = loss.detach().numpy()

        loss.backward(retain_graph=True)
        if senv < maskenv_below:
            grad_strage[smask,:] = iin.grad
        else:
            grad_strage = iin.grad
        S.grad = torch.tensor(grad_strage)
                      
        optimizer.step()
            
    return np.sum(loss_list), grad_strage





def forward(S, params):
    
    # stimulus check
    stimxytsize = S.shape
    # Stimulus aspect ratio; always X/Y
    aspect_ratio = stimxytsize[1]/ stimxytsize[0]

    patchxytsize = list(stimxytsize[:2]) + [params.tsize];
    xypixels = np.prod(patchxytsize[:2]);

    S = S.view(np.int(xypixels), stimxytsize[2])
    S.retain_grad()

    #subtract mean
    if hasattr(params, 'zeromean_value'):
        S = S - params.zeromean_value
        #S = bsxfun( @ minus, S, params.zeromean_value);
    else:
        thismean = torch.mean(S)
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
    
    grad_strage = np.zeros_like(S.detach().numpy())
    loss_list = np.zeros(waveletchannelnum)

    #caluculate loss in each channel
    for iii in range(waveletchannelnum):

        # extract one filter 
        thisgparam = gparams[:,iii].reshape([9,1])
        thesame = 1;

        if any(thisgparam[[0,1,2,3,4,5,8]] != lastgparam[[0,1,2,3,4,5,8]]):
            thesame = 0
        if not thesame:

            if params.gaborcachemode ==2:
                gabors = params.gaborcache[:,:,iii]
                gtw = params.gtwcache[:,:,iii]
            else:
                #obtain parameter
                [gabor0, gabor90, gtw] = make3dgabor_frames(patchxytsize, np.vstack([thisgparam[:7],[[0]]
                                                                                    ,thisgparam[8]]))
                gabors = np.vstack([gabor0.flatten(), gabor90.flatten()])

            if params.gaborcachemode == 1:
                gaborcache[:,:,iii]  = gabors
                gtwcache[:,:,iii] = gtw

            lastgparam = thisgparam

        phaseparam = thisgparam[7]

        if not thesame:
            senv = thisgparam[5]
            if senv < maskenv_below:
                smask = np.where(np.sum(np.abs(gabors), 0) > masklimit)[0]
                
                #dotdelay_frames_pytorch_
                gs, scw, iin = gabors[:,smask], gtw, S[smask,:]
                gs = torch.tensor(gs.astype(np.float32))
                scw = torch.tensor(scw.astype(np.float32))
                iin = torch.tensor(iin, requires_grad = True)

                ktsize = scw.shape[1]
                ktsize2c = int(np.ceil(ktsize/2))
                ktsize2f = int(np.floor(ktsize/2))
                itsize = iin.shape[1]
                #print(type(gs))

                #print(gs.shape)
                #spatial filter
                gout = torch.t(torch.mm(gs, iin))
                #temporal filter 
                outs = torch.mm(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) +  torch.mm(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
                outc = - torch.mm(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + torch.mm(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
                gout.retain_grad()

               
                #the first  harf
                for ii in range(ktsize2c):
                    z = torch.zeros([ktsize2c-ii -1 ,1])
                    outs[:,ii] = torch.cat([outs[ktsize2c -ii - 1:,ii], z.flatten()])
                    outc[:,ii] = torch.cat([outc[ktsize2c-ii - 1:, ii], z.flatten()])

                # the second half

                for ii in range(ktsize2f):
                    ti = ii + ktsize2c
                    z = torch.zeros(ii + 1)
                    end = len(outs)
                    outs[:,ti] = torch.cat([z, outs[:end-ii - 1 ,ti]])
                    outc[:,ti] = torch.cat([z, outc[:end-ii - 1, ti]])

                outs.retain_grad()
                outc.retain_grad()

                chouts = torch.sum(outs,1)
                choutc = torch.sum(outc, 1)
                chouts.retain_grad()
                choutc.retain_grad()

            else:
                #modify
                #[chout0, chout90] = dotdelay_frames_pytorch_(gabors, gtw, S)
                gs, scw, iin = gabors, gtw, S
                
                gs = torch.tensor(gs.astype(np.float32))
                scw = torch.tensor(scw.astype(np.float32))
                iin = torch.tensor(iin, requires_grad = True)

                ktsize = scw.shape[1]
                ktsize2c = int(np.ceil(ktsize/2))
                ktsize2f = int(np.floor(ktsize/2))
                itsize = iin.shape[1]
                #print(type(gs))

                #print(gs.shape)
                #spatial filter
                gout = torch.t(torch.mm(gs, iin))
                #temporal filter 
                outs = torch.mm(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) +  torch.mm(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
                outc = - torch.mm(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + torch.mm(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
                gout.retain_grad()
                
                #the first  harf
                for ii in range(ktsize2c):
                    z = torch.zeros([ktsize2c-ii -1 ,1])
                    outs[:,ii] = torch.cat([outs[ktsize2c -ii - 1:,ii], z.flatten()])
                    outc[:,ii] = torch.cat([outc[ktsize2c-ii - 1:, ii], z.flatten()])

                # the second half

                for ii in range(ktsize2f):
                    ti = ii + ktsize2c
                    z = torch.zeros(ii + 1)
                    end = len(outs)
                    outs[:,ti] = torch.cat([z, outs[:end-ii - 1 ,ti]])
                    outc[:,ti] = torch.cat([z, outc[:end-ii - 1, ti]])



                chouts = torch.sum(outs,1)
                choutc = torch.sum(outc, 1)
                chouts.retain_grad()
                choutc.retain_grad()

        chout0, chout90 = chouts, choutc
        chout0.retain_grad()
        chout90.retain_grad()
        # Squared sum
        chout = torch.sqrt(chout0 **2 + chout90 **2)
        chout.retain_grad()


        Spreproc[:,iii] = chout
    Spreproc.retain_grad()
    return Spreproc

