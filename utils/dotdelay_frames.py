import numpy as np


def dotdelay_frames(gs, scw, iin):
    # function out = dotdelay(kernel, in)
    #
    # calculate linear responses of a system kernel to an input
    #
    # INPUT: [kernel] = kernel N x D matrix where N is number of channels and
    #                   D is number of delay lines.
    # [ iin] = input N x S matrix where S is the number of samples
    #
    # OUTPUT:
    # [out] = vector of kernel responses to input
    #

    ktsize = scw.shape[1]
    ktsize2c = int(np.ceil(ktsize/2))
    ktsize2f = int(np.floor(ktsize/2))
    itsize = iin.shape[1]

    gout = np.dot(gs, iin).T
    outs = np.dot(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
    outc = - np.dot(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))

    #the first  harf
    for ii in range(ktsize2c):
        z = np.zeros([ktsize2c-ii -1 ,1])
        outs[:,ii] = np.hstack([outs[ktsize2c -ii - 1:,ii], z.flatten()])
        outc[:,ii] = np.hstack([outc[ktsize2c-ii - 1:, ii], z.flatten()])

    # the second half

    for ii in range(ktsize2f):
        ti = ii + ktsize2c
        z = np.zeros(ii + 1)
        end = len(outs)
        outs[:,ti] = np.hstack([z, outs[:end-ii - 1 ,ti]])
        outc[:,ti] = np.hstack([z, outc[:end-ii - 1, ti]])

    chouts = np.sum(outs,1)
    choutc = np.sum(outc, 1)

    return chouts, choutc


def dotdelay_frames_valid(gs, scw, iin):
    # function out = dotdelay(kernel, in)
    #
    # calculate linear responses of a system kernel to an input
    #
    # INPUT: [kernel] = kernel N x D matrix where N is number of channels and
    #                   D is number of delay lines.
    # [ iin] = input N x S matrix where S is the number of samples
    #
    # OUTPUT:
    # [out] = vector of kernel responses to input
    #

    ktsize = scw.shape[1]
    ktsize2c = int(np.ceil(ktsize/2))
    ktsize2f = int(np.floor(ktsize/2))
    itsize = iin.shape[1]

    gout = np.dot(gs, iin).T
    #outs = np.dot(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
    #outc = - np.dot(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
    # convoluve operation with properties 'valid'
    chouts = np.convolve(gout[:,0], scw[1,:], 'valid') + np.convolve(gout[:,1], scw[0,:], 'valid')
    choutc =  np.convolve( - gout[:,0], scw[0,:], 'valid') + np.convolve(gout[:,1], scw[1,:], 'valid')

    return chouts, choutc


def dotdelay_frames_valid(gs, scw, iin):
    # function out = dotdelay(kernel, in)
    #
    # calculate linear responses of a system kernel to an input
    #
    # INPUT: [kernel] = kernel N x D matrix where N is number of channels and
    #                   D is number of delay lines.
    # [ iin] = input N x S matrix where S is the number of samples
    #
    # OUTPUT:
    # [out] = vector of kernel responses to input
    #

    ktsize = scw.shape[1]
    ktsize2c = int(np.ceil(ktsize/2))
    ktsize2f = int(np.floor(ktsize/2))
    itsize = iin.shape[1]

    gout = np.dot(gs, iin).T
    #outs = np.dot(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
    #outc = - np.dot(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + np.dot(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
    # convoluve operation with properties 'valid'
    chouts = np.convolve(gout[:,0], scw[1,:], 'valid') + np.convolve(gout[:,1], scw[0,:], 'valid')
    choutc =  np.convolve( - gout[:,0], scw[0,:], 'valid') + np.convolve(gout[:,1], scw[1,:], 'valid')

    return chouts, choutc


def dotdelay_frames_pytorch(gs, scw, iin):
    # function out = dotdelay(kernel, in)
    #
    # calculate linear responses of a system kernel to an input
    #
    # INPUT: [kernel] = kernel N x D matrix where N is number of channels and
    #                   D is number of delay lines.
    # [ iin] = input N x S matrix where S is the number of samples
    #
    # OUTPUT:
    # [out] = vector of kernel responses to input
    #
    
    gs = torch.tensor(gs.astype(np.float32))
    scw = torch.tensor(scw.astype(np.float32))
    iin = torch.tensor(iin, requires_grad = True)

    ktsize = scw.shape[1]
    ktsize2c = int(np.ceil(ktsize/2))
    ktsize2f = int(np.floor(ktsize/2))
    itsize = iin.shape[1]
    #print(type(gs))
    
    #print(gs.shape)
    #print(iin.shape)
    gout = torch.t(torch.mm(gs, iin))
    outs = torch.mm(gout[:,0].reshape(-1,1), scw[1,:].reshape(1,-1)) +  torch.mm(gout[:,1].reshape(-1,1), scw[0,:].reshape(1,-1))
    outc = - torch.mm(gout[:,0].reshape(-1,1),scw[0,:].reshape(1,-1)) + torch.mm(gout[:,1].reshape(-1,1), scw[1,:].reshape(1,-1))
    gout.retain_grad()
    outs.retain_grad()
    outc.retain_grad()
    #print(type(outc))
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

    return chouts, choutc