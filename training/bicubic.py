'''
import sys, time
import torch
import numpy as np
from torch.autograd import Variable


# Interpolation kernel
def u(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0

#Paddnig
def padding(img,B,C,H,W):
    zimg = np.zeros([B,C,H+4,W+4])
    zimg[:,:C,2:H+2,2:W+2] = img
    #Pad the first/last two col and row
    zimg[:,:C,2:H+2,0:2]=img[:,:C,:,0:1]
    zimg[:,:,H+2:H+4,2:W+2]=img[:,:,H-1:H,:]
    zimg[:,:,2:H+2,W+2:W+4]=img[:,:,:,W-1:W]
    zimg[:,:C,0:2,2:W+2]=img[:,:C,0:1,:]
    #Pad the missing eight points
    zimg[0,:C,0:2,0:2]=img[0,:C,0,0]
    zimg[0,:C,H+2:H+4,0:2]=img[0,:C,H-1,0]
    zimg[0,:C,H+2:H+4,W+2:W+4]=img[0,:C,H-1,W-1]
    zimg[0,:C,0:2,W+2:W+4]=img[0,:C,0,W-1]
    return zimg

# Bicubic operation
def bicubic(img):

    a = -0.75

    img = img.cpu().detach().numpy()
    #Get image size
    B, C, H, W = img.shape


    img = padding(img,B,C,H,W)


    #Create new image
    dH = 34
    dW = 34
    dst = np.zeros((B, C, dH, dW))


    h = 1/(34/H)

    for b in range(B):
        for c in range(C):
            for j in range(dH):
                for i in range(dW):
                    x, y = i * h + 2 , j * h + 2

                    x1 = 1 + x - math.floor(x)
                    x2 = x - math.floor(x)
                    x3 = math.floor(x) + 1 - x
                    x4 = math.floor(x) + 2 - x

                    y1 = 1 + y - math.floor(y)
                    y2 = y - math.floor(y)
                    y3 = math.floor(y) + 1 - y
                    y4 = math.floor(y) + 2 - y

                    mat_l = np.matrix([[u(x1,a),u(x2,a),u(x3,a),u(x4,a)]])
                    mat_m = np.matrix([[img[0,c,int(y-y1),int(x-x1)],img[0,c,int(y-y2),int(x-x1)],img[0,c,int(y+y3),int(x-x1)],img[0,c,int(y+y4),int(x-x1)]],
                                       [img[0,c,int(y-y1),int(x-x2)],img[0,c,int(y-y2),int(x-x2)],img[0,c,int(y+y3),int(x-x2)],img[0,c,int(y+y4),int(x-x2)]],
                                       [img[0,c,int(y-y1),int(x+x3)],img[0,c,int(y-y2),int(x+x3)],img[0,c,int(y+y3),int(x+x3)],img[0,c,int(y+y4),int(x+x3)]],
                                       [img[0,c,int(y-y1),int(x+x4)],img[0,c,int(y-y2),int(x+x4)],img[0,c,int(y+y3),int(x+x4)],img[0,c,int(y+y4),int(x+x4)]]])
                    mat_r = np.matrix([[u(y1,a)],[u(y2,a)],[u(y3,a)],[u(y4,a)]])
                    dst[b, c, j, i] = np.asarray( np.dot( np.dot(mat_l, mat_m), mat_r) )

    dst = torch.Tensor(dst).cuda()
    dst = dst.type(torch.cuda.FloatTensor)
    dst = Variable(dst, requires_grad=True).cuda()
    return dst



'''































import math
import torch
import numpy as np
from torch.autograd import Variable
from numba import jit

# Interpolation kernel
@jit
def u(l1, l2, l3, l4,a):
    x1 = abs(l1)
    x2 = abs(l2)
    x3 = abs(l3)
    x4 = abs(l4)

    y1 = 0
    if (x1 >=0) & (x1 <=1):

        y1 = (a+2)*(x1**3)-(a+3)*(x1**2)+1
    elif (x1 > 1) & (x1 <= 2):
        y1 = a*(x1**3)-(5*a)*(x1**2)+(8*a)*x1-4*a


    y2 = 0
    if (x2 >=0) & (x2 <=1):

        y2 = (a+2)*(x2**3)-(a+3)*(x2**2)+1
    elif (x2 > 1) & (x2 <= 2):
        y2 = a*(x2**3)-(5*a)*(x2**2)+(8*a)*x2-4*a


    y3 = 0
    if (x3 >=0) & (x3 <=1):

        y3 = (a+2)*(x3**3)-(a+3)*(x3**2)+1
    elif (x3 > 1) & (x3 <= 2):
        y3 = a*(x3**3)-(5*a)*(x3**2)+(8*a)*x3-4*a


    y4 = 0
    if (x4 >=0) & (x4 <=1):

        y4 = (a+2)*(x4**3)-(a+3)*(x4**2)+1
    elif (x4 > 1) & (x4 <= 2):
        y4 = a*(x4**3)-(5*a)*(x4**2)+(8*a)*x4-4*a


    return y1, y2, y3, y4

@jit
def d(mat):
    mat_l, mat_m, mat_r = mat
    d = np.dot( np.dot(mat_l, mat_m), mat_r)
    return d

@jit
def mat(h,a,i,j,c,img):
    x, y = i * h + 2 , j * h + 2

    fx = math.floor(x)
    x1 = 1 + x -fx
    x2 = x - fx
    x3 = fx + 1 - x
    x4 = fx + 2 - x

    fy = math.floor(y)
    y1 = 1 + y - fy
    y2 = y - fy
    y3 = fy + 1 - y
    y4 = fy + 2 - y

    ny1 = int(y-y1)
    ny2 = int(y-y2)
    ny3 = int(y+y3)
    ny4 = int(y+y4)
    nmx1 = int(x-x1)
    nmx2 = int(x-x2)
    npx3 = int(x+x3)
    npx4 = int(x+x4)

    mat_m = np.array([[img[0,c,ny1,nmx1],img[0,c,ny2,nmx1],img[0,c,ny3,nmx1],img[0,c,ny4,nmx1]],
                       [img[0,c,ny1,nmx2],img[0,c,ny2,nmx2],img[0,c,ny3,nmx2],img[0,c,ny4,nmx2]],
                       [img[0,c,ny1,npx3],img[0,c,ny2,npx3],img[0,c,ny3,npx3],img[0,c,ny4,npx3]],
                       [img[0,c,ny1,npx4],img[0,c,ny2,npx4],img[0,c,ny3,npx4],img[0,c,ny4,npx4]]])

    p1, p2, p3, p4 = u(y1, y2, y3, y4,a)


    return np.array([[u(x1, x2, x3, x4,a)]]),mat_m, np.array([[p1],[p2],[p3],[p4]])


#Paddnig
def padding(img,B,C,H,W):
    zimg = np.zeros([B,C,H+4,W+4])
    zimg[:,:C,2:H+2,2:W+2] = img
    #Pad the first/last two col and row
    zimg[:,:C,2:H+2,0:2]=img[:,:C,:,0:1]
    zimg[:,:,H+2:H+4,2:W+2]=img[:,:,H-1:H,:]
    zimg[:,:,2:H+2,W+2:W+4]=img[:,:,:,W-1:W]
    zimg[:,:C,0:2,2:W+2]=img[:,:C,0:1,:]
    #Pad the missing eight points
    zimg[0,:C,0:2,0:2]=img[0,:C,0,0]
    zimg[0,:C,H+2:H+4,0:2]=img[0,:C,H-1,0]
    zimg[0,:C,H+2:H+4,W+2:W+4]=img[0,:C,H-1,W-1]
    zimg[0,:C,0:2,W+2:W+4]=img[0,:C,0,W-1]
    return zimg

# Bicubic operation
def bicubic(img, newSize, a = -0.75):

    # Coefficient
    img = img.cpu().detach().numpy()


    #Get image size
    B, C, H, W = img.shape

    img = padding(img,B,C,H,W)

    #Create new image
    dH = newSize
    dW = newSize
    dst = np.zeros([B, C, dH, dW])

    h = 1/(newSize/H)

    for b in range(B):
        for c in range(C):
            for j in range(dH):
                for i in range(dW):

                    dst[b, c, j, i] = d(mat(h,a,i,j,c,img))

    dst = torch.Tensor(dst).cuda()
    dst = dst.type(torch.cuda.FloatTensor)
    dst = Variable(dst, requires_grad=True).cuda()
    return dst
