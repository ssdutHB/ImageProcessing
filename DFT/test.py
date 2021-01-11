import numpy as np 
import time
H = 4
W = 5
a = np.zeros((H, W))
print(a)

rows = np.arange(H).reshape(H, 1)
cols = np.arange(W).reshape(1, W)
rows = np.repeat(rows, W, axis=1)
cols = np.repeat(cols, H, axis=0)
print(rows)
print(cols)

pos = zip(rows, cols)
# print(pos.size)
print(np.meshgrid(a))

r = np.arange(H)
c = np.arange(W)
rr, cc = np.meshgrid(r, c, indexing='ij')
print(rr)
print(cc)


def getGaussianFilter(H, W, D0):
    ret = np.zeros((H, W), dtype = np.float)
    center_H = H / 2
    Center_W = W / 2
    for h in range(H):
        for w in range(W):
            ret[h, w] = np.exp(-1.0 * (np.power(h - center_H, 2) + np.power(w - Center_W, 2)) / (2 * D0 * D0))
    return ret


def getGaussianFilter2(H, W, D0):
    ret = np.zeros((H, W), dtype = np.float)
    center_H = H / 2
    Center_W = W / 2
    r = np.arange(H)
    c = np.arange(W)
    rr, cc = np.meshgrid(r, c, indexing='ij')
    ret = np.exp(-1.0 * (np.power(rr - center_H, 2) + np.power(cc - Center_W, 2)) / (2 * D0 * D0))
    # for h in range(H):
        # for w in range(W):
            # ret[h, w] = np.exp(-1.0 * (np.power(h - center_H, 2) + np.power(w - Center_W, 2)) / (2 * D0 * D0))
    return ret

t0 = time.time()
gf1 = getGaussianFilter(512,512,200)
print(time.time()-t0)
t0 = time.time()
gf2 = getGaussianFilter2(512,512,200)
print(time.time()-t0)
