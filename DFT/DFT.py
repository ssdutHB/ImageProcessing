import cv2
import numpy as np 
import matplotlib.pyplot as plt
import cmath
import PIL 

# restrict the pixel value into [0, 255]
def prepareShow(img):
    min_num = img.min()
    max_num = img.max()
    img_ret = ((img - min_num) / (max_num - min_num)) * 255
    return img_ret

# shift the image so that the DC part stays at the center of frequenct spectrum 
def shiftImg(img):
    H, W = img.shape
    ret = np.zeros(img.shape, img.dtype)
    for h in range(H):
        for w in range(W):
            ret[h,w] = img[h,w] * np.power(-1, h+w) 
    return ret

# Naive implementation of DFT
def naiveDFT(img):
    H, W = img.shape
    dft_result = np.zeros((H, W), dtype = np.complex)
    for n in range(H):
        for m in range(W):
            it = 0.0
            for h in range(H):
                for w in range(W):
                    # attention! In python2.7, the n * h / H is integer
                    # while in python 3.6, n * h /H is float 
                    it =  it + img[h, w] * cmath.exp(-2j * np.pi *(1.0 * n * h / H + 1.0 * m * w / W))
            dft_result[n, m] = it
    return dft_result

def naiveIDFT(img):
    H, W = img.shape
    idft_result = np.zeros((H, W), dtype= np.complex)
    for h in range(H):
        for w in range(W):
            it = 0.0
            for n in range(H):
                for m in range(W):
                    it = it + dft_result[n,m] * cmath.exp(2j * np.pi * (1.0 * h * n/H + 1.0 * w * m/W))
            idft_result[h,w] = 1.0 / (H * W) * it
    return idft_result

# 1D DFT
def DFT1d(signal):
    L = signal.shape[0]
    # print(L)
    ret = np.zeros(signal.shape, dtype = np.complex)
    for n in range(L):
        t = 0.0
        for m in range(L):
            t = t + signal[m] * np.exp(-2j * np.pi * 1.0 * n * m / L)
        ret[n] = t
    return ret            

def IDFT1d(signal):
    L = signal.shape[0]
    # print(L)
    ret = np.zeros(signal.shape, dtype = np.complex)
    for n in range(L):
        t = 0.0
        for m in range(L):
            t = t + signal[m] * np.exp(2j * np.pi * 1.0 * n * m / L)
        ret[n] = 1.0 / L * t
    return ret   

# Line first and column second to accelerate DFT
def separateDFT(img):
    H, W = img.shape
    dft_result = np.zeros((H, W), dtype = np.complex)
    for n in range(H):
        curr_line = img[n,:]
        dft_result[n,:] = DFT1d(curr_line)
    for m in range(W):
        curr_col = dft_result[:, m]
        dft_result[:, m] = DFT1d(curr_col)
    return dft_result

def separateIDFT(img):
    H, W = img.shape
    dft_result = np.zeros((H, W), dtype = np.complex)
    for n in range(H):
        curr_line = img[n,:]
        dft_result[n,:] = IDFT1d(curr_line)
    for m in range(W):
        curr_col = dft_result[:, m]
        dft_result[:, m] = IDFT1d(curr_col)
    return dft_result

# We can realize IDFT with the help of DFT and some extra operations
def reusedIDFT(img):
    H, W = img.shape
    conj_img = img.conjugate()
    ret = separateDFT(conj_img)
    ret = ret / (1.0 * H * W)
    ret = ret.conjugate()
    return ret

# 1D FFT
def FFT1d(signal):
    L = signal.shape[0]
    if L == 1:
        return signal
    ret = np.zeros(signal.shape, dtype = np.complex)
    half_signal = signal[0:L/2]
    even_part = FFT1d(signal[::2])
    odd_part = FFT1d(signal[1::2])

    for i in range(L):
        if i < L/2:
            ret[i] = even_part[i] + odd_part[i] * np.exp(-2j * np.pi * 1.0 * i / L)
        else:
            ret[i] = even_part[i-L/2] - odd_part[i-L/2] * np.exp(-2j * np.pi * 1.0 * (i - L/2) / L)
    return ret

def FFT2d(img):
    H, W = img.shape
    fft_result = np.zeros((H, W), dtype = np.complex)
    for n in range(H):
        curr_line = img[n,:]
        fft_result[n,:] = FFT1d(curr_line)
    for m in range(W):
        curr_col = fft_result[:, m]
        fft_result[:, m] = FFT1d(curr_col)
    return fft_result

def IFFT2d(img):
    H, W = img.shape
    conj_img = img.conjugate()
    ret = FFT2d(conj_img)
    ret = ret / (1.0 * H * W)
    ret = ret.conjugate()
    return ret


h0 = 200
w0 = 100
H = 128
W = 128
img_path = "ori_110.png"
# img = np.asarray(PIL.Image.open(img_path))
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype(np.float)
img = img[h0:h0+H, w0:w0+W]
plt.subplot(241),plt.imshow(img,'gray'),plt.title('input'),plt.xticks([]),plt.yticks([])
H, W = img.shape
# print(img)


f=np.fft.fft2(img)
fshift=np.fft.fftshift(f)
magnitude_spectrum=20*np.log(np.abs(fshift))
plt.subplot(245),plt.imshow(magnitude_spectrum,'gray'),plt.title('refer_log_freq'),plt.xticks([]),plt.yticks([])

padded_img = img
# if you need filtering, you should pad the image to avoid ringing.
# padded_img = np.zeros([W * 2, H * 2])
# padded_img[0:H, 0:W] = img

padded_H, padded_W = padded_img.shape
cv2.imwrite("padded_img.png", padded_img)

shifted_img = shiftImg(padded_img)
cv2.imwrite("centered.png", prepareShow(shifted_img[0:H,0:W]))

# dft_result = naiveDFT(shifted_img)
# dft_result = separateDFT(shifted_img)
dft_result = FFT2d(shifted_img)

log_freq=20*np.log(np.abs(dft_result))
angle = np.angle(dft_result)
plt.subplot(242),plt.imshow(angle,'gray'),plt.title('angle'),plt.xticks([]),plt.yticks([])
cv2.imwrite("log_freq.png", prepareShow(log_freq))
plt.subplot(243),plt.imshow(log_freq,'gray'),plt.title('log_freq'),plt.xticks([]),plt.yticks([])

# idft_result = naiveIDFT(dft_result)
# idft_result = separateIDFT(dft_result)
# idft_result = reusedIDFT(dft_result)
idft_result = IFFT2d(dft_result)

idft_real = idft_result.real
rebuild_img = shiftImg(idft_real)
cv2.imwrite("rebuild_img.png", prepareShow(rebuild_img))
plt.subplot(244),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('rebuild_img'),plt.xticks([]),plt.yticks([])

plt.show()






