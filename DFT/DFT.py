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

def filterImg(img, filter):
    return img * filter

# naive way with for loop
def getGaussianFilterNaive(H, W, D0):
    ret = np.zeros((H, W), dtype = np.float)
    center_H = H / 2
    Center_W = W / 2
    for h in range(H):
        for w in range(W):
            ret[h, w] = np.exp(-1.0 * (np.power(h - center_H, 2) + np.power(w - Center_W, 2)) / (2 * D0 * D0))
    return ret

def getGaussianFilter(H, W, D0):
    ret = np.zeros((H, W), dtype = np.float)
    center_H = H / 2
    Center_W = W / 2
    r = np.arange(H)
    c = np.arange(W)
    rr, cc = np.meshgrid(r, c, indexing='ij')
    ret = np.exp(-1.0 * (np.power(rr - center_H, 2) + np.power(cc - Center_W, 2)) / (2 * D0 * D0))
    return ret

def filterWithGaussian(img, D0):
    H, W = img.shape
    s_img = shiftImg(img)
    dft_result = FFT2d(s_img)

    gaussian_filter = getGaussianFilter(H, W, D0)
    filter_dft_result = filterImg(dft_result, gaussian_filter)

    idft_result = IFFT2d(filter_dft_result)
    idft_real = idft_result.real
    rebuild_img = shiftImg(idft_real)

    return gaussian_filter, rebuild_img

def filterWithGaussianHigh(img, D0):
    H, W = img.shape
    s_img = shiftImg(img)
    dft_result = FFT2d(s_img)

    gaussian_filter = 1 - getGaussianFilter(H, W, D0)
    filter_dft_result = filterImg(dft_result, gaussian_filter)

    idft_result = IFFT2d(filter_dft_result)
    idft_real = idft_result.real
    rebuild_img = shiftImg(idft_real)

    return gaussian_filter, rebuild_img

# implementation of (I)DFT and (I)FFT
def assignment1():
    h0 = 0
    w0 = 0
    H = 128
    W = 128
    img_path = "Fig0431(d)(blown_ic_crop).tif"
    # img = np.asarray(PIL.Image.open(img_path))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float)
    # img = img[h0:h0+H, w0:w0+W]
    img = cv2.resize(img, (H, W))
    plt.subplot(241),plt.imshow(img,'gray'),plt.title('input'),plt.xticks([]),plt.yticks([])
    H, W = img.shape
    print(H, W)
    # print(img)

    padded_img = img
    # if you need filtering, you should pad the image to avoid ringing.
    padded_img = np.zeros([W * 2, H * 2])
    padded_img[0:H, 0:W] = img

    f=np.fft.fft2(padded_img)
    fshift=np.fft.fftshift(f)
    magnitude_spectrum=20*np.log(np.abs(fshift))
    plt.subplot(245),plt.imshow(magnitude_spectrum,'gray'),plt.title('refer_log_freq'),plt.xticks([]),plt.yticks([])


    padded_H, padded_W = padded_img.shape
    cv2.imwrite("padded_img.png", padded_img)

    shifted_img = shiftImg(padded_img)
    cv2.imwrite("centered.png", prepareShow(shifted_img[0:H,0:W]))

    # dft_result = naiveDFT(shifted_img)
    # dft_result = separateDFT(shifted_img)
    dft_result = FFT2d(shifted_img)

    # log_freq=20*np.log(np.abs(dft_result))
    log_freq = 1 + np.log(np.abs(dft_result))
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


# gaussian low pass filter
def assignment2():
    h0 = 0
    w0 = 0
    H = 128
    W = 128
    img_path = "Fig0441(a)(characters_test_pattern).tif"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float)
    img = cv2.resize(img, (H, W))
    plt.subplot(261),plt.imshow(img,'gray'),plt.title('input'),plt.xticks([]),plt.yticks([])

    s_img = shiftImg(img)
    dft_result = FFT2d(s_img)

    log_freq = 1 + np.log(np.abs(dft_result))
    angle = np.angle(dft_result)
    plt.subplot(267),plt.imshow(log_freq,'gray'),plt.title('log_freq'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussian(img, H/60)
    plt.subplot(262),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/60'),plt.xticks([]),plt.yticks([])
    plt.subplot(268),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/60'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussian(img, H/20)
    plt.subplot(263),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/20'),plt.xticks([]),plt.yticks([])
    plt.subplot(269),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/20'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussian(img, H/10)
    plt.subplot(264),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/10'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,6,10),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/10'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussian(img, H/4)
    plt.subplot(265),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/4'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,6,11),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/4'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussian(img, H/3*2)
    plt.subplot(266),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/3*2'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,6,12),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/3*2'),plt.xticks([]),plt.yticks([])

    plt.show()


# gaussian high pass filter
def assignment3():
    h0 = 0
    w0 = 0
    H = 128
    W = 128
    img_path = "Fig0441(a)(characters_test_pattern).tif"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float)
    img = cv2.resize(img, (H, W))
    plt.subplot(261),plt.imshow(img,'gray'),plt.title('input'),plt.xticks([]),plt.yticks([])

    s_img = shiftImg(img)
    dft_result = FFT2d(s_img)

    log_freq = 1 + np.log(np.abs(dft_result))
    angle = np.angle(dft_result)
    plt.subplot(267),plt.imshow(log_freq,'gray'),plt.title('log_freq'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussianHigh(img, H/60)
    plt.subplot(262),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/60'),plt.xticks([]),plt.yticks([])
    plt.subplot(268),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/60'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussianHigh(img, H/20)
    plt.subplot(263),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/20'),plt.xticks([]),plt.yticks([])
    plt.subplot(269),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/20'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussianHigh(img, H/10)
    plt.subplot(264),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/10'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,6,10),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/10'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussianHigh(img, H/4)
    plt.subplot(265),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/4'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,6,11),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/4'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussianHigh(img, H/3*2)
    plt.subplot(266),plt.imshow(gaussian_filter,'gray'),plt.title('D0=H/3*2'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,6,12),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/3*2'),plt.xticks([]),plt.yticks([])

    plt.show()

def assignment4():
    h0 = 0
    w0 = 0
    H = 256
    W = 256
    threshold = 0
    img_path = "Fig0457(a)(thumb_print).tif"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float)
    img = cv2.resize(img, (H, W))
    plt.subplot(131),plt.imshow(img,'gray'),plt.title('input'),plt.xticks([]),plt.yticks([])

    gaussian_filter, rebuild_img = filterWithGaussianHigh(img, H/4)
    plt.subplot(132),plt.imshow(prepareShow(rebuild_img),'gray'),plt.title('D0=H/4'),plt.xticks([]),plt.yticks([])
    print(np.max(rebuild_img), np.min(rebuild_img))

    binary_img = np.where(rebuild_img > threshold, 255, 0)
    plt.subplot(133),plt.imshow(prepareShow(binary_img),'gray'),plt.title('binary_img'),plt.xticks([]),plt.yticks([])

    plt.show()

assignment2()