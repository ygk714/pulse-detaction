def band_passing(hf, lf, vector):
    from scipy.fftpack import fft
    y = fft(vector)
    j = 0
    for i in range(len(y)):
        if hf<i or i<lf:
            y[i] = 0
        if i!=0:
            if y[i]>y[j]:
                j = i
    print(j)
    return y

# def smooving(sigy, sigx, mat):
#     import scipy as sp
#     sigma = [sigy, sigx]
#     numat = sp.ndimage.filters.gaussian_filter(mat, sigma, mode='constant')
#     return numat
#
# def smooving_dfive(sigx, sigy, mat):
#     import scipy as sp
#     import numpy as np
#     mx = np.zeroes(np.size(mat))
#     for i in range(5):
#         for j in range(5):
#             if i == 0 or i == 4:
#                 mx[i,j]= 0.25-0.5*sigx
#             elif i == 1 or i == 3:
#                 mx[i, j] = 0.25
#             elif i == 2:
#                 mx[i,j] = sigx
#             if j == 0 or j == 4:
#                 mx[i, j] = (0.25 - 0.5 * sigy)*mx[i, j]
#             elif j == 1 or j == 3:
#                 mx[i, j] = 0.25*mx[i, j]
#             elif j == 2:
#                 mx[i, j] = sigy*mx[i, j]
#     numat = sp.signal.convolve2d(mat, mx)
#     return numat
#
# def smooving_onfive(sigx, sigy, mat):
#     import scipy as sp
#     import numpy as np
#     mx = np.zeroes(np.size(mat))
#     for i in range(5):
#         for j in range(5):
#             mx[i,j]= (0.25-sigx/2)*(0.25-sigy/2)*(mat[i+2,j+2]+mat[i+2,j-2]+mat[i-2,j+2]+mat[i-2,j-2])
#             mx[i,j]= mx[i,j]+(0.25-sigx/2)*0.25*(mat[i+2,j+1]+mat[i+2,j-1]+mat[i-2,j+1]+mat[i-2,j-1])
#             mx[i, j] = mx[i, j] +(0.25-sigy/2)*(0.25)*(mat[i+1,j+2]+mat[i+1,j-2]+mat[i-1,j+2]+mat[i-1,j-2])
#             mx[i, j] = mx[i, j] +(0.25-sigx/2)*sigy*(mat[i+2,j]+mat[i-2,j])
#             mx[i, j] = mx[i, j] + (0.25 - sigy / 2) * sigx * (mat[i , j+2] + mat[i , j-2])
#             mx[i, j] = mx[i, j] + 0.0625*(mat[i+1,j+1]+mat[i+1,j-1]+mat[i-1,j+1]+mat[i-1,j-1])
#             mx[i, j] = mx[i, j] + 0.25 * sigy * (mat[i + 1, j] + mat[i - 1, j])mx[i, j] = mx[i, j] + 0.25 * sigx * (mat[i, j + 1] + mat[i, j - 1])
#             mx[i, j] = mx[i, j] + 0.25 * sigx * (mat[i, j + 1] + mat[i, j - 1])
#             mx[i, j] = mx[i, j] + sigy * sigx * (mat[i, j])
#     numat = sp.signal.convolve2d(mat, mx)
#     return numat
#
#
#
#
#
#
#
#
