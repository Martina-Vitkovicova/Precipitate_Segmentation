import numpy as np

def _norm(arr):
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_min == a_max:
        return arr
    else:
        return (arr - a_min)/(a_max - a_min)

def pink_noise(size, rand, roughness):
    """ Pink noise surface in matplotlib
        Written by PM 2Ring 2023.10.20
    """
    white = rand(size=(size, size))
    white_ft = np.fft.fftshift(np.fft.fft2(white))

    hsize = size / 2
    y, x = np.ogrid[-hsize:hsize, -hsize:hsize]
    freq = (x**2 + y**2) ** (1 / roughness)
    pink_ft = np.divide(white_ft, freq,
      out=np.zeros_like(white_ft), where=freq!=0)

    pink = np.fft.ifft2(np.fft.ifftshift(pink_ft)).real
    return _norm(pink)