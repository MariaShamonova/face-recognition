from abc import abstractmethod

import numpy as np
from scipy.fft import dct


class FeatureGetter:

    @abstractmethod
    def get_feature(self, *args):
        ...


class Histogram(FeatureGetter):
    num_bins: int = 30

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        hist, bins = np.histogram(image, bins=np.linspace(0, 1, self.num_bins))
        return hist


class DFT(FeatureGetter):
    mat_side: int = 13

    def get_feature(self, image: int):

        f = np.fft.fft2(image)

        f = f[0: self.mat_side, 0: self.mat_side]
        return np.abs(f)


class DCT(FeatureGetter):
    mat_side: int = 13

    def get_feature(self, image: int):

        c = dct(image, axis=1)
        c = dct(c, axis=0)
        c = c[0: self.mat_side, 0: self.mat_side]

        return c


