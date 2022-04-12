import io
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from scipy.fft import dct


class FeatureGetter:

    @abstractmethod
    def get_feature(self, *args):
        ...


class Histogram(FeatureGetter):
    num_bins: int = 30

    def plot(self, image: np.ndarray) -> bytes:
        hist, bins = np.histogram(image, bins=np.linspace(0, 1, self.num_bins))
        hist = np.insert(hist, 0, 0.0)
        plt.figure(figsize=(20,10))
        ax = plt.gca()
        plt.xticks( fontsize=30)
        plt.yticks( fontsize=30)
        ax.grid(linewidth=3)
        plt.plot(bins, hist,  linewidth=4)
        path = 'features.png'
        plt.savefig(path)

        return path

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


