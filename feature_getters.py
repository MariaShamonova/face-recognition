import io
from abc import abstractmethod

import cv2
import pylab
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from scipy.fft import dct


class FeatureGetter:

    @abstractmethod
    def plot(self, *args):
        ...

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
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        ax.grid(linewidth=3)
        plt.plot(bins, hist,  linewidth=4)
        path = 'features.png'
        plt.savefig(path)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:
        hist, bins = np.histogram(image, bins=np.linspace(0, 1, self.num_bins))
        return hist


class DFT(FeatureGetter):
    p: int = 13

    def plot(self, image: np.ndarray) -> bytes:
        ftimage = np.fft.fft2(image)
        ftimage = ftimage[0: self.p, 0: self.p]
        ftimage = np.abs(ftimage)

        pylab.imshow(np.abs(ftimage))
        path = 'features.png'
        pylab.savefig(path)

        return path

    def get_feature(self, image: int):

        ftimage = np.fft.fft2(image)

        ftimage = ftimage[0: self.p, 0: self.p]
        return np.abs(ftimage)


class DCT(FeatureGetter):
    p: int = 13

    def plot(self, image: np.ndarray) -> bytes:
        dct_image = dct(image, axis=1)
        dct_image = dct(dct_image, axis=0)
        dct_image = dct_image[0: self.p, 0: self.p]

        pylab.imshow(np.abs(dct_image))
        path = 'features.png'
        pylab.savefig(path)

        return path

    def get_feature(self, image: int):

        c = dct(image, axis=1)
        c = dct(c, axis=0)
        c = c[0: self.p, 0: self.p]

        return c


class Scale(FeatureGetter):
    scale: int = 0.3

    def plot(self, image: np.ndarray) -> bytes:
        h = image.shape[0]
        w = image.shape[1]

        new_size = (int(h * self.scale), int(w * self.scale))

        output = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        min_val, max_val = output.min(), output.max()
        img = 255.0 * (output - min_val) / (max_val - min_val)
        img = img.astype(np.uint8)

        path = 'features.png'
        cv2.imwrite(path, img)

        return path

    def get_feature(self, image: int):
        h = image.shape[0]
        w = image.shape[1]
        new_size = (int(h * self.scale), int(w * self.scale))

        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

class Gradient(FeatureGetter):
    num_bin: int = 0.3

    def plot(self, image: np.ndarray) -> bytes:
        print('not realize')

    def get_feature(self, image: int):
        print('not realize')

