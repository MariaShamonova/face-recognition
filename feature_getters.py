import io
from abc import abstractmethod

import cv2
import pylab
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct


class FeatureGetter:

    @abstractmethod
    def plot(self, *args):
        ...

    @abstractmethod
    def get_feature(self, *args):
        ...

    @abstractmethod
    def get_teach_param(self, *args):
        ...

    @abstractmethod
    def set_param(self, *args):
        ...


class Histogram(FeatureGetter):
    def __init__(self, num_bins: int = 30):
        self.num_bins = num_bins

    def plot(self, image: np.ndarray) -> bytes:

        hist, bins = np.histogram(image, bins=np.linspace(0, 1, self.num_bins))
        hist = np.insert(hist, 0, 0.0)
        plt.figure(figsize=(20,10))
        ax = plt.gca()

        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        ax.grid(linewidth=5)
        plt.rcParams["font.weight"] = 500
        plt.plot(bins, hist,  linewidth=6)
        plt.setp(ax.spines.values(), linewidth=5)
        path = 'results/Histogram.png'
        plt.savefig(path)

        return path

    def get_feature(self, image: np.ndarray) -> np.ndarray:

        hist, bins = np.histogram(image, bins=np.linspace(0, 1, self.num_bins))
        return hist

    def get_teach_param(self, image=None):
        return range(1, 255, 5)

    def set_param(self, num_bins: int):
        self.num_bins = num_bins


class DFT(FeatureGetter):
    def __init__(self, p: int = 13):
        self.p = p

    def plot(self, image: np.ndarray) -> bytes:
        ftimage = np.fft.fft2(image)
        ftimage = ftimage[0: self.p, 0: self.p]
        ftimage = np.abs(ftimage)
        pylab.xticks(fontsize=55)
        pylab.yticks(fontsize=55)
        pylab.rcParams["font.weight"] = 500
        pylab.imshow(np.abs(ftimage))
        path = 'results/DFT.png'
        pylab.savefig(path)

        return path

    def get_feature(self, image: int):
        ftimage = np.fft.fft2(image)
        ftimage = ftimage[0: self.p, 0: self.p]

        return np.abs(ftimage)

    def get_teach_param(self, image=None):
        return range(30)

    def set_param(self, p: int):
        self.p = p


class DCT(FeatureGetter):
    def __init__(self, p: int = 13):
        self.p = p

    def plot(self, image: np.ndarray) -> bytes:
        dct_image = dct(image, axis=1)
        dct_image = dct(dct_image, axis=0)
        dct_image = dct_image[0: self.p, 0: self.p]

        pylab.imshow(np.abs(dct_image))
        path = 'results/DCT.png'
        pylab.savefig(path)

        return path

    def get_feature(self, image: int):

        c = dct(image, axis=1)
        c = dct(c, axis=0)
        c = c[0: self.p, 0: self.p]

        return c

    def get_teach_param(self, image=None):
        return range(30)

    def set_param(self, p: int):
        self.p = p


class Scale(FeatureGetter):
    def __init__(self, scale: int = 0.3):
        self.scale = scale

    def plot(self, image: np.ndarray) -> bytes:
        h = image.shape[0]
        w = image.shape[1]

        new_size = (int(h * self.scale), int(w * self.scale))

        output = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        min_val, max_val = output.min(), output.max()
        img = 255.0 * (output - min_val) / (max_val - min_val)
        img = img.astype(np.uint8)

        path = 'results/Scale.png'
        cv2.imwrite(path, img)

        return path

    def get_feature(self, image: int):
        h = image.shape[0]
        w = image.shape[1]
        new_size = (int(h * self.scale), int(w * self.scale))

        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def get_teach_param(self, image=None):
        return np.arange(0.1, 1.1, 0.1)

    def set_param(self, scale: int):
        self.scale = scale


class Gradient(FeatureGetter):
    def __init__(self, window_width: int = 2):
        self.window_width = window_width

    @staticmethod
    def _calculate_distance(array_1: np.ndarray, array_2: np.ndarray) -> float:
        return np.linalg.norm(np.array(array_1) - np.array(array_2))

    def plot(self, image: np.ndarray) -> bytes:
        height, width = image.shape

        num_steps = int(height / self.window_width)
        gradients = []

        for i in range(num_steps - 2):
            step = i * self.window_width

            start_window = image[step: step + self.window_width]
            end_window = image[step + self.window_width: step + self.window_width * 2]

            gradients.append(self._calculate_distance(start_window, end_window))

        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        plt.xticks(fontsize=55)
        plt.yticks(fontsize=55)
        ax.grid(linewidth=5)
        plt.rcParams["font.weight"] = 500

        plt.setp(ax.spines.values(), linewidth=5)
        plt.plot(range(num_steps - 2), gradients, linewidth=6)
        path = 'results/Gradient.png'
        plt.savefig(path)

        return path

    def get_feature(self, image: int):

        height, width = image.shape

        num_steps = height // self.window_width
        gradients = []

        for i in range(num_steps - 2):
            step = i * self.window_width

            start_window = image[step: step + self.window_width]
            end_window = image[step + self.window_width: step + self.window_width * 2]

            gradients.append(self._calculate_distance(start_window, end_window))

        return gradients

    def get_teach_param(self, image):
        height, width = image.shape

        return range(1,height//2)

    def set_param(self, window_width: int):
        self.window_width = window_width

