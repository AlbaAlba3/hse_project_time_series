import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import boxcox

class BaseTransform(ABC):
    @abstractmethod
    def fit(self, series_list):
        pass
    
    @abstractmethod
    def transform(self, series_list):
        pass 

    @abstractmethod
    def inverse(self, series_list):
        pass

class NullTransform(BaseTransform):
    def fit(self, series_list):
        return series_list
    
    def transform(self, series_list):
        return series_list 

    def inverse(self, series_list):
        return series_list

class Log1pTransform(BaseTransform):
    def fit(self, series_list):
        self.shifts = []
        result = []
        for s in series_list:
            s = np.asarray(s, dtype=np.float64)
            shift = 0.0
            if s.min() < 0:
                shift = abs(s.min()) + 1e-8
                s = s + shift
            result.append(np.log1p(s))
            self.shifts.append(shift)
        return result

    def transform(self, series_list):
        result = []
        for s, shift in zip(series_list, self.shifts):
            s = np.asarray(s, dtype=np.float64)
            if shift != 0:
                s = s + shift
            result.append(np.log1p(s))
        return result

    def inverse(self, series_list):
        result = []
        for s, shift in zip(series_list, self.shifts):
            x = np.expm1(s)
            if shift != 0:
                x = x - shift
            result.append(x)
        return result

class DiffTransform(BaseTransform):
    def __init__(self, order=1):
        self.order = order
        self.last_values = []

    def fit(self, series_list):
        self.last_values = []
        result = []
        for s in series_list:
            s = np.asarray(s, dtype=np.float64)
            self.last_values.append(s[-self.order:].copy())
            diff = s.copy()
            for _ in range(self.order):
                diff = np.diff(diff)
            result.append(diff)
        return result

    def transform(self, series_list):
        result = []
        for s, last in zip(series_list, self.last_values):
            diff = s.copy()
            for _ in range(self.order):
                diff = np.diff(diff)
            result.append(diff)
        return result

    def inverse(self, series_list):
        result = []
        for s, last in zip(series_list, self.last_values):
            restored = s.copy()
            for i in range(self.order):
                restored = np.r_[last[self.order - 1 - i], restored].cumsum()
            result.append(restored[-len(s):])
        return result

class BoxCoxTransform(BaseTransform):
    def __init__(self):
        self.lambdas = []
        self.shifts = []

    def fit(self, series_list):
        self.lambdas = []
        self.shifts = []
        result = []

        for s in series_list:
            s = np.asarray(s, dtype=np.float64)
            shift = 0.0
            if np.any(s <= 0):
                shift = abs(np.min(s)) + 1e-8
                s = s + shift
            transformed, lam = boxcox(s)
            result.append(transformed)
            self.lambdas.append(lam)
            self.shifts.append(shift)

        return result

    def transform(self, series_list):
        result = []
        for s, lam, shift in zip(series_list, self.lambdas, self.shifts):
            s = np.asarray(s, dtype=np.float64)
            if shift != 0:
                s = s + shift
            if abs(lam) < 1e-8:
                transformed = np.log(s)
            else:
                transformed = (np.power(s, lam) - 1) / lam
            result.append(transformed)
        return result

    def inverse(self, series_list):
        result = []
        for s, lam, shift in zip(series_list, self.lambdas, self.shifts):
            s = np.asarray(s, dtype=np.float64)
            
            s = np.clip(s, -1e8, 1e8)

            if abs(lam) < 1e-8:
                inv = np.exp(s)
            else:
                base = lam * s + 1
                base = np.minimum(np.maximum(base, 1e-12), 1e12)   # защита от переполнения
                inv = np.power(base, 1 / lam)

            if shift != 0:
                inv = inv - shift

            result.append(inv)
        return result
    
class TransformPipeline:
    def __init__(self, transforms):
        self.transforms = transforms

    def fit_transform(self, series_list):
        data = series_list
        for t in self.transforms:
            data = t.fit(data)
        return data

    def transform(self, series_list):
        data = series_list
        for t in self.transforms:
            data = t.transform(data)
        return data

    def inverse(self, series_list):
        data = series_list
        for t in reversed(self.transforms):
            data = t.inverse(data)
        return data