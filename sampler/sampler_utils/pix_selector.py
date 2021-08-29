# When sampling micro samples from a low resolution telescope sample, 
# we want to select from the telescope pixels. Three ways of selesction 
# are implemented: random selection, selection on a line, dense selection covering the area

import abc
import numpy as np
import sys

class PixelSelector():
    def __init__(self, h, w) -> None:
        self.h = h
        self.w = w
        self.coord = []
        self.iterator = None

    def reset_iterator(self):
        self.iterator = iter(self.coord)
        return
        
    @abc.abstractmethod
    def select_pixel(self):
        raise NotImplementedError

class RandSelector(PixelSelector):
    def __init__(self, h, w) -> None:
        super(RandSelector, self).__init__(h, w)

    def select_pixel(self, pad_size = (0, 0)):
        idx_row = np.random.randint(pad_size[0], high=self.h - pad_size[0])
        idx_col = np.random.randint(pad_size[1], high=self.w - pad_size[1])
        return np.array([idx_row, idx_col])

class LineSelector(PixelSelector):
    def __init__(self, h, w, step = None) -> None:
        super(LineSelector, self).__init__(h, w)
        if step is None:
            sys.exit("to sample on a line, step size needs to be specialized")
        
        self.step = step
        v_coord = np.arange(int(step/2.0), w, step=step, dtype = int)
        u_coord = int(h/2-1) * np.ones_like(v_coord)
        self.coord = list(np.transpose(np.stack((u_coord, v_coord))))

    def select_pixel(self):
        return next(self.iterator)


class DenseSelector(PixelSelector):
    def __init__(self, h, w, step = None) -> None:
        super(DenseSelector, self).__init__(h, w)
        if step is None:
            sys.exit("to sample on a line, step size needs to be specialized")
        
        self.step = step
        v_coord = np.arange(int(step/2.0), w, step=step, dtype = int)
        u_coord = np.arange(int(step/2.0), h, step=step, dtype = int)
        u_mesh, v_mesh = np.meshgrid(u_coord, v_coord)

        uv_mesh = np.stack((u_mesh, v_mesh))
        uv_mesh = np.reshape(uv_mesh, (uv_mesh.shape[0], -1))
        self.coord = list(np.transpose(uv_mesh))

    def select_pixel(self):
        return next(self.iterator)