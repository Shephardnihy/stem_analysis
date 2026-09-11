from libertem.udf import UDF
import numpy as np
from scipy.ndimage import shift

class PositionAverageSig(UDF):
    def get_result_buffers(self):
        return {
            'sumSig': self.buffer(kind='sig', dtype='float32'),
            'num_frames': self.buffer(kind='single', dtype='int64'),
            'meanSig': self.buffer(kind='sig', dtype='float32', use='result_only'),
        }

    def process_frame(self, frame):
        self.results.sumSig[:] += frame
        self.results.num_frames[:] += 1

    def merge(self, dest, src):
        dest.sumSig[:] += src.sumSig
        dest.num_frames[:] += src.num_frames

    def get_results(self):
        return {
            'meanSig': self.results.sumSig / self.results.num_frames[:],
        }


class CalculateCepstrum(UDF):
    def __init__(self, window = True, pad = True, pad_shape = (1024, 1024), output_shape = (128, 128), *args, **kwargs):
        """
        Cepstrum transform
        parameters
        window : Bool
            If true, a hann window will be applied

        pad : Bool
            If True, the input will be padded

        pad_shape : int
            If pad = True, pad_width specifies the padded frame size

        output_shape : int
            Specify the output shape
        ----------
        """
        super().__init__(*args, window = window, pad = pad, pad_shape = pad_shape, output_shape = output_shape)

    def get_result_buffers(self):
        return {
            "cepstrum": self.buffer(kind = 'nav', dtype = np.int16, extra_shape = self.params.output_shape, where = 'device')
        }

    def process_frame(self, frame):
        output_shape = self.params.output_shape
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]

        frame[frame<1] = 1

        if self.params.window: # Create Windowing
            xx, yy = np.mgrid[0:sx,0:sy]
            w = (1/2-1/2*np.cos(2*np.pi*xx/sx))*(1/2-1/2*np.cos(2*np.pi*yy/sy))


        if self.params.pad and self.params.pad_shape:
            pad_width = ((self.params.pad_shape[0] - sx)//2, (self.params.pad_shape[1] - sy)//2)

        else:
            pad_width = ((0, 0), (0, 0))


        dp_log = np.log(frame)
        dp_pad = np.pad(dp_log*w, pad_width, mode='constant', constant_values=0)

        ceps = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(dp_pad))))
        ceps = ceps[ceps.shape[0]//2 - output_shape[0]//2 : ceps.shape[0]//2 + output_shape[0]//2, ceps.shape[1]//2 - output_shape[1]//2 : ceps.shape[1]//2 + output_shape[1]//2]

        t = np.percentile(ceps,99.99)
        ceps[ceps>t] = t
        ceps = (ceps - ceps.min())/(ceps.max() - ceps.min())*32767


        self.results.cepstrum[:] = ceps.astype("int16")


class GetDescan(UDF):
    """
    Get the diffraction pattern shift by calculating the center-of-mass of the central beam within a specified radius.
    """
    def __init__(self, center, radius, *args, **kwargs):
        super().__init__(*args, center = center, radius = radius, **kwargs)

    def get_result_buffers(self):
        return {
            "DPShift": self.buffer(kind = 'nav', dtype = 'float32', extra_shape = (2,))
        }        

    def process_frame(self, frame):
        r00, c00 = self.meta.dataset_shape[2]//2, self.meta.dataset_shape[3]//2
        r0, c0 = self.params.center
        r = self.params.radius

        rr, cc = np.mgrid[0:self.meta.dataset_shape[2], 0:self.meta.dataset_shape[3]]
        mask = (rr - r0)**2 + (cc - c0)**2 <= r**2

        r_com, c_com = (frame*mask*rr).sum()/ (frame*mask).sum(), (frame*mask*cc).sum()/ (frame*mask).sum()
        shift_r, shift_c = r00 - r_com, c00 - c_com
        self.results.DPShift[:] = np.array([shift_r, shift_c])

class ShiftFrame(UDF):
    """
    Shift the input frame by the specified auxiliary data.

    The auxiliary data should contain the shift values for the frame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_result_buffers(self):
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]
        return {
            "shifted_frame": self.buffer(kind = 'nav', 
                                         dtype = self.meta.dataset_dtype,
                                         extra_shape = (sx, sy))
        }

    def process_frame(self, frame):
        s = self.params.aux_data[:]
        sframe = shift(frame, s).astype(frame.dtype)
        self.results.shifted_frame[:] = sframe[:]

