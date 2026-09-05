from libertem.udf import UDF

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
    def get_backends(self):
        return self.BACKEND_ALL

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
            "cepstrum": self.buffer(kind = 'nav', dtype = self.xp.int16, extra_shape = self.params.output_shape, where = 'device')
        }

    def process_frame(self, frame):
        output_shape = self.params.output_shape
        sx, sy = self.meta.dataset_shape[2], self.meta.dataset_shape[3]

        frame[frame<1] = 1

        if self.params.window: # Create Windowing
            xx, yy = self.xp.mgrid[0:sx,0:sy]
            w = (1/2-1/2*self.xp.cos(2*self.xp.pi*xx/sx))*(1/2-1/2*self.xp.cos(2*self.xp.pi*yy/sy))


        if self.params.pad and self.params.pad_shape:
            pad_width = ((self.params.pad_shape[0] - sx)//2, (self.params.pad_shape[1] - sy)//2)

        else:
            pad_width = ((0, 0), (0, 0))


        dp_log = self.xp.log(frame)
        dp_pad = self.xp.pad(dp_log*w, pad_width, mode='constant', constant_values=0)

        ceps = self.xp.abs(self.xp.fft.fftshift(self.xp.fft.fft2(self.xp.fft.ifftshift(dp_pad))))
        ceps = ceps[ceps.shape[0]//2 - output_shape[0]//2 : ceps.shape[0]//2 + output_shape[0]//2, ceps.shape[1]//2 - output_shape[1]//2 : ceps.shape[1]//2 + output_shape[1]//2]

        t = self.xp.percentile(ceps,99.99)
        ceps[ceps>t] = t
        ceps = (ceps - ceps.min())/(ceps.max() - ceps.min())*32767


        self.results.cepstrum[:] = ceps.astype("int16")
