from cv2 import imdecode, cvtColor, COLOR_BGR2RGB, resize, INTER_CUBIC, imread,imwrite, COLOR_RGB2BGR,COLOR_RGBA2RGB
from skimage import color
import numpy as np

class L0smoothing():

    def __init__(self,load_size=256, win_size=512):
        self.win_size = win_size
        self.load_size = load_size
        self.LoSmooth = 0

    def read_image(self, image_file):
        im_bgr = imdecode(np.fromfile(image_file, dtype=np.uint8), -1)
        self.im_full = im_bgr.copy()
        self.im_rgb = cvtColor(im_bgr, COLOR_BGR2RGB)
        h, w, c = im_bgr.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)
        self.win_w = rw
        self.win_h = rh
        im_bgr = resize(im_bgr, (self.load_size, self.load_size), interpolation=INTER_CUBIC)
        self.im_rgb = cvtColor(im_bgr, COLOR_BGR2RGB)

    def save_result(self,img_url):
        result_rgb = cvtColor(self.result, COLOR_RGB2BGR)
        imwrite(img_url, result_rgb);

    def L0Smoothing(self):
        self.LoSmooth = not self.LoSmooth
        self.compute_result()

    def compute_result(self):
        pred_rgb = resize(self.im_rgb, (self.win_w, self.win_h), interpolation=INTER_CUBIC)
            # pred_rgb = self.im_rgb
        if self.LoSmooth:
            pred_rgb = LoSmooth_DealWith(pred_rgb)
            pred_rgb = np.clip(pred_rgb, 0, 255).astype('uint8')
        self.result = pred_rgb

def LoSmooth_DealWith(image,verbose=False):
    kappa = 2.0
    lambda__ = 2e-2
    N, M, D = np.int32(image.shape)
    assert D == 3, "Error: input must be 3-channel RGB image"
    # Initialize S as I
    S = np.float32(image) / 256

    # Compute image OTF
    size_2D = [N, M]
    fx = np.int32([[1, -1]])
    fy = np.int32([[1], [-1]])
    otfFx = psf2otf(fx, size_2D)
    otfFy = psf2otf(fy, size_2D)

    # Compute F(I)
    FI = np.complex64(np.zeros((N, M, D)))
    FI[:, :, 0] = np.fft.fft2(S[:, :, 0])
    FI[:, :, 1] = np.fft.fft2(S[:, :, 1])
    FI[:, :, 2] = np.fft.fft2(S[:, :, 2])

    # Compute MTF
    MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
    MTF = np.tile(MTF[:, :, np.newaxis], (1, 1, D))

    # Initialize buffers
    h = np.float32(np.zeros((N, M, D)))
    v = np.float32(np.zeros((N, M, D)))
    dxhp = np.float32(np.zeros((N, M, D)))
    dyvp = np.float32(np.zeros((N, M, D)))
    FS = np.complex64(np.zeros((N, M, D)))

    # Iteration settings
    beta_max = 1e5
    beta = 2 * lambda__
    iteration = 0

    # Iterate until desired convergence in similarity
    while beta < beta_max:

        if verbose:
            print("ITERATION %i" % iteration)
        ### Step 1: estimate (h, v) subproblem
        # compute dxSp
        h[:, 0:M - 1, :] = np.diff(S, 1, 1)
        h[:, M - 1:M, :] = S[:, 0:1, :] - S[:, M - 1:M, :]

        # compute dySp
        v[0:N - 1, :, :] = np.diff(S, 1, 0)
        v[N - 1:N, :, :] = S[0:1, :, :] - S[N - 1:N, :, :]

        # compute minimum energy E = dxSp^2 + dySp^2 <= lambda__/beta
        t = np.sum(np.power(h, 2) + np.power(v, 2), axis=2) < lambda__ / beta
        t = np.tile(t[:, :, np.newaxis], (1, 1, 3))

        # compute piecewise solution for hp, vp
        h[t] = 0
        v[t] = 0

        # compute dxhp + dyvp
        dxhp[:, 0:1, :] = h[:, M - 1:M, :] - h[:, 0:1, :]
        dxhp[:, 1:M, :] = -(np.diff(h, 1, 1))
        dyvp[0:1, :, :] = v[N - 1:N, :, :] - v[0:1, :, :]
        dyvp[1:N, :, :] = -(np.diff(v, 1, 0))
        normin = dxhp + dyvp

        FS[:, :, 0] = np.fft.fft2(normin[:, :, 0])
        FS[:, :, 1] = np.fft.fft2(normin[:, :, 1])
        FS[:, :, 2] = np.fft.fft2(normin[:, :, 2])

        # solve for S + 1 in Fourier domain
        denorm = 1 + beta * MTF;
        FS[:, :, :] = (FI + beta * FS) / denorm

        # inverse FFT to compute S + 1
        S[:, :, 0] = np.float32((np.fft.ifft2(FS[:, :, 0])).real)
        S[:, :, 1] = np.float32((np.fft.ifft2(FS[:, :, 1])).real)
        S[:, :, 2] = np.float32((np.fft.ifft2(FS[:, :, 2])).real)

        # update beta for next iteration
        beta *= kappa
        iteration += 1

    # Rescale image
    S = (S * 256)  # type: float
    return S

def prepare_psf(psf, outSize=None, dtype=None):
    if not dtype:
        dtype = np.float32
    psf = np.float32(psf)
    psfSize = np.int32(psf.shape)
    if not outSize:
        outSize = psfSize
    outSize = np.int32(outSize)
    new_psf = np.zeros(outSize, dtype=dtype)
    new_psf[:psfSize[0], :psfSize[1]] = psf[:, :]
    psf = new_psf
    shift = -(psfSize / 2)
    for i in range(shift.size):
        psf = np.roll(psf, int(shift[i]), axis=i)
    return psf

def psf2otf(psf, outSize=None):
    data = prepare_psf(psf, outSize)
    otf = np.fft.fftn(data)
    return np.complex64(otf)







