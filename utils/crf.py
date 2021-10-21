import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        #print(np.max(probmap), "probmap")
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        #print(U, "unary_from_softmax")
        U = np.ascontiguousarray(U)
        #print(U, "unary_from_softmax")

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        #print(d, "DenseCRF2D")
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )
        #print(d, "DenseCRF2D")
        #print(np.array(d), "d")
        Q = d.inference(self.iter_max)
        #print(np.array(Q), "Q")
        Q = np.array(Q).reshape((C, H, W))

        return Q