"""Foreground color cue for few-shot prototypes; no catalog classifier."""
from __future__ import annotations
import numpy as np
from PIL import Image
from rpv.vision.encoders import BaseEncoder,ClassicalEncoder,l2_normalize


class HueSaturationEncoder(BaseEncoder):
    """Hellinger HS histogram; cached independently of DINO and learned heads."""
    dim = 96
    tag = cache_tag = 'foreground_hs128_h24_s4_hellinger_v1'
    use_tta = False

    def __init__(self):
        self.mask_encoder = ClassicalEncoder(size=128)

    def embed_arrays(self, images):
        rows = []
        for image in images:
            a = np.asarray(Image.fromarray(np.asarray(image,dtype=np.uint8)).convert('RGB').resize((128,128)),dtype=np.float32)/255
            hsv = self.mask_encoder._rgb_to_hsv(a)
            mask = self.mask_encoder._foreground_mask(a)
            hist = np.histogram2d(hsv[:,:,0][mask],hsv[:,:,1][mask],bins=(24,4),range=((0,1),(0,1)))[0].ravel()
            rows.append(np.sqrt(hist/max(hist.sum(),1e-12)).astype(np.float32))
        return l2_normalize(np.asarray(rows,dtype=np.float32).reshape(-1,self.dim))


class PrototypeColorEncoder(BaseEncoder):
    """Fuse a shared visual head with color; bank still uses support prototypes."""
    def __init__(self, base, weight):
        if not np.isfinite(weight) or weight<=0:
            raise ValueError('Prototype color weight must be finite and positive.')
        self.base = base
        self.color = HueSaturationEncoder()
        self.weight = float(weight)
        self.dim = base.dim+self.color.dim
        self.use_tta = base.use_tta
        self.tag = f'{base.tag}_prototype_hs96w{weight:g}_v1'
        self.cache_tag = base.cache_tag

    def __getattr__(self, name):
        return getattr(self.base,name)

    def _cache(self):
        return self.base._cache()

    def _cache_dim(self):
        return self.base._cache_dim()

    def _join(self,z,color):
        return l2_normalize(np.concatenate([z,self.weight*color],axis=1)).astype(np.float32)

    def embed_arrays(self,images):
        return self._join(self.base.embed_arrays(images),self.color.embed_arrays(images))

    def embed_paths(self,paths,use_tta=None,batch_size=None):
        paths = list(paths)
        if use_tta is None:
            use_tta = self.use_tta
        return self._join(self.base.embed_paths(paths,use_tta=use_tta,batch_size=batch_size),
                          self.color.embed_paths(paths,use_tta=use_tta,batch_size=batch_size))

    def describe(self):
        return {**self.base.describe(),'tag':self.tag,'dim':self.dim,
                'catalog_backend':'prototype','color_weight':self.weight,'color_dim':96,
                'color_cache':self.color._cache().info() if self.color._cache() else None}
