import random
import math
import re
from PIL import Image, ImageOps, ImageEnhance
import PIL
import numpy as np
import torch.nn as nn
import torch

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

_MAX_LEVEL = 8.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)

def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation

def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)

'''----------Data Augmentation Strategy----------'''
#-----1-----
def auto_contrast(img, label, **__):
    return ImageOps.autocontrast(img), label
#-----2-----
def equalize(img, label, **__):
    return ImageOps.equalize(img), label
#-----3-----
def invert(img, label, **__):
    return ImageOps.invert(img), label
#-----4-----
def rotate(img, label, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs), label
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,]
        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f
        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix)
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs), label
    else:
        return img.rotate(degrees, resample=kwargs['resample']), label
#-----5-----
def posterize(img, label, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep), label
#-----6-----
def solarize(img, label, thresh, **__):
    return ImageOps.solarize(img, thresh), label
#-----7-----
def solarize_add(img, label, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut), label
    else:
        return img, label
#-----8-----
def color(img, label, factor, **__):
    return ImageEnhance.Color(img).enhance(factor), label
#-----9-----
def contrast(img, label, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor), label
#-----10-----
def brightness(img, label, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor), label
#-----11-----
def sharpness(img, label, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor), label
#-----12-----
def shear_x(img, label, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs), label
#-----13-----
def shear_y(img, label, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs), label
#-----14-----
def translate_x_abs(img, label, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs), label
#-----15-----
def translate_y_abs(img, label, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs), label
#-----16-----
def translate_x_rel(img, label, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs), label
#-----17-----
def translate_y_rel(img, label, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs), label
#-----18-----
def identity(img, label, **__):
    return img, label
#-----19-----
def cutout(img, label, level, **kwargs):
    w, h = img.size
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - level // 2, 0, h)
    y2 = np.clip(y + level // 2, 0, h)
    x1 = np.clip(x - level // 2, 0, w)
    x2 = np.clip(x + level // 2, 0, w)
    im_array = np.array(img)
    im_array[y1:y2, x1:x2] = 0
    return Image.fromarray(im_array.astype(np.uint8)), label

'''----------Data Augmentation Level----------'''
def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v
#-----1-----
def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate(level)
    # level = -30.0 + (level / _MAX_LEVEL) * 60.0
    return level,
#-----2-----
def _posterize_img_level_to_arg(level, _hparams):
    # range [1, 4], 'keep 1 up to 4 MSB of original image'
    # return int((level / _MAX_LEVEL) * 4),
    return int((level / _MAX_LEVEL) * 3 + 1),
#-----3-----
def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    return int((level / _MAX_LEVEL) * 256),
#-----4-----
def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _MAX_LEVEL) * 110),
#-----5-----
def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _MAX_LEVEL) * 1.8 + 0.1,
#-----6-----
def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    # level = -0.3 + (level / _MAX_LEVEL) * 0.6
    return level,
#-----7-----
def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return level,
#-----8-----
def _translate_rel_level_to_arg(level, _hparams):
    # range [-0.45, 0.45]
    level = (level / _MAX_LEVEL) * 0.45
    level = _randomly_negate(level)
    # level = -0.45 + (level / _MAX_LEVEL) * 0.9
    return level,
#-----9-----
def _cutout_level_to_arg(level, _hparams):
    '''To define value_range'''
    level = 0 + level/_MAX_LEVEL*(_hparams['cutout_const']-0)
    return int(level),

'''----------Data Augmentation Level----------'''
LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Rotate': _rotate_level_to_arg,
    'PosterizeImg': _posterize_img_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
    'Identity': None,
    'Cutout': _cutout_level_to_arg,
}

'''----------Data Augmentation Strategy----------'''
NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert, # Don't use
    'Rotate': rotate,
    'PosterizeImg': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add, # Don't use
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel, # Don't use
    'TranslateYRel': translate_y_rel, # Don't use
    'Identity': identity,
    'Cutout': cutout,
}

class AutoAugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,)
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img, label):
        if 'magnitude_std' in self.hparams:
            magnitude = np.clip(np.random.randn(1) * self.magnitude_std + np.random.randint(self.magnitude + 1), 0, _MAX_LEVEL)
        elif 'magnitude_uni' in self.hparams:
            magnitude = np.clip(np.random.uniform(low=0, high=self.magnitude), 0, _MAX_LEVEL)
        else:
            assert False, 'Unknown magnitude sample method'
        if self.level_fn is None:
            level_args = tuple()
            magnitude = 10
        else:
            level_args = self.level_fn(magnitude, self.hparams)
        aug_img, aug_label = self.aug_fn(img, label, *level_args, **self.kwargs)
        return aug_img, aug_label, magnitude

_RAND_TRANSFORMS4 = [
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',]

_RAND_TRANSFORMS11 = [
    'AutoContrast',
    'Equalize',
    'Rotate',
    'PosterizeImg',
    'Solarize',
    #'Color',
    #'Contrast',
    #'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    'Identity',]

_RAND_TRANSFORMS14 = [
    'AutoContrast',
    'Equalize',
    'Rotate',
    'PosterizeImg',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    'Identity',]

_RAND_TRANSFORMS15 = [
    'AutoContrast',
    'Equalize',
    'Rotate',
    'PosterizeImg',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    'Identity',
    'Invert',]

class RandAugment:
    def __init__(self, ops, num_layers, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        if choice_weights is None:
            self.choice_weights = np.ones((len(self.ops), len(self.ops))) 
            self.choice_weights = self.choice_weights/self.choice_weights.sum()
        else:
            self.choice_weights = choice_weights

    def __call__(self, img, label):
        op_ids = self.rand_sampler()
        ops_code = []
        for num, idx in enumerate(op_ids):
            if num == 0:
                aug_img, aug_label, magnitude = self.ops[idx](img, label)
            else:
                aug_img, aug_label, magnitude = self.ops[idx](aug_img, aug_label)
            ops_code.append(self.op_ids2code(idx, magnitude))
        ops_code_array = np.stack(ops_code, axis=1).astype(np.float32)
        return (aug_img, aug_label), ops_code_array

    def op_ids2code(self, idx, magnitude):
        out = np.zeros(len(self.ops))
        if idx != None:
            out[idx] = magnitude + 1 # Recording the magnitude.
        return out

    def rand_sampler(self):
        pool_size = len(self.ops)**self.num_layers 
        #print(pool_size, '=================', self.choice_weights.shape)
        op_ids_code = np.random.choice(range(pool_size), 1, p=self.choice_weights.reshape(-1)) # Item1: range; Item2:size; Item3: Probability.
        op_ids = np.unravel_index(op_ids_code, [len(self.ops)]*self.num_layers, 'F') # op_ids is a tuple, and this is why we should use it in for.
        op_ids = [x_[0] for x_ in op_ids[::-1]] #We should inverse due to the unravel function.
        return op_ids # The ids [a, b]

def rand_augment_transform(config_str, hparams, aug_choice, choice_weights=None, magnitude=10): 
    # confi="rand-muni0-w0": rand-random aug (For magnitude); mstd/muni ; m/n/w ; 
    magnitude = magnitude  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:] 
    for c in config:
        cs = re.split(r'(\d.*)', c) 
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'muni':
            hparams.setdefault('magnitude_uni', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    if aug_choice == 4:
        ra_ops = [AutoAugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in _RAND_TRANSFORMS4]
    elif aug_choice == 11:
        ra_ops = [AutoAugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in _RAND_TRANSFORMS11]
    elif aug_choice == 14:
        ra_ops = [AutoAugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in _RAND_TRANSFORMS14]
    elif aug_choice == 15:
        ra_ops = [AutoAugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in _RAND_TRANSFORMS15]
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)