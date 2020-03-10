import numpy as np 


from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
WIDTH, HEIGHT = 236, 137


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((WIDTH, HEIGHT),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((WIDTH, HEIGHT),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), min(WIDTH, HEIGHT) / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((WIDTH, HEIGHT),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), min(WIDTH, HEIGHT) / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((WIDTH, HEIGHT),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


class AugMix:
    """Perform AugMix augmentations and compute mixture
    Args:
        severity: Severity of underlying augmentation operators (between 1 and 10).
        width: width of augmentation chain
        depth: depth of augmentation chain. -1 enables stochastic depth uniformly from [1, 3]
        alpha: probability coefficient for Beta and Dirichlet distributions
    Targets:
        Image
    """
    def __init__(self, mean, std, augmentations = None, severity = 1, width = 3, depth = 1, alpha = 1.):
        self.mean = mean
        self.std = std
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = augmentations
        if not self.augmentations:
            self.augmentations = [
                autocontrast, equalize, posterize, rotate, solarize,
                shear_x, shear_y, translate_x, translate_y
            ]
        self.ws = np.random.dirichlet([alpha] * width).astype(np.float32)
        self.m  = np.float32(np.random.beta(alpha, alpha))

    def __call__(self, image, *args):
        mix = np.zeros_like(image)
        for i in range(self.width):
            image_aug = image.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for j in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = self._apply(image_aug, op)
            mix = np.add(mix, self.ws[i] * self._normalize(image_aug), out = mix, casting = 'unsafe')
        mixed = (1 - self.m) * self._normalize(image_aug) + self.m * mix
        return mixed
    def _normalize(self, image):
        image = image / 255.
        image = (image - self.mean) / self.std
        return image
    def _apply(self, image, op):
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = op(image, self.severity)
        return np.array(image)