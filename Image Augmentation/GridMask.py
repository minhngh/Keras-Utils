import numpy as np
import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F

class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """
    def __init__(self, num_grid = 3, fill_value = 0, rotate = 0, mode = 0, always_apply = True, p = .5):
        super().__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max, self.rand_w_max = [[]] * 2
    def init_masks(self, height, width):
        if not self.masks:
            self.masks = list()
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1)):
                grid_h, grid_w = height / n_g, width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w)))
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                            int(i * grid_h) : int(i * grid_h + grid_h / 2),
                            int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                    if self.mode == 1:
                        this_mask = 1 - this_mask
                    self.masks.append(this_mask)
                    self.rand_h_max.append(grid_h)
                    self.rand_w_max.append(grid_w)
    def apply(self, image, task, rand_h, rand_w, angle, **kwargs):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[..., np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h : rand_h + h, rand_w : rand_w + w].astype(image.dtype)
        return image
    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)
        mid = np.random.randint(len(self.masks))
        mask = self.mask[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0
        return {
            'mask': mask,
            'rand_h': rand_h, 
            'rand_w': rand_w,
            'angle': angle
        }
    @property
    def targets_as_params(self):
        return ['image']
    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')