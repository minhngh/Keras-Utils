from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class MultiOutputDataGenerator(ImageDataGenerator):
    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        targets = None
        target_dims = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.hstack((targets, target))
            target_dims[output] = target.shape[1]
            ordered_outputs.append(output)
        for flowx, flowy in super().flow(x, targets, batch_size = batch_size, shuffle = shuffle):
            target_dict ={}
            i = 0
            for output in ordered_outputs:
                target_dim = target_dims[output]
                target_dict[output] = flowy[:, i: i + target_dim]
                i += target_dim
            yield flowx, target_dict