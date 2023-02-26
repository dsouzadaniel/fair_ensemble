
import random
import numpy as np
import numba
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace

from typing import Callable, Tuple
from ffcv.fields import rgb_image

from ffcv.libffcv import imdecode, resize_crop

from numba import njit


@numba.jit(nopython=True)
def set_seed(seed, indices):
    rep = ""
    for i in indices:
        rep += str(i)
    # local_seed = (hash(rep) + seed) % 2**31
    # print(indices)
    local_seed = indices[0] + seed
    # print("local_seed", local_seed)
    random.seed(local_seed)
    np.random.seed(local_seed)

@numba.jit(nopython=True)
def get_num():
    num = 0
    while True:
        yield num
        num += 1
        
class RandomHorizontalFlip(Operation):
    """Flip the image horizontally with probability flip_prob.
    Operates on raw arrays (not tensors).
    Parameters
    ----------
    flip_prob : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self, flip_prob: float = 0.5, seed: int = None):
        super().__init__()
        self.flip_prob = flip_prob
        self.seed = seed
        self.count = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        flip_prob = self.flip_prob
        seed = self.seed
        count = self.count

        if seed is None:

            def flip(images, _):
                for i in my_range(images.shape[0]):
                    if random.uniform(0, 1) < flip_prob:
                        images[i] = images[i, :, ::-1]
                return images

            flip.is_parallel = True
            return flip

        def flip(images, _, indices):
            
            set_seed(seed, [count])
            values = np.zeros(images.shape[0])
            for i in range(images.shape[0]):
                values[i] = random.uniform(0, 1)
            for i in my_range(images.shape[0]):
                if values[i] < flip_prob:
                    images[i] = images[i, :, ::-1]
            return images

        flip.is_parallel = True
        flip.with_indices = True
        return flip

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)



@njit(parallel=False, fastmath=True, inline="always")
def get_random_crop(
    height: int, width: int, scale, ratio, r1, r2, r3, r4
) -> Tuple[int, int, int, int]:
    area = height * width
    log_ratio = np.log(ratio)
    for trial in range(5):
        target_area = area * (r1[trial] * (scale[1] - scale[0]) + scale[0])
        aspect_ratio = np.exp(r2[trial] * (log_ratio[1] - log_ratio[0]) + log_ratio[0])
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            i = int(r3[trial] * (height - h + 1))
            j = int(r4[trial] * (width - w + 1))
            return i, j, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        j = 0
        w = width
        h = int(round(w / min(ratio)))
        i = int(r3[0] * (height - h + 1))
    elif in_ratio > max(ratio):
        i = 0
        h = height
        w = int(round(h * max(ratio)))
        j = int(r4[0] * (width - w + 1))
    else:
        i = 0
        j = 0
        w = width
        h = height
    return i, j, h, w


@njit(parallel=False, fastmath=True, inline="always")
def get_center_crop(height, width, _, ratio, r1, r2, r3, r4):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2

    return delta_h, delta_w, c, c


class RandomResizedCropRGBImageDecoder(rgb_image.RandomResizedCropRGBImageDecoder):
    """Decoder for :class:`~ffcv.fields.RGBImageField` that performs a Random crop and and a resize operation.
    It supports both variable and constant resolution datasets.
    Parameters
    ----------
    output_size : Tuple[int]
        The desired resized resolution of the images
    scale : Tuple[float]
        The range of possible ratios (in area) than can randomly sampled
    ratio : Tuple[float]
        The range of potential aspect ratios that can be randomly sampled
    """

    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.get_crop = get_random_crop
        self.count = seed

    def generate_code(self) -> Callable:

        jpg = 0

        mem_read = self.memory_read
        my_range = Compiler.get_iterator()
        imdecode_c = Compiler.compile(imdecode)
        resize_crop_c = Compiler.compile(resize_crop)
        # Compiler.compile(lambda seed: np.random.seed(seed))(self.seed)
        get_crop = self.get_crop

        scale = self.scale
        ratio = self.ratio
        seed = self.seed
        count = self.count

        
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)
        if seed is None:

            def decode(indices, my_storage, metadata, storage_state):
                destination, temp_storage = my_storage
                for dst_ix in my_range(len(indices)):
                    source_ix = indices[dst_ix]
                    field = metadata[source_ix]
                    image_data = mem_read(field["data_ptr"], storage_state)
                    height = np.uint32(field["height"])
                    width = np.uint32(field["width"])

                    if field["mode"] == jpg:
                        temp_buffer = temp_storage[dst_ix]
                        imdecode_c(
                            image_data,
                            temp_buffer,
                            height,
                            width,
                            height,
                            width,
                            0,
                            0,
                            1,
                            1,
                            False,
                            False,
                        )
                        selected_size = 3 * height * width
                        temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                        temp_buffer = temp_buffer.reshape(height, width, 3)

                    else:
                        temp_buffer = image_data.reshape(height, width, 3)

                    i, j, h, w = get_crop(
                        height,
                        width,
                        scale,
                        ratio,
                        np.random.rand(5),
                        np.random.rand(5),
                        np.random.rand(5),
                        np.random.rand(5),
                    )
                    resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

                return destination[: len(indices)]

            decode.is_parallel = True
            return decode

        def decode(indices, my_storage, metadata, storage_state):
            # print(my_storage)
            # counter[0] += 1
            # print('count', count)
            set_seed(seed, [count])
            # set_seed(seed, counter)
            r = np.zeros(4 * len(indices) * 5)
            for i in range(4 * 5 * len(indices)):
                r[i] = random.uniform(0, 1)
            r = r.reshape(-1, 4, 5).astype("float32")

            destination, temp_storage = my_storage
            for dst_ix in my_range(len(indices)):
                source_ix = indices[dst_ix]
                field = metadata[source_ix]
                image_data = mem_read(field["data_ptr"], storage_state)
                height = np.uint32(field["height"])
                width = np.uint32(field["width"])

                if field["mode"] == jpg:
                    temp_buffer = temp_storage[dst_ix]
                    imdecode_c(
                        image_data,
                        temp_buffer,
                        height,
                        width,
                        height,
                        width,
                        0,
                        0,
                        1,
                        1,
                        False,
                        False,
                    )
                    selected_size = 3 * height * width
                    temp_buffer = temp_buffer.reshape(-1)[:selected_size]
                    temp_buffer = temp_buffer.reshape(height, width, 3)

                else:
                    temp_buffer = image_data.reshape(height, width, 3)

                i, j, h, w = get_crop(
                    height,
                    width,
                    scale,
                    ratio,
                    r[dst_ix, 0],
                    r[dst_ix, 1],
                    r[dst_ix, 2],
                    r[dst_ix, 3],
                )
                resize_crop_c(temp_buffer, i, i + h, j, j + w, destination[dst_ix])

            return destination[: len(indices)]

        decode.is_parallel = True
        return decode

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, AllocationQuery]:

        widths = self.metadata["width"]
        heights = self.metadata["height"]
        # We convert to uint64 to avoid overflows
        self.max_width = np.uint64(widths.max())
        self.max_height = np.uint64(heights.max())
        output_shape = (self.output_size[0], self.output_size[1], 3)
        my_dtype = np.dtype("<u1")
        return (
            replace(previous_state, jit_mode=True, shape=output_shape, dtype=my_dtype),
            (
                AllocationQuery(output_shape, my_dtype),
                AllocationQuery(
                    (self.max_height * self.max_width * np.uint64(3),), my_dtype
                ),
            ),
        )

