import glob
import numpy as np
import cv2
from PIL import Image
from utils import config


def write(filename, content, class_num=2, color_map=True):
    """
    Save image array to a specified path.
    The image will be automatically recolored via the class number.
    :param filename: The specified path.
    :param content: Numpy array containing the image.
    :param class_num: Total class number.
    :param color_map: Whether change the probability into gray grade.
    """
    if class_num <= 1:
        raise Exception('ERROR: Class number should be >= 2.')
    color_stage = 255. / (class_num - 1) if color_map else 1.0
    new_image = Image.fromarray(np.uint8(content * color_stage))
    new_image.save(filename, "PNG")


def generate_effective_regions(size):
    """
    This function is used to generate effective regions for inference according to the given slide size.
    :param size: Given slide size, should be in the form of [w, h].
    """
    width = size[0]
    height = size[1]
    x_step = int(width / config.CENTER_SIZE)
    y_step = int(height / config.CENTER_SIZE)
    regions = []
    for x in range(0, x_step):
        for y in range(0, y_step):
            regions.append([x * config.CENTER_SIZE, y * config.CENTER_SIZE, 0, 0,
                            config.CENTER_SIZE - 1, config.CENTER_SIZE - 1])
    if not height % config.CENTER_SIZE == 0:
        for x in range(0, x_step):
            regions.append([x * config.CENTER_SIZE, height - config.CENTER_SIZE,
                                0, (y_step + 1) * config.CENTER_SIZE - height,
                            config.CENTER_SIZE - 1, config.CENTER_SIZE - 1])
    if not width % config.CENTER_SIZE == 0:
        for y in range(0, y_step):
            regions.append([width - config.CENTER_SIZE, y * config.CENTER_SIZE,
                                (x_step + 1) * config.CENTER_SIZE - width, 0,
                            config.CENTER_SIZE - 1, config.CENTER_SIZE - 1])
    if not (height % config.CENTER_SIZE == 0 or width % config.CENTER_SIZE == 0):
        regions.append([width - config.CENTER_SIZE, height - config.CENTER_SIZE,
                                (x_step + 1) * config.CENTER_SIZE - width, (y_step + 1) * config.CENTER_SIZE - height,
                        config.CENTER_SIZE - 1, config.CENTER_SIZE - 1])
    return regions


def generate_overlap_tile(region, dimensions):
    """
    This function is used to process border patches.
    """
    shifted_region_x = region[0] - config.BORDER_SIZE
    shifted_region_y = region[1] - config.BORDER_SIZE
    clip_region_x = config.BORDER_SIZE
    clip_region_y = config.BORDER_SIZE
    if region[0] == 0:
        shifted_region_x = shifted_region_x + config.BORDER_SIZE
        clip_region_x = 0
    if region[1] == 0:
        shifted_region_y = shifted_region_y + config.BORDER_SIZE
        clip_region_y = 0
    if region[0] == dimensions[0] - config.CENTER_SIZE:
        shifted_region_x = shifted_region_x - config.BORDER_SIZE
        clip_region_x = 2 * config.BORDER_SIZE
    if region[1] == dimensions[1] - config.CENTER_SIZE:
        shifted_region_y = shifted_region_y - config.BORDER_SIZE
        clip_region_y = 2 * config.BORDER_SIZE
    return [shifted_region_x, shifted_region_y], [clip_region_x, clip_region_y]


def image_to_array(input_image):
    """
    Loads image into numpy array.
    """
    im_array = np.array(input_image.getdata(), dtype=np.uint8)
    im_array = im_array.reshape((input_image.size[0], input_image.size[1]))
    return im_array


def post_processing(image_patch):
    """
    Remove small noisy points.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.FILTER_KERNEL, config.FILTER_KERNEL))
    open_patch = cv2.morphologyEx(image_patch, cv2.MORPH_OPEN, kernel)
    close_patch = cv2.morphologyEx(open_patch, cv2.MORPH_CLOSE, kernel)
    return close_patch


def concat_patches(temp_dir, image_name):
    """
    Concatenate the predicted patches into a thumbnail result.
    """
    prediction_list = glob.glob(temp_dir + image_name + '*_prediction.png')
    patch_list = []
    for prediction_image in prediction_list:
        name_parts = prediction_image.split('/')[-1].split('_')
        pos_x, pos_y = int(name_parts[-3]), int(name_parts[-2])
        patch_list.append([pos_x, pos_y])
    image_patches = []
    patch_list.sort()
    last_x = -1
    row_patch = []
    for position in patch_list:
        pos_x = position[0]
        pos_y = position[1]
        image = Image.open(temp_dir + '_'.join([image_name, str(pos_x), str(pos_y), 'prediction']) + '.png')
        original_width, original_height = image.size
        if original_width < config.THUMBNAIL_RATIO or original_height < config.THUMBNAIL_RATIO:
            continue
        image = image.resize(
            (int(original_width / config.THUMBNAIL_RATIO),
             int(original_height / config.THUMBNAIL_RATIO)), Image.NEAREST)
        image_patch = image_to_array(image)
        if not pos_x == last_x:
            last_x = pos_x
            if len(row_patch) == 0:
                row_patch = image_patch
            else:
                if not len(image_patches) == 0:
                    image_patches = np.column_stack((image_patches, row_patch))
                else:
                    image_patches = row_patch
                row_patch = image_patch
        else:
            row_patch = np.row_stack((row_patch, image_patch))
    prediction = np.column_stack((image_patches, row_patch))
    return prediction
