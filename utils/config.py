import os

INFERENCE_GPUS = [8]
CENTER_SIZE = 2000  # The effective patch size.
BORDER_SIZE = 100  # The boarder size containing surrounding information.
PATCH_SIZE = CENTER_SIZE + 2 * BORDER_SIZE
format_mapping = {
    'tif': 'OpenSlide',
    'svs': 'OpenSlide',
    'ndpi': 'OpenSlide',
    'scn': 'OpenSlide',
    'mrxs': 'OpenSlide',
    'bif': 'OpenSlide',
    'vms': 'OpenSlide',
}
MODEL_NAME = 'stomach'
INPUT_KEY = 'output'
PREDICT_KEY = 'output'
TF_SERVING_HOST = os.environ.get('TF_SERVING_HOST', '127.0.0.1')
TF_SERVING_PORT = int(os.environ.get('TF_SERVING_PORT', 9000))
TEMP_DIR = './temp/'  # Where to save the predicted patches.
THUMBNAIL_RATIO = 10  # Down-sample ratio from the predictions to the thumbnail.
KEEP_TEMP = False  # Whether to keep the predicted patches after generated the thumbnail.
DO_POST_PROCESSING = False  # Whether to post-process the predicted patches.
FILTER_KERNEL = 9  # The kernel size for post-processing.
