"""
Inference module supporting whole slide images using TF Serving.
Eric Wang
Email: eric.wang@thorough.ai
"""
import os
import shutil
from utils.tf_serving import TFServing
import numpy as np
from utils.slide import Slide
from utils import config
from utils.libs import write, generate_effective_regions, generate_overlap_tile, \
    post_processing, concat_patches


class Inference:
    def __init__(self, data_dir, data_list, class_num, result_dir, use_level):
        """
        This is the main inference module for the sake of easy to call.
        :param data_dir: The directory storing the while image slides.
        :param data_list: The text file indicating the slide names.
        :param class_num: Number of predicted classes.
        :param result_dir: Where to put the predicted results.
        :param use_level: Which slide size we want to analyze, 0 for 40x, 1 for 20x, etc.
        """
        if data_dir.endswith('/'):
            self.data_dir = data_dir
        else:
            self.data_dir = data_dir + '/'
        self.data_list = data_list
        self.class_num = class_num
        if result_dir.endswith('/'):
            self.result_dir = result_dir
        else:
            self.result_dir = result_dir + '/'
        self.use_level = use_level
        self.config = config

    @staticmethod
    def _infer(tfs_client, image):
        """
        Inference for an image patch using TF Serving.
        :param tfs_client: TF Serving client.
        :param image: The image patch.
        :return: Predicted heatmap.
        """
        try:
            prediction = tfs_client.predict(image, config.MODEL_NAME)
        except Exception as e:
            print('TF_SERVING_HOST: {}'.format(config.TF_SERVING_HOST))
            print(e)
            raise
        else:
            return prediction

    def run(self):
        """
        Proceeds the inference procedure.
        """
        inference_list = open(self.data_list).readlines()
        tfs_client = TFServing(config.TF_SERVING_HOST, config.TF_SERVING_PORT)
        for item in inference_list:
            image_name, image_suffix = item.split('\n')[0].split('/')[-1].split('.')
            print('[INFO] Analyzing: ' + self.data_dir + item.split('\n')[0])
            if not image_suffix in self.config.format_mapping.keys():
                print('[ERROR] File ' + item + ' format not supported yet.')
                continue
            image_handle = Slide(self.data_dir + item.split('\n')[0])
            image_dimensions = image_handle.level_dimensions[self.use_level]
            regions = generate_effective_regions(image_dimensions)
            index = 0
            region_num = len(regions)
            temp_dir = self.config.TEMP_DIR + image_name + '/'
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            for region in regions:
                shifted_region, clip_region = generate_overlap_tile(region, image_dimensions)
                index += 1
                if index % 1 == 0:
                    print('[INFO]  Progress: ' + str(index) + ' / ' + str(region_num))
                input_image = np.array(image_handle.read_region(
                    location=(int(shifted_region[0]),
                              int(shifted_region[1])),
                    level=self.use_level, size=(self.config.PATCH_SIZE, self.config.PATCH_SIZE)))[:, :, 0: 3]
                prediction_result = self._infer(tfs_client, input_image)
                prediction_result = prediction_result[clip_region[0]: (self.config.CENTER_SIZE + clip_region[0]),
                                    clip_region[1]: (self.config.CENTER_SIZE + clip_region[1])]
                prediction_result = prediction_result[region[2]:(region[4] + 1), region[3]:(region[5] + 1)]
                if self.config.DO_POST_PROCESSING:
                    prediction_result = post_processing(prediction_result)
                write(temp_dir + image_name + '_' + str(region[0]) + '_' + str(region[1])
                      + '_prediction.png', prediction_result, self.class_num)
            print('[INFO] Postprocessing...')
            full_prediction = concat_patches(temp_dir, image_name)
            write(self.result_dir +
                  '_'.join([image_name, 'prediction_thumbnail']) + '.png', full_prediction)
            if not self.config.KEEP_TEMP:
                shutil.rmtree(temp_dir)
            print('[INFO] Prediction saved to ' + self.result_dir + '_'.join(
                [image_name, 'prediction_thumbnail']) + '.png')
