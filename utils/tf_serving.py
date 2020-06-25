"""
Basic components for using TF Serving.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import io
import numpy as np
from PIL import Image
import grpc
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc._cython import cygrpc
import config


def decode_tensor(contents, downsample_factor):
    images = [Image.open(io.BytesIO(content)) for content in contents]
    dsize = (images[0].size[0] * downsample_factor, images[0].size[1] * downsample_factor)
    images = [image.resize(dsize, Image.BILINEAR) for image in images]
    mtx = np.array([np.asarray(image, dtype=np.uint8) for image in images], dtype=np.uint8)
    mtx = mtx.transpose((1, 2, 0))
    return mtx


class TFServing(object):
    """
    Tensorflow Serving client, send prediction request.
    """

    def __init__(self, host, port):
        super(TFServing, self).__init__()

        channel = self._insecure_channel(host, port)
        self._stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    def _insecure_channel(self, host, port):
        channel = grpc.insecure_channel(
            target=host if port is None else '{}:{}'.format(host, port),
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)])
        return grpc.beta.implementations.Channel(channel)

    def predict(self, image_input, model_name):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.inputs[config.INPUT_KEY].CopyFrom(tf.contrib.util.make_tensor_proto(
            image_input.astype(np.uint8, copy=False)))
        try:
            result = self._stub.Predict(request, 1000.0)
            image_prob = np.array(result.outputs[config.PREDICT_KEY].int_val)
        except Exception as e:
            raise e
        else:
            return image_prob.astype(np.uint8)
