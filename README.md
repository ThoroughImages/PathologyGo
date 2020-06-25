# PathologyGo
Core components of PathologyGo, the AI assistance system designed for histopathological inference.

<b>Dependency</b>

* Docker
* Python 2.7 and 3.x
* openslide
* tensorflow_serving
* grpc
* pillow
* numpy
* opencv-python

<b>Dockerized TensorFlow Serving</b>

* GPU version: [GitHub](https://github.com/physicso/tensorflow_serving_gpu), [Docker Hub](https://hub.docker.com/r/physicso/tf_serving_gpu).
* CPU version: [Docker Hub](https://hub.docker.com/r/tensorflow/serving).

<b>Quick Start</b>

This code is easy to implement. Just change the path to your data repo:

```python
from utils import config
GPU_LIST = config.INFERENCE_GPUS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join('{0}'.format(n) for n in GPU_LIST)
from inference import Inference


if __name__ == '__main__':
    pg = Inference(data_dir='/path/to/data/', data_list='/path/to/list',
                    class_num=2, result_dir='./result', use_level=1)
    pg.run()

```

You may configure all the model-specific parameters in `utils/config.py`.

<b>Example</b>

Use the [CAMELYON16](https://camelyon16.grand-challenge.org/) test dataset as an example, 
the data path should be `/data/CAMELYON/`, and the content of the data list is

```
001.tif
002.tif
...
```

The predicted heatmaps will be written to `./result`.

<b>DIY Notes</b>

You may use other exported models. You can change the model name for TensorFlow Serving in `utils/config.py`. Just remember to modify `class_num` and `use_level`.

Note that the default input / output tensor name should be `input` / `output`.
