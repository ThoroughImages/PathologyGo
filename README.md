# PathologyGo
Core components of PathologyGo, the AI assistance system designed for histopathological inference.

<b>Inference</b>
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