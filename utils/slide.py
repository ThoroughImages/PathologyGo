"""
This class provides a unified interface decoding various slide formats.
Cancheng Liu
Email: liucancheng@thorough.ai
"""
import os.path
from openslide import OpenSlide
import config


class Slide(object):
    def __init__(self, image_path):
        super(Slide, self).__init__()
        self.image_path = image_path.strip()
        self.image_file = os.path.basename(image_path)
        self.image_name, self.suffix = self.image_file.split('.')
        if not self.suffix in config.format_mapping.keys():
            raise Exception('Error: File format ' + self.suffix + ' is supported yet.')
        self._slide = OpenSlide(image_path)
        self._level_downsamples = self._slide.level_downsamples
        self._level_dimensions = self._slide.level_dimensions
        self.width, self.height = self._level_dimensions[0]

    @property
    def dimensions(self):
        return self._level_dimensions[0]

    @property
    def level_dimensions(self):
        return self._level_dimensions

    @property
    def level_downsamples(self):
        return self._level_downsamples

    def read_region(self, level, location, size):
        (x, y), (w, h) = location, size
        try:
            _ds = self._level_downsamples[level]
            patch = self._slide.read_region(
                level=level, location=(int(x * _ds), int(y * _ds)), size=size)
        except Exception:
            raise
        else:
            return patch

    def get_thumbnail(self):
        try:
            image = self._slide.read_region(
                location=(0, 0), level=0, size=(1000, 1000))
        except Exception:
            raise
        else:
            return image
