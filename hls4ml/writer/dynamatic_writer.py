import glob
import os
import stat
import tarfile
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np
import yaml

from hls4ml.writer.writers import Writer

config_filename = 'hls4ml_config.yml'


class DynamaticWriter(Writer):

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))

        srcpath = os.path.join(filedir, '../templates/dynamatic/nnet_utils/')
        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'

        if os.path.exists(dstpath):
            rmtree(dstpath)

        copytree(srcpath, dstpath)


    def write_hls(self, model):
        self.write_nnet_utils(model)
