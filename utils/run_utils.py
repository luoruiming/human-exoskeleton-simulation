from utils.mpi_tools import mpi_fork, msg
from utils.serialization_utils import convert_json
import base64
from copy import deepcopy
import cloudpickle
import json
import numpy as np
import os
import os.path as osp
import psutil
import string
import subprocess
from subprocess import CalledProcessError
import sys
from textwrap import dedent
import time
from tqdm import trange
import zlib

DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datastamp=False):
    datastamp = datastamp

    ymd_time = time.strftime("%Y-%M-%D_") if datastamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        if datastamp:
            hms_time = time.strftime("%Y-%M-%D_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = os.path.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=os.path.join(data_dir, relpath), exp_name=exp_name)
    return logger_kwargs

