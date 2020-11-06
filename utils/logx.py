import json
import joblib
import shutil
import numpy as np
import torch
import warnings
import os
import time
import atexit
from utils.mpi_tools import proc_id, mpi_statistics_scalar
from utils.serialization_utils import convert_json


class Logger:

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        if proc_id() == 0:
            self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
            if os.path.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway." % self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(os.path.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print("Logging data to %s" % self.output_file.name)

        else:
            self.output_dir = None
            self.output_file = None

        self.first_row = None
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg):
        if proc_id() == 0:
            print(msg)

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_config(self, config):
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id() == 0:
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            print("Saving config:\n")
            print(output)
            with open(os.path.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, itr=None):
        if proc_id() == 0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl' % itr
            try:
                joblib.dump(state_dict, os.path.join(self.output_dir, fname))
            except:
                self.log('Warning: could not pickle state_dict.')
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch.simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        if proc_id() == 0:
            assert hasattr(self, 'pytorch_saver_elements'), "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = os.path.join(self.output_dir, fpath)
            fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
            fname = os.path.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        if proc_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%s' + '%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-"*n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers)+"\n")
                self.output_file.write("\t".join(map(str,vals))+"\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row=False

class EpochLogger(Logger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, states[0])
            if not(average_only):
                super().log_tabular('Std' + key, states[1])
            if with_min_and_max:
                super().log_tabular('Max' + key, states[3])
                super().log_tabular('Min' + key, states[2])
        self.epoch_dict[key] = []

    def get_states(self, key):
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi_statistics_scalar(vals)
