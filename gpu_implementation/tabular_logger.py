import os
import sys
import traceback as traceback_module
import time
from collections import OrderedDict
import json
import numpy as np
import tempfile


DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


def record_tabular(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    CURRENT().record_tabular(key, val)


def dump_tabular():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    CURRENT().dump_tabular()


def __log(level, *args):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    CURRENT().log(level, *[str(a) for a in args])


def debug(*args):
    __log(DEBUG, *args)


def info(*args):
    __log(INFO, *args)


log = info


def warn(*args):
    __log(WARN, *args)


def error(*args):
    __log(ERROR, *args)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    CURRENT().set_level(level)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return CURRENT().get_dir()


def get_expt_dir():
    sys.stderr.write("get_expt_dir() is Deprecated. Switch to get_dir()\n")
    return get_dir()

# ================================================================
# Backend
# ================================================================


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.number):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


__DEFAULT = None
def DEFAULT():
    global __DEFAULT
    if __DEFAULT is None:
        set_default(TabularLogger())
    return __DEFAULT

def set_default(logger, replace=True):
    global __DEFAULT
    if __DEFAULT is None or replace:
        __DEFAULT = logger
    return __DEFAULT

__CURRENT = None
def CURRENT():
    global __CURRENT
    if __CURRENT is None:
        __CURRENT = DEFAULT()
    return __CURRENT

class TabularLogger(object):
    def __init__(self, dir=None, format='{asctime} {message}\n', datefmt='%m/%d/%Y %I:%M:%S %p'):
        self.format = format
        self.datefmt = datefmt
        self.name2val = OrderedDict()  # values this iteration
        self.level = INFO
        self.cassandra_level = WARN
        self.text_outputs = [sys.stdout]
        self.tbwriter = None
        self.experiment_name = None
        self.dir = dir
        if dir is None:
            dir = self.log_dir()
        if dir is not None:
            try:
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                self.text_outputs.append(open(os.path.join(dir, "log.txt"), "a+"))
            except:
                self.exception("Unable to save to {}".format(dir))

        set_default(self, False)

    def log_dir(self):
        if self.dir:
            return self.dir
        self.dir = tempfile.mkdtemp()
        return self.dir

    # Logging API, forwarded
    # ----------------------------------------
    def record_tabular(self, key, val):
        self.name2val[key] = val

    def flush_tabular(self):
        self.name2val.clear()

    def dump_tabular(self):
        # Create strings for printing
        key2str = OrderedDict()
        for (key, val) in self.name2val.items():
            try:
                if hasattr(val, "__float__"):
                    valstr = "%-8.3g" % val
                else:
                    valstr = val
                assert self._truncate(key) not in key2str, 'Truncated tabular key has already been used!'
                key2str[self._truncate(key)] = self._truncate(str(valstr))
            except:
                self.log(INFO, 'Cannot dump_tabular: {}:{}'.format(key, val))
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))
        # Write to all text outputs
        self._write_text("-" * (keywidth + valwidth + 13), "\n")
        for (key, val) in key2str.items():
            self._write_text("| ", key, " " * (keywidth - len(key)),
                             " | ", val, " " * (valwidth - len(val)), " |\n")
        self._write_text("-" * (keywidth + valwidth + 13), "\n")
        for f in self.text_outputs:
            try:
                f.flush()
            except OSError:
                sys.stderr.write('Warning! OSError when flushing.\n')
        # Write to tensorboard
        if self.tbwriter is not None:
            self.tbwriter.write_values(self.name2val)
            self.name2val.clear()

    def log(self, level, *args):
        if self.level <= level:
            self._do_log(*args)

    def exception(self, *args):
        result = "".join(traceback_module.format_exception(*sys.exc_info()))
        self.log(ERROR, result, *args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for f in self.text_outputs[1:]:
            f.close()
        if self.tbwriter:
            self.tbwriter.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, *args):
        self._write_text(*args)
        for f in self.text_outputs:
            try:
                f.flush()
            except OSError:
                print('Warning! OSError when flushing.')

    def _write_text(self, *strings):
        s = self.format.format(asctime=time.strftime(self.datefmt), message=' '.join(strings))
        for f in self.text_outputs:
            f.write(s)

    def _truncate(self, s):
        if len(s) > 33:
            return "..." + s[-30:]
        else:
            return s

def log_dir():
    return CURRENT().log_dir()

def flush_tabular():
    return CURRENT().flush_tabular()

def set_log_dir(dir):
    CURRENT().dir = dir


def exception(exception, *args):
    CURRENT().exception(exception, *args)
