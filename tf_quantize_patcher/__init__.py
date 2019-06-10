import sys
import os
import importlib.util


def patch():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    spec = importlib.util.spec_from_file_location("tensorflow.contrib.quantize", current_dir + "/quantize/__init__.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tensorflow.contrib.quantize"] = mod
    spec.loader.exec_module(mod)
    sys.modules["tensorflow.contrib"].quantize = mod


patch()
