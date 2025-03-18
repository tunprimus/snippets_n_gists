#!/usr/bin/env python3
from importlib import util
import subprocess
import sys


class PipFinder:
    @classmethod
    def find_spec(cls, module_name, path, target=None):
        """
        Attempt to install a missing module using pip and return its spec.

        If the module is not installed, we attempt to install it using pip.
        If the installation fails, we return None. Otherwise, we return the
        module's spec.
        """
        print(f"Module {module_name!r} not installed. Attempt to pip install.")
        cmd = f"{sys.executable} -m pip3 install {module_name}"
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError:
            return None

        return util.find_spec(module_name)


sys.meta_path.append(PipFinder)
