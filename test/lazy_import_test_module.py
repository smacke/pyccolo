# -*- coding: utf-8 -*-
import sys

import numpy as np

assert list(np.arange(5)) == list(range(5))
print(len([mod for mod in sys.modules if "numpy" in mod]))
