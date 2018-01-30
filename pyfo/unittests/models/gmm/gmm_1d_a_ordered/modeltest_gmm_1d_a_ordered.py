from pyfo.pyfoppl.foppl import imports
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import pyfo.unittests.models.gmm.gmm_1d_a_ordered.gmm_1d_a_ordered as test


### model
print(test.model)
# test.model.display_graph()
print(test.model.gen_pdf_code)