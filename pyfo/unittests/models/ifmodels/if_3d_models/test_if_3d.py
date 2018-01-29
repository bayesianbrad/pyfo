from pyfo.pyfoppl.foppl import imports
##  from pyfo.unittests.models.ifmodels.if_3d_model import model
import pyfo.unittests.models.ifmodels.if_3d_models.if_3d_a as test_a
import pyfo.unittests.models.ifmodels.if_3d_models.if_3d_b as test_b

# model
print(test_a.model)
test_a.model.display_graph()

print(test_b.model)
test_b.model.display_graph()