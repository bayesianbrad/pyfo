from pyfo.pyfoppl.foppl import imports
# from pyfo.pyfoppl.tests import test_if_src
import tests.test_if_src as test_if_src

# what is the type of test_if_src, obj of which class?

print(dir(test_if_src))
print("=" * 50)
print(test_if_src.code)
print("=" * 50)
print(test_if_src.graph)
print("=" * 50)
print(help(test_if_src.model))


