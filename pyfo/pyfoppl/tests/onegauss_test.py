from pyfo.pyfoppl.foppl import imports  # useful!
import pyfo.pyfoppl.tests.onegauss as test_onegauss

if __name__ == "__main__":
    print("=" * 50)
    print(test_onegauss.code)
    # print("=" * 50)
    # print(test_onegauss.graph)
    # print("=" * 50)
    # print(help(test_onegauss.model))

    f = open('onegauss_model.py', 'w')
    f.write(test_onegauss.code)
    f.close()

