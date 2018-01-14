from pyfo.pyfoppl.foppl import imports  # useful!
import pyfo.pyfoppl.tests.gauss_2d as test

if __name__ == "__main__":
    print("=" * 50)
    print(test.code)
    # print("=" * 50)
    # print(test.graph)
    # print("=" * 50)
    # print(help(test.model))

    f = open('gauss_2d_model.py', 'w')
    f.write(test.code)
    f.close()

