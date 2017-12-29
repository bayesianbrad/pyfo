from py_foppl.foppl.compiler import *
from py_foppl.foppl.foppl_parser import *
from py_foppl.foppl.model_generator import *

from pyfo.py_foppl.foppl.graphs import *

if __name__ == "__main__":

    G = Graph({'x2', 'x3', 'x5', 'x06', 'x10', 'x12', 'x30'},
              {('x2', 'x06'), ('x2', 'x10'), ('x2', 'x12'),
               ('x3', 'x06'), ('x5', 'x10'), ('x06', 'x12'),
               ('x3', 'x30'), ('x10', 'x30')},
              {'x2': '20',
               'x3': 'dist.Normal(0, 1)',
               'x5': 'dist.Binom(5)',
               'x06': 'x2 + x3',
               'x10': 'x2 * x5',
               'x12': 'x06 - x2'},
              {'x30': '7.0'})
    #print(G)
    #print("=" * 50)
    #m = Model_Generator(G)
    #print(m.generate_class())
    #print("=== done ===")

    program = """
    (let [x (sample (normal 1.0 5.0))
          y (+ x 1)]
          (observe (normal y 2.0) 7.0)
          y)
    """
    ast = parse(program)
    print(ast)

    c = Compiler()

    graph, expr = c.walk(ast)
    print(graph)
    print("=" * 30)
    print(expr)

    print("===== DONE =====")