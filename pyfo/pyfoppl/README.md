# FOPPL in Python

**This repository is deprecated and has been superseded by a 
new implementation
[PyFOPPL-2](https://github.com/Tobias-Kohn/PyFOPPL-2).**

This is an implementation of a Anglican/Clojure-based 
_First Order Probabilistic Programming Language_ (FOPPL) in
Python.
The design of the FOPPL language is due to 
[Frank Wood](http://www.robots.ox.ac.uk/~fwood/), 
Jan-Willem van de Meent, and Brooks Paige.

The package takes FOPPL-code as input and creates a 
graph-based model for it.

Please note that this is a work in progress and by no means
complete, yet.

## Usage

_Minimal system requirements: Python 3.4 (we have tested the system
on Python 3.5 and Python 3.6)._

You might have a FOPPL-model such as the following, saved as
a file named `my_model.clj` in the parent-directory of your
project:
```clojure
(let [x (sample (normal 0.0 1.0))]
  (observe (normal x 0.707))
  x)
```
After you enable FOPPL-auto-imports through 
`import foppl.imports`, you just import your model as you
would with a normal Python-module: 
```python
import foppl.imports
import my_model

print(my_model.model.gen_prior_samples())
```
The imported module exposes the following three fields:
- `model`: the compiled model as a class with several 
   class-methods such as `gen_prior_samples()`.
- `graph`: the graph as created from the original FOPPL
   program.
- `code`: the Python-code, that was created from the 
   graph and then compiled into the model-class.
   
### Options
   
The FOPPL-compiler supports some options, which must be set before
importing any FOPPL-programs. The options can be set like this:
```python
from foppl import Options
Options.uniform_conditionals = False

import my_model

...
```
Details of the available options can be found in the file
`foppl/__init__.py`.

### Drawing the Graph

If you have the modules `networkx`, `matplotlib`, and `graphviz`
installed (the last one being optional), you can get a visual
representation of the graph.
```python
model = my_model.model
model.graph.draw_graph()
```

## Hacking

_NB: The design of the compiler follows as closely as possible an 
implementation in Clojure by Brooks Paige and Jan-Willem van 
de Meent, however, with various modifications and extensions._

### Overview: How the Compiler Works

_The compiler takes a Clojure-like input (the
FOPPL program), creates a graph representing the probabilistic
structure of the program, and then transforms the graph into a
directly usable Python model._

The FOPPL source code is first read into clojure-like datastructures,
such as forms and symbols. The datastructures can be found in
the module `foppl_objects` and the reader responsible for the
transformation in `foppl_reader`.

The parser then transforms the clojure-datastructures into an
Abstract Syntax Tree (AST). The parser can be found in 
`foppl_parser` and the AST in `foppl_ast`.

Once we have the AST, we can use common tree-walking algorithms.
The compiler as found in the module `compiler` walks the AST
and creates a graphical model (found in `graphs`). Finally, 
the `model_generator` generates the Python-code for a class,
which represents the original FOPPL-program as a model.

An important part of the FOPPL are distributions. The names
of the distributions must be converted from FOPPL to their
Python counterparts, and we need to determine the parameters
of the different distributions as well as whether a given
distribution is continuous or discrete. The necessary
defnitions are to be found in the module `foppl_distributions`.

The module `imports` contains the magic behind importing FOPPL
directly from inside Python. It basically register a new 
importer in the Python system, which looks for FOPPL-code
and compiles it, once the FOPPL-code has been found.

### Changing the Model-Class Creation

The model-generator has two major mechanisms for customization.

1. Redefine its fields `interface_name` and `interface_source`
   to have the class derive from a particular class or interface.
   While `interface_name` specifies the name of the class or
   interface, `interface_source` specifies the module where the
   interface can be found, or an empty string `''` if no 
   module needs to be imported.
   
   By settings the list `imports`, you can also specify any
   number of modules to be imported at the beginning of the
   file (the interface as discussed above is imported 
   automatically and needs not be part of this list).
   
2. During the assembly of the class, each method starting with
   `_gen_` is called to return the body, or an argument and a
   body for the method to be created. The body can either be
   a string or a list of strings. In the case of a list of 
   strings, the individual strings in the list are interpreted
   as lines and joined together with line-breaks `'\n'` in
   between. In either case, the body is indented automatically,
   
   Indentation must only be taken care of when using compound
   statements such as `if` inside the method's body. You should
   use the tab-character `'\t'` for all indentations.
   
   The method `_gen_vars()`, for instance, returns a string
   such as `"return ['x1', 'x2']"`. This is then used to create
   the following method in the model class:
   ```python
    @classmethod
    def gen_vars(self):
        return ['x1', 'x2']
   ```

   The method `_gen_pdf` returns a tuple `('state', body)` where
   `body` is a string as, e.g., `"x = 12\nreturn x+34"`. These
   two values are used to create a method with a parameter:
    ```python
    @classmethod
    def gen_pdf(self, state):
        x = 12 
        return x + 34
    ```
    
### Adding New Functions and 'Macros'

The short story: in order to add a new function `foo`, add a method
`visit_call_foo(self, node)` to the compiler (see example below).  

The long story:
there are two possible places to define new functions or programming
structures: as part of the *parser*, or as part of the *compiler*.

1. **Parser**: the parser operates on general forms (lists), made up
   of other forms, symbols, and values, respectively. A function call
   such as `(apply f 12)` is a form containing two symbols (`apply`
   and `f`, resp.), as well as the value `12`. The parser's
   responsibility is to recognize the structure of such forms (in
   particular of *special forms* such as `if`, `let`, `def`, etc.)
   and transform the structures into dedicated AST-nodes. In the
   case of `apply`, for instance, the parser creates an instance
   of an `AstFunctionCall`-node, which takes a function's name and
   a list of arguments.
   
   As soon as you introduce a new specialized AST-node, you must 
   update the parser to recognize the respective structure in the
   code and transform it to your AST-node. However, you may also
   want to reuse existing AST-nodes. This is done, for instance,
   in the case of functions such as `first`, which are translated
   by the parser to `(get SEQ 0)` and the like. Hence, `first`
   never occurs inside the AST, and later stages of the compiler
   have fewer cases to handle.
   
2. **Compiler**: the compiler walks each node of the AST in a 
   DFS (Depth First Search). Every AST-node that inherits from
   `Node` exposes a method `walk(walker)`, which looks for the
   appropriate `visit`-method in the walker to call. For instance,
   the AST-node `AstBinary` looks for a method 
   `walker.visit_binary(node: AstBinary)` and, when found, calls
   this method with itself as an argument. If no such method
   exists, the AST-node calls the generic `visit_node` intead.
   
   The AST itself does not impose any restrictions on what values
   the `visit`-methods should return. For the compiler, however,
   each `visit`-method returns a tuple containing a graph and an
   expression as string. The graph gives information about the
   samples and observed values in the AST-node (and its subnodes)
   as well as the relationships between the various random
   values. The expression is the actual expression as Python
   code.
   
   If you want to change how a specific AST-node is translated
   to a graph and/or expression, you change its specific
   `visit`-method. You have to write a new specific 
   `visit`-method if you have added new AST-nodes.
   
   Note that `FunctionCall`-nodes try to find a specific
   `visit`-method before using the generic `visit_functioncall`.
   The function call `(sample (normal 0 1))` looks for the
   method `visit_call_sample(node: AstFunctionCall)` first.
   This allows you to easily add very specific behaviour for
   some function calls.
   
   Finally, the compiler uses a *scope-stack*. For nodes such
   as `let`, the compiler pushes a new scope onto the stack and
   then defines all bindings within this scope. At the end of
   `let`, the scope is popped from the stack and all bindings
   are thereby discarded.
   
   We distinguish between functions and other symbols, since
   functions are treated differently and are not first-class
   objects in FOPPL.
   
### The Role of the Optimizer

In order to create a finite graphical model of the given 
FOPPL-program, we need to impose several restrictions on the 
available structures. For instance, we do not allow or support
recursion, and the builtin loop-structure needs an explicit
constant number of iterations. Data vectors must all be fully
known at compile time.

In various cases, the original severe restrictions can be
slightly relaxed if the compiler is capable of evaluating some
parts of the program, and simplify it. Consider, for instance,
the following example:
```clojure
(let [original_data [1 2 3 4]
      data (map (fn [x] (* x x)) original_data)]
        ; code working with data
      )
```
To the naive compiler, `data` is an unknown data structure in
this case, and it can not evaluate its length or contents. When
the compiler, however, can perform partial evaluation on the
code, it will resolve `data` to be `[1, 4, 9, 16]`.

This partial evaluations are done by the `Optimizer`.

### Example of a Custom Function

Let's say, we want to add a `max`-function to our compiler. In
order to do so, we add a method `visit_call_max` to the compiler
class. This method must return a tuple, comprising the graph of
the node, as well as the Python expression as a string.

In our case, we assume that `max` always has two arguments. Both
of these arguments must be 'compiled' on their own. Afterwards,
we merge the graphs of both arguments, and create a new Python
expression for the result.
```python
def visit_call_max(self, node: AstFunctionCall):
    if len(node.args) == 2:
        graph_A, expr_A = node.args[0].walk(self)   # compile first arg
        graph_B, expr_B = node.args[1].walk(self)   # compile second arg
        graph = graph_A.merge(graph_B)              # merge graphs
        expr = "max({}, {})".format(expr_A, expr_B) # create expression
        return graph, expr
    else:
        raise SyntaxError("Too many or too few arguments for 'max'")
```    
However, we might want to do some optimizations first. If both
arguments are constant and can be evaluated during compile time, we
do so. The first step towards this is calling the optimization-step
on both arguments.
```python
def visit_call_max(self, node: AstFunctionCall):
    if len(node.args) == 2:
        arg_A = self.optimize(node.args[0])
        arg_B = self.optimize(node.args[1])
        if isinstance(arg_A, AstValue) and isinstance(arg_B, AstValue):
            result = max(arg_A.value, arg_B.value)
            return Graph.EMPTY, repr(result)
        # as before...
        graph_A, expr_A = arg_A.walk(self) 
        graph_B, expr_B = arg_B.walk(self) 
        graph = graph_A.merge(graph_B)     
        expr = "max({}, {})".format(expr_A, expr_B)
        return graph, expr
    else:
        raise SyntaxError("Too many or too few arguments for 'max'")
```

## License

MIT. See [LICENSE.txt](LICENSE.txt).

## Contributors

- Tobias Kohn
- Bradley Gram-Hansen
- Yuan Zhou
