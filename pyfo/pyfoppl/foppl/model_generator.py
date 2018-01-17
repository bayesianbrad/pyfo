#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
#
# 21. Dec 2017, Tobias Kohn
# 16. Jan 2018, Tobias Kohn
#
import datetime
import importlib
from .graphs import Graph
from .runtime_functions import runtime_functions
from . import Options

class Model_Generator(object):

    def __init__(self, graph: Graph, name: str = 'model'):
        self.graph = graph
        self.name = name
        if Options.model_interface:
            name, source = Options.model_interface
        else:
            name, source = 'object', ''
        self.interface_name = name
        self.interface_source = source
        self.imports = Options.model_imports.copy()
        self._output = None

    def generate_class(self) -> str:
        """
        Generates the model-class and returns the Python source code for the model.

        :return:  A string containing the source code for the model.
        """
        # We only generate the output if it has not yet been generated before.
        if self._output is None:
            self.__generate_source()
        return self._output

    def generate_class_and_import(self, name: str):
        """
        Generated the model for the graph as a separate Python-module and imports it.
        After the importing, the model class is returned.

        :param name: [required] the name of the module to be
        :return:     The created model-class.
        """
        assert(len(name) > 0)
        with open(name + ".py", mode='w') as f:
            f.writelines([self.generate_class()])
        importlib.invalidate_caches()
        result = importlib.import_module(name)
        globals()[name] = result
        return getattr(result, self.name)

    def __generate_source(self):
        """
        Generates the source-code of the model-class with all data and methods included.

        :return: A string containing the entire source code of the file.
        """
        self._output = ''

        # We add the current date and time
        self._output += '#\n'
        self._output += '# Generated: {}\n'.format(datetime.datetime.now())
        self._output += '#\n'

        # We add the imports, used functions and the header of the class...
        self._output += '\n'.join(self.imports)
        if len(self.interface_source) > 0:
            self._output += '\nfrom {} import {}'.format(self.interface_source, self.interface_name)
        if len(self.graph.used_functions) > 0:
            self._output += '\n\n#Runtime functions:'
            for f in self.graph.used_functions:
                if f in runtime_functions:
                    self._output += '\n' + runtime_functions[f]
        self._output += '\n\nclass {name}({interface}):\n'.format(
            name = self.name,
            interface = self.interface_name
        )

        # We add the doc-string, if there is any...
        docstring = self._generate_docstring()
        if docstring is not None and len(docstring) > 0:
            self._output += '\t"""\n\t{}\n\t"""\n'.format(docstring.replace('\n', '\n\t'))

        # We add all the vertices and edges of the graph to our model
        if self.graph:
            #self._output += '\tvertices = '
            #self._output += str(self.graph.vertices)
            #self._output += '\n\tarcs = '
            #self._output += str(self.graph.arcs)
            #self._output += '\n\tnames = '
            #self._output += repr(self.graph.original_names)
            #self._output += '\n\tcond_functions = '
            #self._output += self.graph.get_conditional_functions().replace('\n', '\n\t')
            #self._output += '\n\tdisc_dists = '
            #self._output += self.graph.get_discrete_distributions().replace('\n', '\n\t')
            #self._output += '\n'
            self._output += self._format_method(name='get_vertices',
                                                code='vertices = {}\n'.format(str(self.graph.vertices)) +
                                                     'return list(vertices)')
            self._output += self._format_method(name='get_arcs',
                                                code='arcs = {}\n'.format(str(self.graph.arcs)) +
                                                     'return list(arcs)')
            self._output += self._format_method(name='get_discrete_distribution',
                                                code='disc_dists = {}\n'.format(self.graph.get_discrete_distributions()) +
                                                     'return disc_dists')
            self._output += self._format_method(name='get_cond_functions',
                                                code='cond_functions = {}\n'.format(self.graph.get_conditional_functions()) +
                                                     'return cond_functions')
            self._output += self._format_method(name='get_dist_parameter_size', args='name',
                                                code='dist_sizes = {}\n'.format(self.graph.get_distribution_sizes()) +
                                                     'if name in dist_sizes:\n'
                                                     '\treturn dist_sizes[name]\n'
                                                     'else:\n'
                                                     '\treturn None')

            # We go through the class and call each method that starts with '_gen_'. The methods are expected
            # to return a string with the code for a function or method to be included
            for m in sorted(dir(self)):
                if m.startswith('_gen_'):
                    result = getattr(self, m)()
                    if result:
                        if len(result) == 2:
                            args, code = result
                        else:
                            args = None
                            code = result
                        result = self._format_method(name=m[1:], args=args, code=code)
                        self._output += result

        return self._output

    def _generate_docstring(self) -> str:
        """
        Returns the doc-string of the model-class. Per default, this doc-string includes a 'pretty-print' of the
        graphical model.

        :return: The doc-string of the class to be included in the model.
        """
        if self.graph:
            return repr(self.graph)
        else:
            return ""

    def _format_method(self, *, name: str=None, args=None, code=None) -> str:
        """
        Takes a name, possibly arguments, and the body of a function, and creates a proper method out of it to be
        included in the model class.

        The code does ot need to be indented properly. Indentation is taken care of by this function.

        :param name: The name of the method or function. Must be a non-empty string.
        :param args: Possible arguments, either as a string or as a tuple/list of strings.
        :param code: The body of the method or function, either as a string or as a tuple/list of strings.
        :return:     The method or function as Python code as a string.
        """
        # Create the arguments with 'self' as the first argument
        if args is None:
            args = 'self'
        elif type(args) is str:
            args = 'self, ' + args
        else:
            args = 'self, ' + ', '.join(args)

        # Make sure the code is indented correctly
        if type(code) is str:
            code = code.replace('\n', '\n\t')
        else:
            code = '\n\t'.join(code)
        code = code.replace('\n', '\n\t')

        # Create the actual method as a class-method.
        # You could, if desired, change this, so that normal methods are created instead.
        return ('\n\t@classmethod\n'
                '\tdef {name}({args}):\n\t\t'
                '{code}\n').format(name=name, args=args, code=code)

    def _gen_vars(self):
        V = self.graph.sampled_variables
        if len(V) > 0:
            return "return ['{}']".format("', '".join(sorted(list(V))))
        else:
            return 'return []'

    def _gen_all_keys(self):
        V = self.graph.not_observed_variables
        if len(V) > 0:
            return "return ['{}']".format("', '".join(sorted(list(V))))
        else:
            return 'return []'

    def _gen_ordered_vars(self):
        return None

    def _gen_cond_vars(self):
        vars = self.graph.cond_vars.difference(self.graph.if_vars)
        if len(vars) > 0:
            return "return ['{}']".format("', '".join(vars))
        else:
            return "return []"

    def _gen_cont_vars(self):
        vars = self.graph.cont_vars.difference(self.graph.if_vars)
        if len(vars) > 0:
            return "return ['{}']".format("', '".join(vars))
        else:
            return "return []"

    def _gen_if_vars(self):
        vars = [v for v in self.graph.if_vars if not v.startswith("f") and not v.startswith("cond_")]
        if len(vars) > 0:
            return "return ['{}']".format("', '".join(vars))
        else:
            return "return []"

    def _gen_if_functions(self):
        if Options.uniform_conditionals:
            vars = [v for v in self.graph.if_vars if v.startswith("f")]
            if len(vars) > 0:
                return "return ['{}']".format("', '".join(vars))
            else:
                return "return []"
        else:
            return None

    def _gen_disc_vars(self):
        vars = self.graph.disc_vars
        if len(vars) > 0:
            return "return ['{}']".format("', '".join(vars))
        else:
            return "return []"

    def _gen_prior_samples(self):
        graph = self.graph
        result = []
        for v in graph.sorted_var_list:
            code = graph.get_code_for_variable(v)
            if code.startswith('dist.'):
                result.append("dist_{v} = {code}".format(v=v, code=code))
                if graph.is_observed_variable(v):
                    result.append("{} = {}".format(v, graph.observed_values[v]))
                else:
                    result.append("{v} = dist_{v}.sample()".format(v=v))

            else:
                result.append("{} = {}".format(v, code))

        result += [
            "state = {}",
            "for _gv in self.gen_all_keys():",
            "\tstate[_gv] = locals()[_gv]",
            "return state  # dictionary"
        ]
        return '\n'.join(result)

    def _gen_pdf(self):
        graph = self.graph
        p_index = 10000
        result = []
        p_vars = []
        for v in graph.sorted_var_list:
            code = graph.get_code_for_variable(v)
            if code.startswith('dist.'):
                result.append("dist_{v} = {code}".format(v=v, code=code))
                if graph.is_observed_variable(v):
                    # result.append("{} = {}".format(v, graph.observed_values[v]))
                    s = "p{p_index} = dist_{v}.log_pdf({w})".format(p_index=p_index, v=v, w=graph.observed_values[v])
                else:
                    result.append("{v} = state['{v}']".format(v=v))
                    s = "p{p_index} = dist_{v}.log_pdf({v})".format(p_index=p_index, v=v)
                if v in graph.observed_conditions:
                    s += " if {} else 0".format(graph.observed_conditions[v])
                result.append(s)
                p_vars.append("p{p_index}".format(p_index=p_index))
                p_index += 1

            else:
                result.append("{} = {}".format(v, code))

        # Let's get rid of values, which are computed but never used
        #while len(result) > 0 and not result[-1].startswith('p'):
        #   del result[-1]

        result.append("_lcls = locals()")
        result.append("for key in state:\n\tif key in _lcls:\n\t\tstate[key] = _lcls[key]")
        if len(p_vars) > 0:
            result.append("logp = " + (" + ".join(p_vars)))
            result.append("return logp")
        else:
            result.append("return 0")
        return 'state', '\n'.join(result)
