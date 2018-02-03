# Understanding how to use metaclassses
# potentially useful for when creating program() class and interaction between leapfrog
# and potential class. Or.... get rid of potential class and have program class directily interact
# with the HMC sampler.

# a metaclass solution to replace __init__
from inspect import Parameter, Signature

def make_sig(names):
    return Signature(
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
            for name in names)

class Structure:
    __signiture__ = make_sig([])
    def __init__(self, *args, **kwargs):
        bound = self.__signiture__.bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            setattr(self, name, value)
class StructMeta(type):
    def __new__(cls, name, bases, clsdict):
        clsobj = super().__new__(cls,name,bases,clsdict)
        sig = make_sig(clsobj.fields) # reads _params attribute and makes a proper signiture oout of it
        setattr(clsobj, '__signature__',sig)
        return clsobj
class Structure(metaclass=StructMeta):
    _params = []
    def __init__(self, *args, **kwargs):
        bound = self.__signiture__.bind(*args, **kwargs)
        for name, val in bound.arguments.items():
            setattr(self,name,str)