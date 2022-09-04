from typing import Iterable, Callable, Tuple


class Material:
    def __init__(self, name: str, args: Iterable[str], **kwargs):
        """
        Semiconductor compound / alloy class.

        Parameters
        ----------
        name : str
            Material name.
        args : Iterable[str]
            Arguments of material parameters. "Arguments" (or `args`) here and
            elsewhere in the class documentation refer to parameters that are
            not defined by the material, e.g., alloy composition or
            temperature. These appear as arguments in functions defining
            material "parameters" (e.g., bandgap, electron mobility, etc),
            hence the name.
        **kwargs
            Default values of `args` (optional).

        """
        self.name = name
        # dictionaries, name format: key2value (p -- parameter, a -- argument)
        self.p2functions = {}
        self.a2params = {arg: [] for arg in args}  # arg -> params that use arg
        self.p2args = {}                           # param -> function args
        self.a2defaults = {}                       # default values of args
        if kwargs:
            self.set_defaults(**kwargs)

    def get_arguments(self) -> Tuple[str]:
        "Return a tuple of arguments required for calculations."
        return tuple(self.a2params.keys())

    def get_parameters(self) -> Tuple[str]:
        "Return a tuple of parameters defined by material."
        return tuple(self.p2functions.keys())

    def set_defaults(self, **kwargs):
        """
        Set default values of arguments (independent variables, such as alloy
        composition or temperature).
        """
        args = self.get_arguments()
        for k, v in kwargs.items():
            if k not in args:
                raise KeyError(f'Unknown parameter {k}')
            self.a2defaults[k] = v

    def set_param(self, name: str, func: Callable):
        """
        Add material parameter `name` as a function `func`.
        """
        self.remove_param(name)
        self.p2functions[name] = func
        args = func.__code__.co_varnames[:func.__code__.co_argcount]
        assert all(a in self.a2params.keys() for a in args), 'Unknown argument'
        self.p2args[name] = args
        for arg in args:
            self.a2params[arg].append(name)

    def set_params(self, **kwargs):
        for name, func in kwargs.items():
            self.set_param(name, func)

    def remove_param(self, name: str) -> bool:
        """
        Remove parameter `name` from material. Returns `False` if parameter
        does not exist, `True` if successfully removed parameter.
        """
        if name not in self.p2functions:
            return False
        self.p2functions.pop(name)
        args = self.p2args.pop(name)
        for arg in args:
            self.a2params[arg].remove(name)
        return True

    def calculate(self, param: str, **kwargs):
        """
        Calculate material parameter `param` using optional keyword arguments.
        Return type depends on the original function, normally `numpy.ndarray`.
        """
        if param == 'C_dop':
            return self._calculate_Cdop(**kwargs)
        required_args = self.p2args[param]
        kw = {}  # will pass those to the function
        for arg in required_args:
            if arg not in kwargs:  # try to use defaults
                assert arg in self.a2defaults, f'Error: {arg} not specified'
                kw[arg] = self.a2defaults[arg]
            else:
                kw[arg] = kwargs[arg]
        return self.p2functions[param](**kw)

    def _calculate_Cdop(self, **kwargs):
        return self.calculate('Nd', **kwargs) - self.calculate('Na', **kwargs)

    def calculate_all(self, **kwargs) -> dict:
        """
        Calculate every material parameter for which `kwargs` are sufficient.
        """
        rv = {}
        for param in self.p2functions.keys():
            if not all(arg in kwargs for arg in self.p2args[param]):
                continue
            rv[param] = self.calculate(param, **kwargs)
        return rv
