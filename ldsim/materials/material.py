from typing import Iterable, Callable, Dict


class Material:
    def __init__(self, name: str, args: Iterable[str]):
        """
        Semiconductor compound / alloy class.

        Parameters
        ----------
        name: str
            Material name.
        args: Iterable[str]
            Arguments of material parameters.

        """
        self.name = name
        # dictionaries: name = values, comment = keys
        self.functions = {}                      # parameters
        self.params = {arg: [] for arg in args}  # arguments
        self.args = {}                           # parameters

    def set_param(self, name: str, func: Callable):
        """
        Add material parameter `name` as a function `func`.
        """
        self.remove_param(name)
        self.functions[name] = func
        args = func.__code__.co_varnames[:func.__code__.co_argcount]
        assert all(a in self.params.keys() for a in args), 'Unknown argument'
        self.args[name] = args
        for arg in args:
            self.params[arg].append(name)

    def set_params(self, params: Dict[str, Callable]):
        for name, func in params.items():
            self.set_param(name, func)

    def remove_param(self, name: str) -> bool:
        """
        Remove parameter `name` from material. Returns `False` if parameter
        does not exist, `True` if successfully removed parameter.
        """
        if name not in self.functions:
            return False
        self.functions.pop(name)
        args = self.args.pop(name)
        for arg in args:
            self.params[arg].remove(name)
        return True

    def calculate(self, param: str, **kwargs):
        """
        Calculate material parameter `param` using optional keyword arguments.
        Return type depends on the original function, normally `numpy.ndarray`.
        """
        args = self.args[param]
        filtered = {arg: kwargs[arg] for arg in args}
        return self.functions[param](**filtered)

    def calculate_all(self, **kwargs) -> dict:
        """
        Calculate every material parameter for which `kwargs` are sufficient.
        """
        rv = {}
        for param in self.functions.keys():
            if not all(arg in kwargs for arg in self.args[param]):
                continue
            rv[param] = self.calculate(param, **kwargs)
        return rv
