import inspect


class Singleton(type):
    """A Singleton metaclass ensures at most one instance of a class exists.

    Args:
        check_args (optional, bool): If `True` when passed as a kwd (see
            example below) then it verifies that each call to the classes
            `__init__` method has the same set of arguments. If not, a
            `ValueError` is raised. Default: `False`.

    Example:

        >>> class Foo(metaclass=Singleton, check_args=True):
        ...     def __init__(self, val):
        ...         self.val = val
        ...
        >>> a = Foo(val=6)
        >>> b = Foo(6)
        >>> a is b
        True
        >>> Foo(val=8)
        Traceback (most recent call last):
            ...
        ValueError: Foo instance already exists but previously initialised
        differently...
    """
    def __new__(metacls, name, bases, namespace, **kwds):
        return type.__new__(metacls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwds):
        cls.__check_args = 'check_args' in kwds
        cls.__instance = None

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__call__(*args, **kwargs)
            if cls.__check_args:
                cls.__args = _get_init_arguments(cls, *args, **kwargs)
        elif cls.__check_args:
            err_msg = (cls.__name__ + ' instance already exists but '
                       'previously initialised differently - '
                       'instance: %s, call: %s')
            args = _get_init_arguments(cls, *args, **kwargs)
            if args != cls.__args:
                raise ValueError(err_msg % (cls.__args, args))

        return cls.__instance


def _get_init_arguments(cls, *args, **kwargs):
    """Returns an OrderedDict of args passed to cls.__init__ given [kw]args."""
    init_args = inspect.signature(cls.__init__)
    bound_args = init_args.bind(None, *args, **kwargs)
    bound_args.apply_defaults()
    arg_dict = bound_args.arguments
    del arg_dict['self']
    return arg_dict
