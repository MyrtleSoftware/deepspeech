import pytest

from deepspeech.utils.singleton import Singleton


class Foo(metaclass=Singleton):
    def __init__(self, val):
        self.val = val


class Bar(metaclass=Singleton, check_args=True):
    def __init__(self, val):
        self.val = val


def test_singleton_created_once():
    a = Foo(3)
    b = Foo(3)
    assert a is b


def test_singleton_check_args_matching_ok():
    a = Bar(5)
    b = Bar(val=5)
    assert a is b


def test_singleton_check_args_nonmatching_raises_value_error():
    Bar(val=5)
    with pytest.raises(ValueError):
        Bar(6)
