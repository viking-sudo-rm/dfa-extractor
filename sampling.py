"""Classes to sample strings from a formal language."""

from abc import ABCMeta, abstractmethod
import random


def random_string(n: int) -> str:
    return "".join(random.choice(["a", "b"]) for _ in range(n))


class Sampler(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, min_n: int, max_n: int):
        return NotImplemented


class BalancedSampler(Sampler):
    """Return `n_samples` random strings, plus a sample with a valid prefix of every length up to `length`."""

    def __init__(self, lang):
        self.lang = lang

    def sample(self, n_samples: int, length: int):
        for n in range(length):
            string = self.lang.sample(n)
            string += random_string(length - n)
            yield string
        for n in range(n_samples):
            string = self.lang.sample(n)
            yield random_string(length)


class RandomSampler(Sampler):
    """Return 2 * length random samples."""
    def sample(self, n_samples: int, length: int):
        for _ in range(n_samples):
            yield random_string(length)
