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
    """Choose half the samples to contain strings positive at max length. The other half are random."""

    def __init__(self, lang):
        self.lang = lang

    def sample(self, n_samples: int, length: int):
        assert n_samples > 1, "For balanced sampler, need more than 1 sample."
        n_half = n_samples // 2
        for _ in range(n_half):
            yield self.lang.sample(length)
            yield random_string(length)


class RandomSampler(Sampler):
    """Return 2 * length random samples."""
    def sample(self, n_samples: int, length: int):
        for _ in range(n_samples):
            yield random_string(length)

class TestSampler(Sampler):
    """Return half positive and half negative examples of random length in {0, `length`}"""

    def __init__(self, lang):
        self.lang = lang

    def sample(self, n_samples: int, length: int):
        for i in range(n_samples):
            if (i < n_samples // 2):
                yield self.lang.sample(random.randint(0, length))
            else:
                yield random_string(random.randint(0, length))
