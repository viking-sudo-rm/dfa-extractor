from unittest import TestCase

from languages import *


class TestLanguages(TestCase):

    def test_tomita3(self):
        tomita3 = Tomita3()
        words = list(tomita3.generate(0, 10))
        assert all(tomita3.valid(w) for w in words)

    def test_tomita4(self):
        tomita4 = Tomita4()
        words = list(tomita4.generate(0, 10))
        assert not any("aaa" in w for w in words)
    
    def test_tomita5(self):
        tomita5 = Tomita5()

        # Make sure samples are the right length.
        samp = tomita5.sample(10)
        assert len(samp) == 10
        assert samp.count("a") % 2 == 0
        assert samp.count("b") % 2 == 0

        # Make sure the sample holds for generating multiple samples.
        words = list(tomita5.generate(0, 10))
        assert len(words) == 10
        assert all(w.count("a") % 2 == 0 for w in words)
        assert all(w.count("b") % 2 == 0 for w in words)
