from unittest import TestCase
import re

from languages import *


class TestLanguages(TestCase):

    def test_tomita3(self):
        random.seed(2)
        tomita3 = Tomita3()
        words = list(tomita3.generate(0, 30))
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

    def test_tomita6(self):
        tomita6 = Tomita6()
        words = list(tomita6.generate(0, 10))
        a_counts = [w.count("a") % 3 for w in words]
        b_counts = [w.count("b") % 3 for w in words]
        self.assertEqual(a_counts, b_counts)

    def test_tomita7(self):
        tomita7 = Tomita7()
        words = list(tomita7.generate(0, 10))
        assert all(re.match(r"a*b*a*b*", w) for w in words)

    def test_tomita1_trace(self):
        tomita1 = Tomita1()
        aab = tomita1.trace_acceptance("aab")
        assert aab == [1, 1, 1, 0]

    def test_tomita2_trace(self):
        tomita2 = Tomita2()
        abab = tomita2.trace_acceptance("abab")
        assert abab == [1, 0, 1, 0, 1]

    def test_tomita3_trace(self):
        tomita3 = Tomita3()
        babbabbb = tomita3.trace_acceptance("babbabbb")
        assert babbabbb == [1, 1, 1, 0, 1, 1, 0, 1, 0]
    
    def test_tomita4_trace(self):
        tomita4 = Tomita4()
        baabaaa = tomita4.trace_acceptance("baabaaa")
        assert baabaaa == [1, 1, 1, 1, 1, 1, 1, 0]
    
    def test_tomita5_trace(self):
        tomita5 = Tomita5()
        bababb = tomita5.trace_acceptance("bababb")
        assert bababb == [1, 0, 0, 0, 1, 0, 1]
    
    def test_tomita6_trace(self):
        tomita6 = Tomita6()
        abaabb = tomita6.trace_acceptance("abaabb")
        assert abaabb == [1, 0, 1, 0, 0, 0, 1]
    
    def test_tomita7_trace(self):
        tomita7 = Tomita7()
        bbabaab = tomita7.trace_acceptance("bbabaab")
        assert bbabaab == [1, 1, 1, 1, 1, 1, 1, 0]
    