import random

class AbstarGenerator:

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "ab" * n

class AbbastarGenerator:

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "abba" * n

# class EvenaGenerator:
#
#     # reg expression is (b + ab^*ab^*)^*
#     def generate(self, min_n, max_n):
#         for n in range(min_n, max_n):
#             p1 = random.random()
#             if (p1 < 0.5):
#                 yield "b" * n
#             else:
#                 k = random.int(1, n // 4) # outer repetitions
#                 count_of_bs = n - 2*k
#                 count1 = random.uniform
