import random

class Tomita1:

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "a" * n

class Tomita2:

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "ab" * n

class Tomita3:

    # ugly, replace with regex
    def valid(self, str):
        count_a, count_b = 0, 0
        flag_a, flag_b = False, False
        for c in str:
            if (c == 'a' and flag_b):
                if (count_b % 2 == 1 and count_a % 2 == 1):
                    return False
                flag_a = True
                flag_b = False
                count_a = 1
                count_b = 0
            elif (c == 'a' and not flag_b):
                count_a += 1
            else: # c == 'b'
                flag_b = True
                flag_a = False
                count_b += 1
        return not (count_b % 2 == 1 and count_a % 2 == 1)

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            cand = ''.join(random.choices("ab", k=n))
            while (not (self.valid(cand))): # one may use instead the := operator
                cand = ''.join(random.choices("ab", k=n))
            yield cand

class Tomita4:

    # ugly, replace with regex
    def valid(self, str):
        count = 0
        for c in str:
            if (c == 'b'):
                count += 1
                if (count == 3):
                    return False
            else:
                count = 0
        return True

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            cand = ''.join(random.choices("ab", k=n))
            while (not (self.valid(cand))): # one may use instead the := operator
                cand = ''.join(random.choices("ab", k=n))
            yield cand

class Tomita5:

    # ugly, replace with regex
    def valid(self, str):
        count_a , count_b = 0, 0
        for c in str:
            if (c == 'b'):
                count_b += 1
            else:
                count_a += 1
        return (count_a % 2 == 0 and count_b % 2 == 0)

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            if (n % 2 == 1):
                continue
            cand = ''.join(random.choices("ab", k=n))
            while (not (self.valid(cand))): # one may use instead the := operator
                cand = ''.join(random.choices("ab", k=n))
            yield cand

class Tomita6:

    # ugly, replace with regex
    def valid(self, str):
        count_a , count_b = 0, 0
        for c in str:
            if (c == 'b'):
                count_b += 1
            else:
                count_a += 1
        dif = count_b - count_a
        return (dif % 3 == 0 and dif >= 0)

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            cand = ''.join(random.choices("ab", k=n))
            while (not (self.valid(cand))): # one may use instead the := operator
                cand = ''.join(random.choices("ab", k=n))
            yield cand

class Tomita7:

    def generate(self, min_n, max_n):
        generator = "baba"
        for n in range(min_n, max_n):
            length = [0] * 5
            for i in range(1, 5):
                length[i] = random.randint(0, n - length[i-1])
            sub = random.sample(length[1:], 4)
            final = ""
            for i in range(4):
                final += generator[i] * sub[i]
            yield final

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
#                 k = random.randint(1, n // 4) # outer repetitions
#                 count_of_bs = n - 2*k
#                 count1 = random.uniform
