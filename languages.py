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

    """An odd number of a's must be followed be an even number of b's.

    Choose tokens randomly, except at the last index.
    """

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield self.sample(n)

    def sample(self, n):
        """Running through a DFA."""
        state = "even_a"
        tokens = []
        for i in range(n):
            if state == "even_a":
                token = random.choice(["a", "b"])
                state = "odd_a" if token == "a" else "even_a"
            elif state == "odd_a":
                if i == n - 1:
                    token = "a"
                else:
                    token = random.choice(["a", "b"])
                state = "even_a" if token == "a" else "odd_b"
            elif state == "odd_b":
                token = "b"
                state = "even_b"
            else:  # even_b
                if i == n - 1:
                    token = "a"
                else:
                    token = random.choice(["a", "b"])
                state = "odd_a" if token == "a" else "odd_b"
            tokens.append(token)
        return "".join(tokens)

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

    # def generate(self, min_n, max_n):
    #     for n in range(min_n, max_n):
    #         cand = ''.join(random.choices("ab", k=n))
    #         while (not (self.valid(cand))): # one may use instead the := operator
    #             cand = ''.join(random.choices("ab", k=n))
    #         yield cand

class Tomita4:

    """All strings where three a's don't occur in a row."""

    def generate(self, min_n, max_n):
        # Don't really need to generate one per length here, but we do.
        for n in range(min_n, max_n):
            state = 0
            tokens = []
            for _ in range(n):
                token = "b" if state == 2 else random.choice(["a", "b"])
                tokens.append(token)
                if token == "b":
                    state = 0
                else:
                    state += 1
            yield "".join(tokens)



class Tomita5:

    """All strings w where #_a(w) and #_b(w) are even."""

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            if n % 2 == 0:
                yield self.sample(n)
                yield self.sample(n)


    def sample(self, n):
        """Where n must be divisible by 2."""
        k = n // 2
        k1 = random.randint(0, k)
        k2 = k - k1
        tokens = []
        tokens.extend("a" for _ in range(2 * k1))
        tokens.extend("b" for _ in range(2 * (k2)))
        random.shuffle(tokens)
        return "".join(tokens)

class Tomita6:

    """The number of a's and b's is the same mod 3.

    Can do this the dumb way since it should be relatively fast."""

    def sample(self, n):
        tokens = []
        diff = 0  # #(a) - #(b)
        for i in range(n):
            # if i == n - 3:
            #     if diff == 1:
            #         token = "a"
            #     elif diff == 2:
            #         token = "b"
            #     else:
            #         token = random.choice(["a", "b"])
            if i == n - 2:
                if diff == 1:
                    token = "a"
                elif diff == 2:
                    token = "b"
                else:
                    token = random.choice(["a", "b"])
            elif i == n - 1:
                if diff == 1:
                    token = "b"
                elif diff == 2:
                    token = "a"
                else:
                    raise NotImplementedError
            else:
                token = random.choice(["a", "b"])
            
            if token == "a":
                diff = (diff + 1) % 3
            else:
                diff = (diff - 1) % 3
            tokens.append(token)
        
        return "".join(tokens)
    
    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            if n == 1:
                continue  # Not possible.
            yield self.sample(n)
            
    # # ugly, replace with regex
    # def valid(self, str):
    #     count_a , count_b = 0, 0
    #     for c in str:
    #         if (c == 'b'):
    #             count_b += 1
    #         else:
    #             count_a += 1
    #     dif = count_b - count_a
    #     # return (dif % 3 == 0 and dif >= 0)
    #     return (dif % 3 == 0)

    # def generate(self, min_n, max_n):
    #     for n in range(min_n, max_n):
    #         cand = ''.join(random.choices("ab", k=n))
    #         while (not (self.valid(cand))): # one may use instead the := operator
    #             cand = ''.join(random.choices("ab", k=n))
    #         yield cand

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
