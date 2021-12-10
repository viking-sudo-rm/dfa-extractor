import random

class Tomita1:

    """The language a^*."""

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "a" * n

    def trace_acceptance(self, string):
        state = 1
        states = []
        for token in string:
            states.append(state)
            if token == "b":
                state = 0
        states.append(state)
        return states


class Tomita2:

    """The language (ab)^*."""

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "ab" * n

    def trace_acceptance(self, string):
        state = "b"
        states = []
        for token in string:
            states.append(state)
            if state == "b" and token == "a":
                state = "a"
            elif state == "a" and token == "b":
                state = "b"
            else:
                state = "!"
        states.append(state)
        return [int(s == "b") for s in states]


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

    def trace_acceptance(self, string):
        status = []
        for idx in range(len(string) + 1):
            is_valid = self.valid(string[:idx])
            status.append(int(is_valid))
        return status

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

    def trace_acceptance(self, string):
        state = 0
        states = []
        for token in string:
            states.append(state)
            if state == -1:
                continue

            if token == "b":
                state = 0
            elif token == "a" and state == 2:
                state = -1
            else:
                state += 1
        states.append(state)
        return [int(s != -1) for s in states]


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

    def trace_acceptance(self, string):
        statuses = []
        even_a = True
        even_b = True
        for token in string:
            status = int(even_a and even_b)
            statuses.append(status)

            if token == "a":
                even_a = not even_a
            else:
                even_b = not even_b

        status = int(even_a and even_b)
        statuses.append(status)
        return statuses


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

    def trace_acceptance(self, string):
        status = []
        for idx in range(len(string) + 1):
            is_valid = self.valid(string[:idx])
            status.append(int(is_valid))
        return status

    def valid(self, str):
        count_a , count_b = 0, 0
        for c in str:
            if (c == 'b'):
                count_b += 1
            else:
                count_a += 1
        dif = count_b - count_a
        # return (dif % 3 == 0 and dif >= 0)
        return (dif % 3 == 0)

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

    def trace_acceptance(self, string):
        states = []
        state = "b1"
        for token in string:
            states.append(state)
            if state == "b1" and token == "a":
                state = "a1"
            elif state == "a1" and token == "b":
                state = "b2"
            elif state == "b2" and token == "a":
                state = "a2"
            elif state == "a2" and token == "b":
                state = "!"
        states.append(state)
        return [int(s != "!") for s in states]

class AbbastarGenerator:

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "abba" * n
