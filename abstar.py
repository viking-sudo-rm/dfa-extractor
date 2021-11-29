class AbstarGenerator:

    def generate(self, min_n, max_n):
        for n in range(min_n, max_n):
            yield "ab" * n
