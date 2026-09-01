"""Deterministic seeded PRNG (mulberry32) — every artifact reproducible from a seed."""
def mulberry32(seed: int):
    state = seed & 0xFFFFFFFF
    def nxt():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        r = (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        r = r ^ t
        return (r ^ (r >> 14)) & 0xFFFFFFFF
    return nxt

class Rng:
    """Small deterministic random source with the usual helpers."""
    def __init__(self, seed: int):
        self._n = mulberry32(seed)
        self.seed = seed

    def below(self, n: int) -> int:
        return self._n() % n

    def range(self, lo: int, hi: int) -> int:  # inclusive
        return lo + self.below(hi - lo + 1)

    def chance(self, p: float) -> bool:
        return self.below(10_000) < int(p * 10_000)

    def pick(self, seq):
        return seq[self.below(len(seq))]

    def shuffle(self, seq):
        seq = list(seq)
        for i in range(len(seq) - 1, 0, -1):
            j = self.below(i + 1)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    def sample(self, seq, k):
        return self.shuffle(seq)[:k]
