import random
from collections import defaultdict

START, END = "<START>", "<END>"


class BigramLM:
    """First-order autoregressive model: P(X_t | X_{t-1})."""

    def __init__(self):
        # counts[prev][next] = how often `next` followed `prev`
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probs = {}

    # 1 + 2: read training sentences and count transitions
    def fit(self, sentences):
        for sentence in sentences:
            tokens = [START] + sentence + [END]
            for prev, nxt in zip(tokens, tokens[1:]):
                self.counts[prev][nxt] += 1

        # 3: turn counts into conditional probabilities
        self.probs = {}
        for prev, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[prev] = {tok: c / total for tok, c in nexts.items()}

    # 4: show P(next | prev) for a given previous token
    def show(self, prev):
        if prev not in self.probs:
            print(f"Unknown token: {prev!r}")
            return
        print(f"P(next | {prev!r}):")
        for tok, p in sorted(self.probs[prev].items(), key=lambda kv: -kv[1]):
            print(f"  {tok:<10} {p:.3f}")

    # 5: most probable next token
    def predict(self, prev):
        dist = self.probs.get(prev)
        if not dist:
            return None
        return max(dist, key=dist.get)

    # sample one token from P(. | prev)
    def sample_next(self, prev):
        dist = self.probs[prev]
        return random.choices(list(dist), weights=list(dist.values()), k=1)[0]

    # 6 + 7: generate by repeated sampling, stop at <END>
    def generate(self, max_len=30):
        prev, out = START, []
        for _ in range(max_len):
            nxt = self.sample_next(prev)
            if nxt == END:
                break
            out.append(nxt)
            prev = nxt
        return " ".join(out)


if __name__ == "__main__":
    data = [
        "the cat sat on the mat".split(),
        "the dog sat on the rug".split(),
        "the cat chased the dog".split(),
        "a dog chased a cat".split(),
    ]

    lm = BigramLM()
    lm.fit(data)

    lm.show("the")
    print("\nMost likely after 'the':", lm.predict("the"))
    print("Most likely after 'sat':", lm.predict("sat"))

    print("\nGenerated sentences:")
    for _ in range(5):
        print(" ", lm.generate())
