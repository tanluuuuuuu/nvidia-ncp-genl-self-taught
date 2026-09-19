import unittest

def bpe_merge(tokens, pair):
    # tokens is a list of strings (the current symbols). pair is a tuple (a, b).
    # Walk left to right. Every time symbol a is immediately followed by symbol b,
    # fuse the two into one new symbol a + b, then continue PAST the fused pair.
    #
    # Build a new list; do not merge a symbol you just produced.
    new_tokens = []
    i = 0
    while i < len(tokens) - 1:
        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    new_tokens.extend(tokens[i:])
    return new_tokens

class TestBpeMerge(unittest.TestCase):
    """Pin down the contract: left-to-right scan, fuse pair, skip PAST it."""

    def test_merges_single_occurrence_of_pair(self):
        self.assertEqual(
            bpe_merge(["h", "e", "l", "l", "o"], ("l", "l")),
            ["h", "e", "ll", "o"],
        )

    def test_merges_all_non_overlapping_occurrences(self):
        self.assertEqual(
            bpe_merge(["a", "b", "a", "b"], ("a", "b")),
            ["ab", "ab"],
        )

    def test_skips_past_fused_pair_no_remerge(self):
        # aaa + (a,a): merge the first two, the leftover lone 'a' must survive.
        self.assertEqual(bpe_merge(["a", "a", "a"], ("a", "a")), ["aa", "a"])

    def test_merges_pair_ending_at_last_index(self):
        self.assertEqual(bpe_merge(["x", "a", "b"], ("a", "b")), ["x", "ab"])

    def test_last_symbol_matching_first_pair_element_is_left_alone(self):
        # 'a' at the end has no successor: no merge, and no IndexError.
        self.assertEqual(bpe_merge(["b", "a"], ("a", "a")), ["b", "a"])

    def test_no_merge_when_pair_absent(self):
        self.assertEqual(bpe_merge(["a", "b", "c"], ("x", "y")), ["a", "b", "c"])

    def test_no_merge_for_non_adjacent_symbols(self):
        self.assertEqual(bpe_merge(["a", "x", "a"], ("a", "a")), ["a", "x", "a"])

    def test_empty_token_list(self):
        self.assertEqual(bpe_merge([], ("a", "b")), [])

    def test_does_not_mutate_input(self):
        tokens = ["a", "a", "a"]
        bpe_merge(tokens, ("a", "a"))
        self.assertEqual(tokens, ["a", "a", "a"])


if __name__ == "__main__":
    unittest.main()