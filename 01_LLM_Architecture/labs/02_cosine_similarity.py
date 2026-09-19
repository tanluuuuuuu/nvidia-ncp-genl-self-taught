import numpy as np
import unittest

def cosine_similarity(a, B):
    # a is a 1-D embedding of shape (d,). B is a stack of embeddings, shape (m, d):
    # one row per candidate. Return a 1-D array of shape (m,): the cosine
    # similarity between a and each row of B.
    #
    # Cosine similarity is the dot product divided by both vector lengths, so
    # only the direction matters, not how long the vectors are.
    res = np.dot(a, B.T) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1))
    return res.flatten()


class TestCosineSimilarity(unittest.TestCase):
    """Pin down the contract: direction-only scores, one per row of B."""

    def assert_similar(self, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)

    def test_identical_vector_gives_one(self):
        self.assert_similar(
            cosine_similarity(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0, 0.0]])),
            [1.0],
        )

    def test_magnitude_does_not_matter(self):
        # Same direction, very different lengths -> still 1.0.
        a = np.array([1.0, 2.0, 3.0])
        B = np.array([[2.0, 4.0, 6.0], [0.1, 0.2, 0.3]])
        self.assert_similar(cosine_similarity(a, B), [1.0, 1.0])

    def test_opposite_direction_gives_minus_one(self):
        self.assert_similar(
            cosine_similarity(np.array([1.0, 0.0]), np.array([[-2.0, 0.0]])),
            [-1.0],
        )

    def test_orthogonal_vectors_give_zero(self):
        self.assert_similar(
            cosine_similarity(np.array([3.0, 0.0]), np.array([[0.0, 5.0]])),
            [0.0],
        )

    def test_hand_computed_known_value(self):
        # a=[1,2,3], b=[4,5,6]: dot=4+10+18=32, |a|^2=14, |b|^2=77
        # -> 32 / sqrt(14*77) ~ 0.9746318
        a = np.array([1.0, 2.0, 3.0])
        B = np.array([[4.0, 5.0, 6.0]])
        self.assert_similar(cosine_similarity(a, B), [32.0 / np.sqrt(14.0 * 77.0)])

    def test_one_score_per_candidate_row(self):
        a = np.array([1.0, 0.0])
        B = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.assert_similar(cosine_similarity(a, B), [1.0, 0.0, -1.0])

    def test_output_is_1d_of_shape_m(self):
        a = np.array([1.0, 2.0])
        B = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        result = cosine_similarity(a, B)
        self.assertEqual(np.shape(result), (3,))


if __name__ == "__main__":
    unittest.main()
