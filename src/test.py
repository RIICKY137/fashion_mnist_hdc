import numpy as np
from hdc import generate_random_hypervector, bind, bundle, similarity

def test_generate_random_hv():
    hv = generate_random_hypervector()
    assert isinstance(hv, np.ndarray), "Hypervector should be a numpy array"
    assert set(np.unique(hv)).issubset({0, 1}), "Values should be 0/1"
    print("[PASS] generate_random_hypervector produces binary HV.")

def test_bind_inverse():
    hv1 = generate_random_hypervector()
    hv2 = generate_random_hypervector()

    hv3 = bind(hv1, hv2)
    recovered = bind(hv3, hv1)

    sim = similarity(recovered, hv2)
    print(f"Similarity (bind inverse test): {sim:.4f}")

    assert sim > 0.95, "Binding XOR inverse should recover the vector"
    print("[PASS] bind() XOR property works correctly.")

def test_bundle_majority():
    hv1 = np.array([0, 1, 1, 0, 1])
    hv2 = np.array([1, 0, 1, 0, 1])
    hv3 = np.array([1, 1, 0, 0, 1])

    result = bundle([hv1, hv2, hv3])
    expected = np.array([1, 1, 1, 0, 1])

    assert np.array_equal(result, expected), "Bundling majority vote incorrect"
    print("[PASS] bundle() majority vote works.")

def test_similarity():
    hv1 = np.array([0, 1, 1, 0])
    hv2 = np.array([0, 1, 1, 0])
    hv3 = np.array([1, 0, 0, 1])

    assert similarity(hv1, hv2) == 1.0, "Similarity of identical HV should be 1"
    assert similarity(hv1, hv3) == 0.0, "Similarity of completely different HV should be 0"
    print("[PASS] similarity() correct for simple cases.")

if __name__ == "__main__":
    test_generate_random_hv()
    test_bind_inverse()
    test_bundle_majority()
    test_similarity()
    print("\n✅ All HDC tests passed.")
