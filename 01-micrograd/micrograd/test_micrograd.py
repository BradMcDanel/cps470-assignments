"""
Tests for the MicroGrad implementation.

Run with: python test_micrograd.py

All tests should pass when your implementation is complete.
"""

from micrograd import Value


def test_add():
    """Test addition and its gradient."""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()

    assert c.data == 5.0, f"Expected 5.0, got {c.data}"
    assert a.grad == 1.0, f"Expected a.grad=1.0, got {a.grad}"
    assert b.grad == 1.0, f"Expected b.grad=1.0, got {b.grad}"
    print("test_add passed!")


def test_mul():
    """Test multiplication and its gradient."""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()

    assert c.data == 6.0, f"Expected 6.0, got {c.data}"
    assert a.grad == 3.0, f"Expected a.grad=3.0, got {a.grad}"
    assert b.grad == 2.0, f"Expected b.grad=2.0, got {b.grad}"
    print("test_mul passed!")


def test_pow():
    """Test power and its gradient."""
    a = Value(3.0)
    b = a ** 2
    b.backward()

    assert b.data == 9.0, f"Expected 9.0, got {b.data}"
    assert a.grad == 6.0, f"Expected a.grad=6.0, got {a.grad}"
    print("test_pow passed!")


def test_neg():
    """Test negation."""
    a = Value(3.0)
    b = -a
    b.backward()

    assert b.data == -3.0, f"Expected -3.0, got {b.data}"
    assert a.grad == -1.0, f"Expected a.grad=-1.0, got {a.grad}"
    print("test_neg passed!")


def test_sub():
    """Test subtraction."""
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    c.backward()

    assert c.data == 2.0, f"Expected 2.0, got {c.data}"
    assert a.grad == 1.0, f"Expected a.grad=1.0, got {a.grad}"
    assert b.grad == -1.0, f"Expected b.grad=-1.0, got {b.grad}"
    print("test_sub passed!")


def test_div():
    """Test division."""
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    c.backward()

    assert c.data == 3.0, f"Expected 3.0, got {c.data}"
    assert abs(a.grad - 0.5) < 1e-6, f"Expected a.grad=0.5, got {a.grad}"
    assert abs(b.grad - (-1.5)) < 1e-6, f"Expected b.grad=-1.5, got {b.grad}"
    print("test_div passed!")


def test_relu():
    """Test ReLU activation."""
    # Positive input
    a = Value(3.0)
    b = a.relu()
    b.backward()
    assert b.data == 3.0, f"Expected 3.0, got {b.data}"
    assert a.grad == 1.0, f"Expected a.grad=1.0, got {a.grad}"

    # Negative input
    c = Value(-3.0)
    d = c.relu()
    d.backward()
    assert d.data == 0.0, f"Expected 0.0, got {d.data}"
    assert c.grad == 0.0, f"Expected c.grad=0.0, got {c.grad}"
    print("test_relu passed!")


def test_chain():
    """Test a chain of operations."""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b        # c = 6
    d = c + a        # d = 8
    e = d * c        # e = 48
    e.backward()

    # e = (a*b + a) * (a*b) = a^2*b^2 + a^2*b
    # de/da = 2ab^2 + 2ab = 2*2*9 + 2*2*3 = 36 + 12 = 48
    assert e.data == 48.0, f"Expected 48.0, got {e.data}"
    assert a.grad == 48.0, f"Expected a.grad=48.0, got {a.grad}"
    # de/db = 2a^2*b + a^2 = 2*4*3 + 4 = 24 + 4 = 28
    assert b.grad == 28.0, f"Expected b.grad=28.0, got {b.grad}"
    print("test_chain passed!")


def test_reuse():
    """Test that a value can be used multiple times in the graph."""
    a = Value(3.0)
    b = a + a  # Same value used twice
    b.backward()

    assert b.data == 6.0, f"Expected 6.0, got {b.data}"
    assert a.grad == 2.0, f"Expected a.grad=2.0, got {a.grad}"
    print("test_reuse passed!")


def test_scalar_ops():
    """Test operations with Python scalars."""
    a = Value(2.0)
    b = a + 3      # Value + scalar
    c = 3 + a      # scalar + Value
    d = a * 4      # Value * scalar
    e = 4 * a      # scalar * Value

    assert b.data == 5.0, f"Expected 5.0, got {b.data}"
    assert c.data == 5.0, f"Expected 5.0, got {c.data}"
    assert d.data == 8.0, f"Expected 8.0, got {d.data}"
    assert e.data == 8.0, f"Expected 8.0, got {e.data}"
    print("test_scalar_ops passed!")


def test_neuron():
    """Test a simple neuron computation."""
    # Inputs
    x1 = Value(2.0)
    x2 = Value(0.0)
    # Weights
    w1 = Value(-3.0)
    w2 = Value(1.0)
    # Bias
    b = Value(6.8813735870195432)
    # Neuron: relu(x1*w1 + x2*w2 + b)
    n = (x1 * w1 + x2 * w2 + b).relu()
    n.backward()

    assert abs(n.data - 0.8813735870195432) < 1e-9
    assert abs(x1.grad - (-3.0)) < 1e-9
    assert abs(w1.grad - 2.0) < 1e-9
    print("test_neuron passed!")


def test_compare_pytorch():
    """Compare gradients with PyTorch."""
    import torch

    # MicroGrad computation
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    d = a + c
    e = d ** 2
    f = e.relu()
    f.backward()

    # PyTorch computation
    a_pt = torch.tensor(2.0, requires_grad=True)
    b_pt = torch.tensor(3.0, requires_grad=True)
    c_pt = a_pt * b_pt
    d_pt = a_pt + c_pt
    e_pt = d_pt ** 2
    f_pt = torch.relu(e_pt)
    f_pt.backward()

    # Compare
    assert abs(f.data - f_pt.item()) < 1e-6, f"Forward pass mismatch"
    assert abs(a.grad - a_pt.grad.item()) < 1e-6, f"Gradient mismatch for a"
    assert abs(b.grad - b_pt.grad.item()) < 1e-6, f"Gradient mismatch for b"
    print("test_compare_pytorch passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running MicroGrad Tests")
    print("=" * 50)

    tests = [
        test_add,
        test_mul,
        test_pow,
        test_neg,
        test_sub,
        test_div,
        test_relu,
        test_chain,
        test_reuse,
        test_scalar_ops,
        test_neuron,
        test_compare_pytorch,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"{test.__name__} ERROR: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
