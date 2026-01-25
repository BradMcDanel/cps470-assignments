"""
MicroGrad: A tiny autograd engine.

This module implements automatic differentiation (autograd) from scratch.
Your task is to complete the Value class so that gradients are computed
correctly via backpropagation.
"""


class Value:
    """
    A wrapper around a scalar value that tracks gradients.

    Attributes:
        data (float): The actual numerical value
        grad (float): The gradient of the final output with respect to this value
        _backward (callable): Function to compute gradients for inputs
        _prev (set): The set of Value objects that were inputs to this operation
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        # Internal variables for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # The operation that produced this node (for debugging)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Addition: self + other  (PROVIDED AS EXAMPLE)

        Forward pass: out.data = self.data + other.data
        Backward pass: Both inputs receive the upstream gradient unchanged
                       (since d(a+b)/da = 1 and d(a+b)/db = 1)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # The _backward function computes gradients for the inputs.
        # out.grad contains the gradient flowing back from downstream.
        # We accumulate (+=) because a value might be used multiple times.
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplication: self * other

        Forward pass: out.data = self.data * other.data
        Backward pass: d(a*b)/da = b, d(a*b)/db = a
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # TODO: Accumulate gradients for self and other
            # Hint: Use the chain rule. What is d(out)/d(self)? d(out)/d(other)?
            pass

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Power: self ** other (where other is a constant, not a Value)

        Forward pass: out.data = self.data ** other
        Backward pass: d(a^n)/da = n * a^(n-1)
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # TODO: Accumulate gradient for self
            # Hint: Power rule from calculus
            pass

        out._backward = _backward
        return out

    def __neg__(self):
        """Negation: -self"""
        # TODO: Implement using multiplication
        pass

    def __sub__(self, other):
        """Subtraction: self - other"""
        # TODO: Implement using addition and negation
        pass

    def __truediv__(self, other):
        """Division: self / other"""
        # TODO: Implement using multiplication and power
        # Hint: a / b = a * b^(-1)
        pass

    def __radd__(self, other):
        """Reverse addition: other + self (when other is not a Value)"""
        return self + other

    def __rmul__(self, other):
        """Reverse multiplication: other * self (when other is not a Value)"""
        return self * other

    def __rsub__(self, other):
        """Reverse subtraction: other - self (when other is not a Value)"""
        return Value(other) + (-self)

    def __rtruediv__(self, other):
        """Reverse division: other / self (when other is not a Value)"""
        return Value(other) * (self ** -1)

    def __lt__(self, other):
        """Less than comparison (compares data values)."""
        other = other if isinstance(other, Value) else Value(other)
        return self.data < other.data

    def backward(self):
        """
        Compute gradients for all nodes in the computational graph.

        This method should:
        1. Build a topological ordering of all nodes in the graph
        2. Set self.grad = 1.0 (the gradient of the output with respect to itself)
        3. Call _backward() on each node in reverse topological order

        What is topological order?
        --------------------------
        A topological ordering ensures that for every node, all nodes it depends on
        come before it in the ordering. For example, if c = a + b, then both a and b
        must appear before c in the topological order.

        For backpropagation, we need REVERSE topological order: we start from the
        output and work backwards, ensuring we process each node only after we've
        processed all nodes that depend on it. This guarantees that out.grad is
        fully accumulated before we call _backward().

        How to build it:
        ----------------
        Use depth-first search (DFS). Visit a node's children (in _prev) before
        adding the node itself to the list. This naturally produces topological order.
        """
        # TODO: Build topological order using DFS
        topo = []
        visited = set()

        def build_topo(v):
            # Hint: if v not in visited, mark it visited, recurse on v._prev, then append v
            pass

        build_topo(self)

        # TODO: Set the gradient of the output node to 1.0
        # TODO: Call _backward() on each node in reverse topological order
        pass
