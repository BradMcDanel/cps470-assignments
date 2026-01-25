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
        Addition: self + other

        Forward pass: out.data = self.data + other.data
        Backward pass: Both inputs receive the upstream gradient unchanged
                       (since d(a+b)/da = 1 and d(a+b)/db = 1)
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # TODO: Accumulate gradients for self and other
            # Hint: The gradient flows through addition unchanged
            pass

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

    def relu(self):
        """
        ReLU activation: max(0, x)

        Forward pass: out.data = max(0, self.data)
        Backward pass: gradient is 1 if self.data > 0, else 0
        """
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # TODO: Accumulate gradient for self
            # Hint: ReLU passes gradient through if input > 0, else blocks it
            pass

        out._backward = _backward
        return out

    def backward(self):
        """
        Compute gradients for all nodes in the computational graph.

        This method should:
        1. Build a topological ordering of all nodes in the graph
        2. Set self.grad = 1.0 (the gradient of the output with respect to itself)
        3. Call _backward() on each node in reverse topological order

        Hint: Use depth-first search to build the topological order.
        """
        # TODO: Build topological order
        topo = []
        visited = set()

        def build_topo(v):
            # TODO: Implement topological sort using DFS
            pass

        build_topo(self)

        # TODO: Set the gradient of the output node to 1.0
        # TODO: Call _backward() on each node in reverse topological order
        pass
