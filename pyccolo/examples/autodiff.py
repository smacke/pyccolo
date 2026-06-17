# -*- coding: utf-8 -*-
"""
Reverse-mode automatic differentiation (autograd) implemented with Pyccolo.

You write *ordinary* numeric Python -- including ``numpy`` calls like ``np.exp``,
``np.dot``, ``np.sum`` and operators like ``@`` -- and get correct gradients,
with no special "autodiff namespace."

How it works:

  * ``Var`` is a reverse-mode tape node wrapping a numpy array.  Arithmetic
    operators are overloaded so that running the program builds a computation
    graph; ``Var.backward()`` then walks it in reverse to accumulate gradients.

  * Operator overloading alone is *not enough*.  The moment user code calls a
    numpy function -- ``np.exp(x)`` -- numpy's ufunc machinery takes over and the
    gradient link is lost.  (We set ``__array_ufunc__ = None`` so this fails
    loudly with a ``TypeError`` instead of silently producing a wrong gradient.)
    Pyccolo supplies the missing piece: the ``before_call`` event lets a handler
    *replace the function being called*.  We swap ``np.exp`` for a differentiable
    ``d_exp`` (etc.) transparently, so idiomatic numpy code "just differentiates."

  * The same trick routes scalar ``math.exp`` / ``math.log`` / ... through the
    numpy-backed primitives, so plain ``math`` code differentiates too.

  * Calls into *your own* Python helper functions differentiate automatically:
    ``before_call`` instruments the helper on demand (and caches it), so the
    tape flows through its body with no rule needed.  Only C-level primitives
    (numpy/math) require entries in ``_INTERCEPT``.

  * If a ``Var`` reaches a numpy/math function we have *no* rule for, an
    ``AutodiffWarning`` is emitted (at call time, only when a ``Var`` is actually
    an argument) so a dropped gradient does not pass silently.

Headline demo (``python pyccolo/examples/autodiff.py``): logistic regression
trained from scratch by gradient descent, whose loss uses ``np.exp`` and
``np.log`` -- exercising the call-interception path on a real ML task.

Caveats (honest scope):
  * Per-function: ``grad`` / ``value_and_grad`` instrument the function you pass.
    Helper functions defined in *other* modules are not instrumented, so numpy
    calls there would not be intercepted (inline them, or instrument whole
    modules with an import hook -- not done here).
  * ``Var`` has no ``__float__``, and its ``__array__`` raises -- an
    un-intercepted numpy call fails loudly instead of silently dropping the
    gradient. (The raising ``__array__`` also makes ``Var`` statically
    array-like, so the numpy-using helpers type-check as ``Tensor = Var |
    ndarray`` rather than ``Any``.)
  * Broadcasting and arbitrary-rank tensors are handled generally: every binary
    op reduces its gradient back over broadcast axes (``_unbroadcast``), and
    matmul covers 1-D / 2-D / mixed / batched shapes.  Indexing reads
    (``x[...]``) gather forward and scatter-add backward.
  * No in-place mutation (``__setitem__``), no higher-order derivatives, no
    forward mode.
  * ``inspect.getsource`` is required, so REPL/lambda targets are unsupported.
"""
import functools
import inspect
import logging
import math
import sys
import sysconfig
import warnings
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np

import pyccolo as pyc

logger = logging.getLogger(__name__)

# Type aliases. ``Var`` is the tape node; an ``Operand`` is anything the ops and
# primitives accept -- a ``Var`` or a plain number/array that is lifted to one.
Scalar = Union[int, float]
Array = np.ndarray
ArrayLike = Union[Scalar, Array]
# An operand is a Var, a plain number/array, or a ``Weight`` proxy (a late-bound
# reference to an ambient parameter; see ``with`` support near ``ParamDict``).
Operand = Union["Var", ArrayLike, "Weight"]
# A differentiable tensor value: a ``Var`` during tracing, or a plain ndarray
# when the same helper is run outside the tape (e.g. at eval time).
Tensor = Union["Var", np.ndarray]
Axis = Optional[Union[int, Tuple[int, ...]]]
# Named aliases for numpy interop whose precise types are intractable to spell
# out (so the unavoidable ``Any`` is localized and documented, not bare).
Index = Any  # a NumPy __getitem__ key: int / slice / ndarray / tuple / None / ...
Shape = Any  # dim(s) for reshape: an int or a tuple of ints (np.reshape overloads)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _unbroadcast(grad: Array, shape: Tuple[int, ...]) -> Array:
    """Sum ``grad`` over axes that were broadcast, so it matches ``shape``."""
    grad = np.asarray(grad, dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


def _value(x: Operand) -> ArrayLike:
    if isinstance(x, Weight):
        x = x._live()
    return x.value if isinstance(x, Var) else x


# ---------------------------------------------------------------------------
# The reverse-mode tape node.
# ---------------------------------------------------------------------------
class Var:
    # Tell numpy to defer operators (``ndarray + Var``, ``ndarray @ Var``) to our
    # reflected methods, and to fail loudly on un-intercepted ufuncs.
    __array_ufunc__ = None

    def __init__(self, value: ArrayLike, _parents: Tuple["Var", ...] = ()) -> None:
        self.value: Array = np.asarray(value, dtype=float)
        self.grad: Array = np.zeros_like(self.value)
        self._parents = _parents
        self._backward: Callable[[], None] = lambda: None

    # -- construction helpers ------------------------------------------------
    def _unary(
        self, value: ArrayLike, grad_fn: Callable[[Array, Array], Array]
    ) -> "Var":
        out = Var(value, _parents=(self,))

        def _backward() -> None:
            self.grad = self.grad + _unbroadcast(
                grad_fn(self.value, out.grad), self.value.shape
            )

        out._backward = _backward
        return out

    def _binary(
        self,
        other: Operand,
        value_fn: Callable[[Array, Array], Array],
        grad_fn: Callable[[Array, Array, Array], Tuple[Array, Array]],
    ) -> "Var":
        other = _lift(other)
        out = Var(value_fn(self.value, other.value), _parents=(self, other))

        def _backward() -> None:
            ga, gb = grad_fn(self.value, other.value, out.grad)
            self.grad = self.grad + _unbroadcast(ga, self.value.shape)
            other.grad = other.grad + _unbroadcast(gb, other.value.shape)

        out._backward = _backward
        return out

    # -- arithmetic ----------------------------------------------------------
    def __add__(self, o: Operand) -> "Var":
        return self._binary(o, lambda a, b: a + b, lambda a, b, g: (g, g))

    __radd__ = __add__  # addition is commutative

    def __mul__(self, o: Operand) -> "Var":
        return self._binary(o, lambda a, b: a * b, lambda a, b, g: (g * b, g * a))

    __rmul__ = __mul__  # multiplication is commutative

    def __sub__(self, o: Operand) -> "Var":
        return self._binary(o, lambda a, b: a - b, lambda a, b, g: (g, -g))

    def __rsub__(self, o: Operand) -> "Var":
        return self._binary(o, lambda a, b: b - a, lambda a, b, g: (-g, g))

    def __truediv__(self, o: Operand) -> "Var":
        return self._binary(
            o, lambda a, b: a / b, lambda a, b, g: (g / b, -g * a / (b * b))
        )

    def __rtruediv__(self, o: Operand) -> "Var":
        return self._binary(
            o, lambda a, b: b / a, lambda a, b, g: (-g * b / (a * a), g / a)
        )

    def __neg__(self) -> "Var":
        return self._unary(-self.value, lambda a, g: -g)

    def __pow__(self, p: Operand) -> "Var":
        if isinstance(p, Var):
            # general x ** y == exp(y * log x)
            return d_exp(p * d_log(self))
        return self._unary(self.value**p, lambda a, g: g * p * a ** (p - 1))

    def __abs__(self) -> "Var":
        return d_abs(self)

    def __matmul__(self, o: Operand) -> "Var":
        return _matmul(self, o)

    def __rmatmul__(self, o: Operand) -> "Var":
        return _matmul(o, self)

    # -- comparisons return plain (non-differentiable) booleans --------------
    def __lt__(self, o: Operand) -> Array:
        return self.value < _value(o)

    def __le__(self, o: Operand) -> Array:
        return self.value <= _value(o)

    def __gt__(self, o: Operand) -> Array:
        return self.value > _value(o)

    def __ge__(self, o: Operand) -> Array:
        return self.value >= _value(o)

    def __eq__(self, o: Operand) -> Array:  # type: ignore[override]
        return self.value == _value(o)

    def __ne__(self, o: Operand) -> Array:  # type: ignore[override]
        return self.value != _value(o)

    __hash__ = None  # type: ignore[assignment]

    # -- indexing (gather forward, scatter-add backward) ---------------------
    def __getitem__(self, key: Index) -> "Var":
        out = Var(self.value[key], _parents=(self,))

        def _backward() -> None:
            grad = np.zeros_like(self.value)
            np.add.at(grad, key, out.grad)  # scatter-add handles repeated indices
            self.grad = self.grad + grad

        out._backward = _backward
        return out

    # -- numpy-style methods -------------------------------------------------
    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Var":
        return d_sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Var":
        return d_mean(self, axis=axis, keepdims=keepdims)

    def max(self, axis: Axis = None, keepdims: bool = False) -> "Var":
        return d_max(self, axis=axis, keepdims=keepdims)

    def min(self, axis: Axis = None, keepdims: bool = False) -> "Var":
        return d_min(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: Shape) -> "Var":
        return d_reshape(self, *shape)

    @property
    def T(self) -> "Var":
        return d_transpose(self)

    # -- non-differentiable metadata (read-only views of the underlying array) -
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def size(self) -> int:
        return self.value.size

    def __array__(self, *args: object, **kwargs: object) -> Array:
        # No concrete array view: a numpy call reaching here was NOT intercepted,
        # so fail loudly instead of silently building an object array. Defining it
        # also makes Var statically array-like, so numpy-typed code checks.
        raise TypeError(
            "Var has no array conversion; this numpy call should have been "
            "intercepted -- is the calling function instrumented?"
        )

    def __repr__(self) -> str:
        return f"Var({self.value!r})"

    # -- reverse pass --------------------------------------------------------
    def backward(self) -> None:
        topo: List[Var] = []
        visited = set()

        def build(v: "Var") -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for parent in v._parents:
                build(parent)
            topo.append(v)

        build(self)
        for v in topo:
            v.grad = np.zeros_like(v.value)
        self.grad = np.ones_like(self.value)
        for v in reversed(topo):
            v._backward()


def _lift(x: Operand) -> Var:
    if isinstance(x, Weight):
        x = x._live()
    return x if isinstance(x, Var) else Var(x)


def detach(x: Operand) -> Var:
    """A fresh leaf with the same value but no gradient history (stop-gradient)."""
    if isinstance(x, Weight):
        x = x._live()
    return Var(x.value if isinstance(x, Var) else x)


# ---------------------------------------------------------------------------
# Parameters (optional, opt-in). A bare array/number is trainable -- the
# default -- so existing code is unchanged. Wrap a value in a ``Param`` only
# when you need to *freeze* it (``frozen``) or *tie* it to another leaf
# (``tied``). ``value_and_grad`` reads these markers when it lifts argument
# leaves onto the tape; the pytree machinery and ``sgd_update`` carry ``Param``s
# through transparently, preserving the wrapper so updates stay in-structure.
# ---------------------------------------------------------------------------
@dataclass
class Param:
    """A model-parameter leaf carrying differentiation metadata.

    ``trainable=False`` holds the value fixed: its gradient comes back ``None``
    and ``sgd_update`` leaves it untouched. ``tie`` groups leaves that are the
    *same* underlying weight -- every ``Param`` sharing a ``tie`` key is backed
    by one tape node, so the gradient accumulates once and is reported
    identically at each position (tied params must be initialized equal; the
    shared node adopts the first occurrence's value).
    """

    value: Array
    trainable: bool = True
    tie: object = None
    # Stamped by the ``params{...}`` DSL with the block that declared this param,
    # so ``value_and_grad`` can reject a weight also passed in by hand.
    origin: object = None

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=float)


class _Frozen:
    """A non-trainable parameter, as a call or a subscript: ``frozen(v)`` /
    ``frozen[v]`` (the bracket form reads naturally inside a ``params{...}`` block)."""

    def __call__(self, value: ArrayLike) -> Param:
        return Param(np.asarray(value, dtype=float), trainable=False)

    def __getitem__(self, value: ArrayLike) -> Param:
        return self(value)


frozen = _Frozen()


class _TieRef:
    """Marker produced by ``tied[w]``: tie this slot to the sibling parameter whose
    value is ``target`` (the same array), reusing its init -- resolved by ``params``."""

    __slots__ = ("target",)

    def __init__(self, target: object) -> None:
        self.target = target


class _Tied:
    """Weight tying. ``tied(key, value)`` ties every parameter sharing ``key``;
    ``tied[w]`` (inside a ``params(...)`` block) ties to the sibling parameter ``w``
    by reference -- just the name, with no restatement of its initializer."""

    def __call__(self, key: object, value: ArrayLike) -> Param:
        return Param(np.asarray(value, dtype=float), tie=key)

    def __getitem__(self, ref: object) -> "_TieRef":
        return _TieRef(ref)


tied = _Tied()


# ---------------------------------------------------------------------------
# Differentiable primitives (vector-Jacobian-product rules).  Each computes the
# primal with numpy on ``.value`` and wires the local derivative.
# ---------------------------------------------------------------------------
def d_exp(x: Operand) -> Var:
    x = _lift(x)
    v = np.exp(x.value)
    return x._unary(v, lambda a, g: g * v)


def d_log(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.log(x.value), lambda a, g: g / a)


def d_sin(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.sin(x.value), lambda a, g: g * np.cos(a))


def d_cos(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.cos(x.value), lambda a, g: -g * np.sin(a))


def d_tanh(x: Operand) -> Var:
    x = _lift(x)
    v = np.tanh(x.value)
    return x._unary(v, lambda a, g: g * (1 - v * v))


def d_sqrt(x: Operand) -> Var:
    x = _lift(x)
    v = np.sqrt(x.value)
    return x._unary(v, lambda a, g: g / (2 * v))


def d_abs(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.abs(x.value), lambda a, g: g * np.sign(a))


def d_square(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.square(x.value), lambda a, g: g * 2 * a)


def d_sinh(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.sinh(x.value), lambda a, g: g * np.cosh(a))


def d_cosh(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.cosh(x.value), lambda a, g: g * np.sinh(a))


def d_arctan(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.arctan(x.value), lambda a, g: g / (1 + a * a))


def d_log1p(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.log1p(x.value), lambda a, g: g / (1 + a))


def d_expm1(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.expm1(x.value), lambda a, g: g * np.exp(a))


def d_reciprocal(x: Operand) -> Var:
    x = _lift(x)
    return x._unary(np.reciprocal(x.value), lambda a, g: -g / (a * a))


def _elementwise_max(a: Operand, b: Operand, pick_a: Callable[..., Array]) -> Var:
    a, b = _lift(a), _lift(b)
    out = Var(pick_a(a.value, b.value), _parents=(a, b))

    def _backward() -> None:
        mask = (a.value == out.value).astype(float)
        a.grad = a.grad + _unbroadcast(out.grad * mask, a.value.shape)
        b.grad = b.grad + _unbroadcast(out.grad * (1 - mask), b.value.shape)

    out._backward = _backward
    return out


def d_maximum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, np.maximum)


def d_minimum(a: Operand, b: Operand) -> Var:
    return _elementwise_max(a, b, np.minimum)


def d_clip(
    x: Operand, a_min: Optional[Operand] = None, a_max: Optional[Operand] = None
) -> Var:
    out = _lift(x)
    if a_min is not None:
        out = d_maximum(out, a_min)
    if a_max is not None:
        out = d_minimum(out, a_max)
    return out


def d_where(cond: Array, a: Operand, b: Operand) -> Var:
    a, b = _lift(a), _lift(b)
    cond = np.asarray(cond)
    out = Var(np.where(cond, a.value, b.value), _parents=(a, b))

    def _backward() -> None:
        a.grad = a.grad + _unbroadcast(np.where(cond, out.grad, 0.0), a.value.shape)
        b.grad = b.grad + _unbroadcast(np.where(cond, 0.0, out.grad), b.value.shape)

    out._backward = _backward
    return out


def _matmul_grads(a: Array, b: Array, g: Array) -> Tuple[Array, Array]:
    if a.ndim == 1 and b.ndim == 1:
        return g * b, g * a
    if a.ndim == 2 and b.ndim == 1:
        return np.outer(g, b), a.T @ g
    if a.ndim == 1 and b.ndim == 2:
        return b @ g, np.outer(a, g)
    return g @ b.swapaxes(-1, -2), a.swapaxes(-1, -2) @ g


def _matmul(a: Operand, b: Operand) -> Var:
    a, b = _lift(a), _lift(b)
    out = Var(a.value @ b.value, _parents=(a, b))

    def _backward() -> None:
        da, db = _matmul_grads(a.value, b.value, out.grad)
        a.grad = a.grad + _unbroadcast(da, a.value.shape)
        b.grad = b.grad + _unbroadcast(db, b.value.shape)

    out._backward = _backward
    return out


def d_sum(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    x = _lift(x)
    out = Var(x.value.sum(axis=axis, keepdims=keepdims), _parents=(x,))

    def _backward() -> None:
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        x.grad = x.grad + np.broadcast_to(g, x.value.shape)

    out._backward = _backward
    return out


def _reduced_count(x: Var, axis: Axis) -> int:
    if axis is None:
        return int(x.value.size)
    axes = axis if isinstance(axis, tuple) else (axis,)
    return int(np.prod([x.value.shape[a] for a in axes]))


def d_mean(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    x = _lift(x)
    return d_sum(x, axis=axis, keepdims=keepdims) / _reduced_count(x, axis)


def d_var(
    x: Operand,
    axis: Axis = None,
    dtype: object = None,
    out: object = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: object,
) -> Var:
    # Composed from mean/centering/square -- gradient flows for free; the numpy
    # signature order (a, axis, dtype, out, ddof, keepdims) is mirrored so
    # positional intercepted calls line up. dtype/out are ignored.
    x = _lift(x)
    centered = x - d_mean(x, axis=axis, keepdims=True)
    n = _reduced_count(x, axis)
    return d_sum(centered * centered, axis=axis, keepdims=keepdims) / (n - ddof)


def d_std(
    x: Operand,
    axis: Axis = None,
    dtype: object = None,
    out: object = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: object,
) -> Var:
    return d_var(x, axis=axis, ddof=ddof, keepdims=keepdims) ** 0.5


def _reduce_select(
    x: Operand, axis: Axis, keepdims: bool, reducer: Callable[..., Array]
) -> Var:
    # max/min: the gradient flows only to the selected element(s), split on ties.
    x = _lift(x)
    kept = reducer(x.value, axis=axis, keepdims=True)
    out = Var(reducer(x.value, axis=axis, keepdims=keepdims), _parents=(x,))

    def _backward() -> None:
        g = out.grad
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        mask = (x.value == kept).astype(float)
        mask /= mask.sum(axis=axis, keepdims=True)
        x.grad = x.grad + mask * g

    out._backward = _backward
    return out


def d_max(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    return _reduce_select(x, axis, keepdims, np.max)


def d_min(x: Operand, axis: Axis = None, keepdims: bool = False) -> Var:
    return _reduce_select(x, axis, keepdims, np.min)


def d_concatenate(seq: Sequence[Operand], axis: int = 0) -> Var:
    parts = [_lift(s) for s in seq]
    out = Var(
        np.concatenate([p.value for p in parts], axis=axis), _parents=tuple(parts)
    )

    def _backward() -> None:
        splits = np.cumsum([p.value.shape[axis] for p in parts])[:-1]
        for part, gpart in zip(parts, np.split(out.grad, splits, axis=axis)):
            part.grad = part.grad + gpart

    out._backward = _backward
    return out


def d_transpose(x: Operand, axes: Optional[Tuple[int, ...]] = None) -> Var:
    x = _lift(x)
    out = Var(np.transpose(x.value, axes), _parents=(x,))

    def _backward() -> None:
        if axes is None:
            x.grad = x.grad + np.transpose(out.grad)
        else:
            x.grad = x.grad + np.transpose(out.grad, np.argsort(axes))

    out._backward = _backward
    return out


def d_reshape(x: Operand, *shape: Shape) -> Var:
    x = _lift(x)
    newshape = shape[0] if len(shape) == 1 else shape
    out = Var(np.reshape(x.value, newshape), _parents=(x,))

    def _backward() -> None:
        x.grad = x.grad + out.grad.reshape(x.value.shape)

    out._backward = _backward
    return out


def d_expand_dims(x: Operand, axis: int) -> Var:
    x = _lift(x)
    pos = axis if axis >= 0 else axis + x.value.ndim + 1
    shape = list(x.value.shape)
    shape.insert(pos, 1)
    return d_reshape(x, tuple(shape))


# The ``*stack`` family is just ``concatenate`` after a shape fix-up, so composing
# the existing differentiable primitives gives correct gradients for free.
def d_stack(seq: Sequence[Operand], axis: int = 0) -> Var:
    # join along a NEW axis: expand each input at ``axis``, then concatenate there.
    return d_concatenate([d_expand_dims(s, axis) for s in seq], axis=axis)


def _atleast_2d_row(x: Operand) -> Var:
    x = _lift(x)
    if x.value.ndim == 0:
        return d_reshape(x, (1, 1))
    if x.value.ndim == 1:
        return d_reshape(x, (1, x.value.shape[0]))
    return x


def d_vstack(seq: Sequence[Operand]) -> Var:
    # row-wise: 1-D inputs become single rows, then concatenate along axis 0.
    return d_concatenate([_atleast_2d_row(s) for s in seq], axis=0)


def d_hstack(seq: Sequence[Operand]) -> Var:
    # column-wise: concatenate along axis 1, except 1-D inputs join along axis 0.
    parts = [_lift(s) for s in seq]
    axis = 0 if all(p.value.ndim <= 1 for p in parts) else 1
    return d_concatenate(parts, axis=axis)


def d_column_stack(seq: Sequence[Operand]) -> Var:
    # 1-D inputs become columns ((n,) -> (n, 1)); then concatenate along axis 1.
    parts = []
    for s in seq:
        p = _lift(s)
        parts.append(d_reshape(p, (p.value.shape[0], 1)) if p.value.ndim == 1 else p)
    return d_concatenate(parts, axis=1)


def _atleast_3d_depth(x: Operand) -> Var:
    x = _lift(x)
    if x.value.ndim == 0:
        return d_reshape(x, (1, 1, 1))
    if x.value.ndim == 1:
        return d_reshape(x, (1, x.value.shape[0], 1))
    if x.value.ndim == 2:
        return d_reshape(x, x.value.shape + (1,))
    return x


def d_dstack(seq: Sequence[Operand]) -> Var:
    # depth-wise: stack along a third axis (after promoting inputs to 3-D).
    return d_concatenate([_atleast_3d_depth(s) for s in seq], axis=2)


# ---------------------------------------------------------------------------
# The tracer: intercept numpy/math calls and route them to the d_* primitives.
# ---------------------------------------------------------------------------
# Only numpy ufuncs / math C-functions need entries here: they bypass our Python
# operator overloads (computing in C, or failing the disabled __array_ufunc__),
# so we must supply an explicit primitive + backward. Builtins like abs / sum /
# min / max are deliberately absent -- they dispatch through dunders (__abs__,
# repeated __add__, __lt__) onto ops that already have backward passes, so the
# tape builds itself and a rule would be redundant.
#
# Each differentiable primitive is listed once alongside every numpy/math
# callable that should route to it (e.g. np.exp and math.exp share d_exp); the
# flat lookup the tracer uses is denormalized from this below.
_RULES: Dict[Callable[..., object], Tuple[object, ...]] = {
    # elementwise unary
    d_exp: (np.exp, math.exp),
    d_log: (np.log, math.log),
    d_sin: (np.sin, math.sin),
    d_cos: (np.cos, math.cos),
    d_tanh: (np.tanh, math.tanh),
    d_sqrt: (np.sqrt, math.sqrt),
    d_sinh: (np.sinh, math.sinh),
    d_cosh: (np.cosh, math.cosh),
    d_arctan: (np.arctan, math.atan),
    d_log1p: (np.log1p, math.log1p),
    d_expm1: (np.expm1, math.expm1),
    d_abs: (np.abs,),
    d_square: (np.square,),
    d_reciprocal: (np.reciprocal,),
    # elementwise binary
    d_maximum: (np.maximum,),
    d_minimum: (np.minimum,),
    # selection
    d_where: (np.where,),
    d_clip: (np.clip,),
    # reductions
    d_sum: (np.sum,),
    d_mean: (np.mean,),
    d_var: (np.var,),
    d_std: (np.std,),
    d_max: (np.max, np.amax),
    d_min: (np.min, np.amin),
    # linear algebra / shape / structure
    _matmul: (np.dot, np.matmul),
    d_transpose: (np.transpose,),
    d_reshape: (np.reshape,),
    d_expand_dims: (np.expand_dims,),
    d_concatenate: (np.concatenate,),
    d_stack: (np.stack,),
    d_vstack: (np.vstack,),
    d_hstack: (np.hstack,),
    d_column_stack: (np.column_stack,),
    d_dstack: (np.dstack,),
}

_INTERCEPT: Dict[object, Callable[..., object]] = {
    fn: impl for impl, fns in _RULES.items() for fn in fns
}


class AutodiffWarning(UserWarning):
    """Emitted when a ``Var`` flows into a numpy/math function we cannot differentiate."""


def _contains_var(values: Iterable[object]) -> bool:
    for v in values:
        if isinstance(v, Var):
            return True
        if isinstance(v, (list, tuple)) and any(isinstance(x, Var) for x in v):
            return True
    return False


def _is_mathy(func: Callable[..., object]) -> bool:
    # numpy ufuncs / numpy functions / the math module are the calls that bypass
    # our operator overloading and silently drop gradients; builtins like abs/sum
    # work through dunder dispatch and must NOT be flagged.
    if isinstance(func, np.ufunc):
        return True
    module = getattr(func, "__module__", None) or ""
    return module == "math" or module.startswith("numpy")


# Directories holding stdlib / installed packages -- functions defined here are
# treated as "library" code and left alone (only *your* code is instrumented).
_LIB_DIRS = tuple(
    p
    for p in {
        sysconfig.get_paths().get("stdlib"),
        sysconfig.get_paths().get("platstdlib"),
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
    }
    if p
)


def _is_user_function(func: Callable[..., object]) -> bool:
    """True for a plain Python function defined in user (non-library) code.

    The autodiff primitives (``d_*`` etc.) live in this module but are only ever
    reached via ``_INTERCEPT`` / operator dispatch, never as a call site inside
    instrumented user code, so they are never handed to this predicate.
    """
    if not inspect.isfunction(func):  # excludes C builtins, numpy ufuncs, methods
        return False
    module = getattr(func, "__module__", "") or ""
    # Skip numpy and pyccolo *core* -- but not this example's own helpers.
    if module.startswith("numpy"):
        return False
    if module.startswith("pyccolo.") and not module.startswith("pyccolo.examples"):
        return False
    filename = getattr(getattr(func, "__code__", None), "co_filename", "") or ""
    if not filename or filename.startswith("<"):  # <stdin>, <string>, ...
        return False
    return not any(filename.startswith(d) for d in _LIB_DIRS)


_WRAPPERS: Dict[Callable[..., object], Callable[..., object]] = {}


def _warn_wrapper(func: Callable[..., object]) -> Callable[..., object]:
    wrapper = _WRAPPERS.get(func)
    if wrapper is not None:
        return wrapper
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: object, **kwargs: object) -> object:
        if _contains_var(args) or _contains_var(kwargs.values()):
            warnings.warn(
                f"autodiff: no differentiation rule for {name!r}; "
                "the gradient will not flow through this call",
                AutodiffWarning,
                stacklevel=2,
            )
        return func(*args, **kwargs)

    _WRAPPERS[func] = _wrapped
    return _wrapped


class AutodiffTracer(pyc.BaseTracer):
    # Instrument whichever file the differentiated function lives in.
    instrument_all_files = True

    # ``instrumented`` rebinds ``__code__`` and is not idempotent, so each helper
    # is instrumented exactly once and reused.
    _helpers: Dict[Callable[..., object], Callable[..., object]] = {}

    def _instrument_helper(self, func: Callable[..., object]) -> Callable[..., object]:
        cached = self._helpers.get(func)
        if cached is not None:
            return cached
        try:
            instrumented = self.instrumented(func)
        except (OSError, TypeError, SyntaxError):
            # No retrievable/parseable source (e.g. a closure over free vars, a
            # C-accelerated shim): leave it alone rather than fail.
            instrumented = func
        self._helpers[func] = instrumented
        return instrumented

    def resolve_call(self, func: Callable[..., object]) -> Callable[..., object]:
        """Map a callable to its autodiff-aware version.

        Swap an intercepted numpy/math function for its differentiable primitive;
        instrument a user helper on demand so the tape flows through its body; or
        wrap an un-ruled numpy/math function so it warns if a Var flows in. This is
        the logic ``before_call`` applies; it is exposed so other call mechanisms
        (e.g. pipescript's ``|>`` via its application hooks) can participate too.
        """
        replacement = _INTERCEPT.get(func)
        if replacement is not None:
            return replacement
        if _is_user_function(func):
            return self._instrument_helper(func)
        if _is_mathy(func):
            return _warn_wrapper(func)
        return func  # builtins, methods, etc. pass through

    @pyc.register_handler(pyc.before_call)
    def handle_before_call(
        self, func: Callable[..., object], node: object, *_: object, **__: object
    ) -> Callable[..., object]:
        return self.resolve_call(func)


# ---------------------------------------------------------------------------
# Pytrees: nested list/tuple/dict containers with arrays/scalars/Vars as leaves.
# Let parameters be one structured object (a dict of weights, say) instead of a
# pile of positional arrays, so value_and_grad returns gradients with the same
# shape. list/tuple/dict (and subclasses) are nodes; everything else is a leaf.
# (No namedtuple/custom-node registration; a namedtuple round-trips as a tuple.)
# ---------------------------------------------------------------------------
class _LeafMarker:
    """Placeholder marking a leaf position in a treedef."""


_LEAF = _LeafMarker()

# A leaf is a Param, a Var, a plain number/array, or None (``None`` appears in a
# gradient pytree at a frozen/non-numeric leaf). A pytree is a leaf or a nested
# list/tuple/dict of pytrees; a treedef mirrors that shape with leaves replaced
# by ``_LeafMarker``.
Leaf = Optional[Union["Param", Operand]]
PyTree = Union[Leaf, List["PyTree"], Tuple["PyTree", ...], Dict[str, "PyTree"]]
TreeDef = Union[
    _LeafMarker, List["TreeDef"], Tuple["TreeDef", ...], Dict[str, "TreeDef"]
]


def _unwrap(x: object) -> object:
    """Resolve a ``Weight`` proxy to its current value; leave anything else."""
    return x._live() if isinstance(x, Weight) else x


def _as_arr(x: object) -> Array:
    """``_unwrap`` an operand and view it as an ndarray for typing (a ``Var``
    operand still dispatches correctly at runtime)."""
    return cast(Array, _unwrap(x))


def _deep_unwrap(x: object) -> object:
    """``_unwrap`` through lists/tuples (e.g. ``np.concatenate([w1, w2])``)."""
    if isinstance(x, Weight):
        return x._live()
    if isinstance(x, list):
        return [_deep_unwrap(e) for e in x]
    if isinstance(x, tuple):
        return tuple(_deep_unwrap(e) for e in x)
    return x


def _any_recording(obj: object) -> bool:
    """True if a numpy call over ``obj`` should record -- i.e. some operand
    resolves to a ``Var`` (training), vs. plain arrays (inference)."""
    if isinstance(obj, Var):
        return True
    if isinstance(obj, Weight):
        return isinstance(obj._live(), Var)
    if isinstance(obj, (list, tuple)):
        return any(_any_recording(e) for e in obj)
    return False


class Weight:
    """A late-bound proxy for an ambient parameter (injected by ``with weights:``).

    It forwards every operation to the *current* value of its weight: the plain
    array during inference (so nothing is taped) and the live ``Var`` during a
    ``grad`` pass (so the op records). The proxy is mode-agnostic -- the mode is
    whatever its owner is currently bound to -- which is what lets a single model
    definition serve both inference and training without binding a ``Var`` into it.
    It speaks numpy's dispatch protocols, so ``np.exp(w)`` / ``np.sum(w)`` on a bare
    weight route to the differentiable primitive when recording and to plain numpy
    otherwise.
    """

    __slots__ = ("_owner", "_key")

    def __init__(self, owner: "ParamDict", key: str) -> None:
        self._owner = owner
        self._key = key

    def _live(self) -> Union[Var, ArrayLike]:
        return self._owner._resolve_weight(self._key)

    # The live value is a Var or ndarray; both support the operators below, but
    # mypy can't express "whichever it is, it has @/+/...", so we view it as an
    # ndarray for typing -- the runtime op dispatches correctly to Var either way.
    def _arr(self) -> Array:
        return cast(Array, self._live())

    # -- operators: forward to the live value, unwrapping proxy operands ----------
    def __matmul__(self, o: object) -> Operand:
        return self._arr() @ _as_arr(o)

    def __rmatmul__(self, o: object) -> Operand:
        return _as_arr(o) @ self._arr()

    def __add__(self, o: object) -> Operand:
        return self._arr() + _as_arr(o)

    __radd__ = __add__

    def __sub__(self, o: object) -> Operand:
        return self._arr() - _as_arr(o)

    def __rsub__(self, o: object) -> Operand:
        return _as_arr(o) - self._arr()

    def __mul__(self, o: object) -> Operand:
        return self._arr() * _as_arr(o)

    __rmul__ = __mul__

    def __truediv__(self, o: object) -> Operand:
        return self._arr() / _as_arr(o)

    def __rtruediv__(self, o: object) -> Operand:
        return _as_arr(o) / self._arr()

    def __pow__(self, o: object) -> Operand:
        return self._arr() ** _as_arr(o)

    def __neg__(self) -> Operand:
        return -self._arr()

    def __getitem__(self, key: Index) -> Operand:
        return cast(Operand, self._arr()[key])

    @property
    def T(self) -> Operand:
        return self._arr().T

    @property
    def shape(self) -> Tuple[int, ...]:
        return np.shape(_value(self))

    @property
    def ndim(self) -> int:
        return np.ndim(_value(self))

    @property
    def size(self) -> int:
        return int(np.size(_value(self)))

    # -- numpy dispatch: route a ufunc / array-function over a bare weight --------
    def __array_ufunc__(
        self, ufunc: object, method: str, *inputs: object, **kwargs: object
    ) -> object:
        if method != "__call__":
            return NotImplemented
        if _any_recording(inputs):
            prim = _INTERCEPT.get(ufunc)
            if prim is not None:
                return prim(*[_unwrap(i) for i in inputs], **kwargs)
            return _warn_wrapper(cast(Callable[..., object], ufunc))(
                *[_deep_unwrap(i) for i in inputs], **kwargs
            )
        return cast(Callable[..., object], ufunc)(
            *[_deep_unwrap(i) for i in inputs], **kwargs
        )

    def __array_function__(
        self,
        func: Callable[..., object],
        types: object,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        if _any_recording(args):
            prim = _INTERCEPT.get(func)
            if prim is not None:
                return prim(*args, **kwargs)
            return _warn_wrapper(func)(*[_deep_unwrap(a) for a in args], **kwargs)
        return func(*[_deep_unwrap(a) for a in args], **kwargs)

    def __repr__(self) -> str:
        return f"Weight({self._key!r})"


class ParamDict(dict):
    """A parameter dict that also allows attribute access: ``model.w`` reads
    ``model["w"]`` (and ``model.w = v`` sets it). ``params(...)`` returns one at
    every dict level. It is an ordinary ``dict`` otherwise, so the pytree
    machinery, ``value_and_grad`` and ``sgd_update`` treat it as a dict; the
    flatten/unflatten round trip preserves the type, so attribute access survives
    an optimizer step (and gradient pytrees come back attribute-accessible too). A
    weight named like a dict method (``items``) is still reachable via
    ``model["items"]``.

    Used as a context manager, ``with weights:`` injects a ``Weight`` proxy for each
    parameter into the caller's module/cell globals (collision-checked, removed on
    exit), so a forward pass can reference the weights *unqualified* -- ``w`` rather
    than ``weights.w`` -- and still serve both clean inference and training. Drive
    training with :meth:`grad` (binds the weights to ``Var``s, backprops, returns a
    gradient ``ParamDict``) and :meth:`step` (in-place SGD).
    """

    # Live binding during a grad pass (name -> Var/array), and the injected-globals
    # bookkeeping for the context manager. Both are real instance attributes, set
    # via the leading-underscore path in ``__setattr__``.
    _live: Optional[Dict[str, "Leaf"]] = None
    _scope: Optional[Tuple[Dict[str, object], List[str]]] = None

    # Leading-underscore attributes are real instance state (e.g. the live-binding
    # used during a grad pass); every other name maps to a dict item.
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    # -- ambient-weights context manager + tape-style training --------------------
    def _resolve_weight(self, key: str) -> Union[Var, ArrayLike]:
        """The current value of ``key``: the live ``Var`` during a grad pass, else
        the parameter's plain array (inference)."""
        live = getattr(self, "_live", None)
        if live is not None and key in live:
            return cast(Union[Var, ArrayLike], live[key])
        leaf = self[key]
        return (
            leaf.value if isinstance(leaf, Param) else cast(Union[Var, ArrayLike], leaf)
        )

    def __enter__(self) -> "ParamDict":
        g = sys._getframe(1).f_globals
        injected: List[str] = []
        for key in self:
            if not isinstance(self[key], Param):
                continue
            existing = g.get(key)
            if key in g and not (
                isinstance(existing, Weight) and existing._owner is self
            ):
                for name in injected:
                    del g[name]
                raise ValueError(
                    f"with-weights: name {key!r} already exists in the enclosing "
                    "scope; rename the parameter or the existing variable"
                )
            g[key] = Weight(self, key)
            injected.append(key)
        self._scope = (g, injected)
        return self

    def __exit__(self, *exc: object) -> None:
        scope = self._scope
        if scope is not None:
            g, injected = scope
            for key in injected:
                got = g.get(key)
                if isinstance(got, Weight) and got._owner is self:
                    del g[key]
            self._scope = None

    def grad(self, objective: Callable[[], object]) -> Tuple[Array, "ParamDict"]:
        """Run ``objective`` (a no-arg callable returning a scalar, built from this
        model) with the weights bound to ``Var``s, backprop, and return
        ``(value, grads)`` where ``grads`` is a ``ParamDict`` of gradients (``None``
        at frozen weights). Tied weights share one gradient. Trainable leaves only;
        nest by calling per sub-tree."""
        tie_vars: Dict[object, Var] = {}
        live: Dict[str, Leaf] = {}
        grad_vars: Dict[str, Optional[Var]] = {}
        for key in self:
            leaf = self[key]
            if isinstance(leaf, Param) or _is_numeric(leaf):
                var, call_value = _wrap_leaf(cast(Leaf, leaf), tie_vars)
                live[key] = call_value
                grad_vars[key] = var
            else:
                live[key] = cast(Leaf, leaf)
                grad_vars[key] = None
        runner = _INSTRUMENTED.get(objective)
        if runner is None:
            runner = _make_runner(objective)  # cache: instrumented() is not idempotent
            _INSTRUMENTED[objective] = runner
        prev = getattr(self, "_live", None)
        self._live = live
        try:
            out = runner()
        finally:
            self._live = prev
        if isinstance(out, Var):
            out.backward()
            value: Array = out.value
        else:
            value = np.asarray(out, dtype=float)
        grads = ParamDict(
            {
                key: (gv.grad if gv is not None else None)
                for key, gv in grad_vars.items()
            }
        )
        return value, grads

    def step(self, grads: "ParamDict", lr: float) -> None:
        """One in-place SGD step: ``p.value -= lr * grad`` for each trainable leaf
        (frozen leaves, whose gradient is ``None``, are left untouched). The model's
        proxies read the updated values on the next call."""
        for key in self:
            leaf = self[key]
            g = grads.get(key)
            if isinstance(leaf, Param) and leaf.trainable and g is not None:
                leaf.value = cast(Array, leaf.value) - lr * cast(Array, g)


def tree_flatten(tree: PyTree) -> Tuple[List[Leaf], TreeDef]:
    """Return ``(leaves, treedef)`` -- the leaves in a fixed order, plus a
    structure description that ``tree_unflatten`` can rebuild from."""
    leaves: List[Leaf] = []

    def build(node: PyTree) -> TreeDef:
        if isinstance(node, list):
            return [build(child) for child in node]
        if isinstance(node, tuple):
            return tuple(build(child) for child in node)
        if isinstance(node, dict):
            built = {key: build(node[key]) for key in sorted(node)}
            # Preserve a dict subtype (e.g. ParamDict) so it survives a round trip.
            return ParamDict(built) if isinstance(node, ParamDict) else built
        leaves.append(node)
        return _LEAF

    return leaves, build(tree)


def tree_unflatten(treedef: TreeDef, leaves: Iterable[Leaf]) -> PyTree:
    """Rebuild a pytree of the given ``treedef`` from a flat ``leaves`` iterable."""
    it = iter(leaves)

    def build(td: TreeDef) -> PyTree:
        if isinstance(td, list):
            return [build(child) for child in td]
        if isinstance(td, tuple):
            return tuple(build(child) for child in td)
        if isinstance(td, dict):
            built = {key: build(td[key]) for key in td}
            return ParamDict(built) if isinstance(td, ParamDict) else built
        return next(it)

    return build(treedef)


def tree_leaves(tree: PyTree) -> List[Leaf]:
    return tree_flatten(tree)[0]


def tree_structure(tree: PyTree) -> TreeDef:
    """The treedef of ``tree`` -- equal (``==``) iff two trees have the same shape."""
    return tree_flatten(tree)[1]


def tree_map(func: Callable[..., Leaf], tree: PyTree, *rest: PyTree) -> PyTree:
    """Apply ``func`` leafwise across one or more same-structured pytrees."""
    leaves, treedef = tree_flatten(tree)
    rest_leaves = [tree_flatten(other)[0] for other in rest]
    return tree_unflatten(treedef, [func(*xs) for xs in zip(leaves, *rest_leaves)])


def _sgd_step(p: Leaf, g: Leaf, lr: float) -> Leaf:
    """One leafwise SGD step, preserving ``Param`` wrappers and skipping anything
    with no gradient (``g is None`` -- a frozen or non-numeric leaf)."""
    if isinstance(p, Param):
        if not p.trainable or g is None:
            return p  # frozen / no gradient: held fixed, wrapper preserved
        return replace(p, value=cast(Array, p.value) - lr * cast(Array, g))
    if g is None:
        return p
    return cast(Array, p) - lr * cast(Array, g)


def sgd_update(params: PyTree, grads: PyTree, lr: float) -> PyTree:
    """One SGD step over an arbitrary param pytree: ``p <- p - lr * g`` leafwise.

    Both pytrees must share a structure (e.g. the gradient pytree that
    ``value_and_grad`` returns for ``params``); the result has the same structure.
    ``Param`` leaves are carried through as ``Param``s -- frozen ones (gradient
    ``None``) are left untouched, trainable ones are stepped in place.
    """
    return tree_map(lambda p, g: _sgd_step(p, g, lr), params, grads)


# ---------------------------------------------------------------------------
# Declarative parameter blocks (optional). ``params(...)`` is the ergonomic
# entry point for building a parameter pytree: name each weight as a keyword,
# write plain numbers/arrays for trainable weights, and use ``frozen`` / ``tied``
# for the rest. Bare numerics are wrapped into trainable ``Param``s and every
# leaf is stamped with this block's identity, so ``value_and_grad`` can reject a
# weight that is also handed in by hand (one weight, one owner). Nest by passing
# a dict (or another ``params(...)``) as a value.
#
# This is the library form of the DSL and needs no pipescript. The literal
# ``params{ w1 = ...; b1 = ... }`` brace surface is sugar over it, available under
# pipescript via ``register_pipescript_params_macro`` (below) -- the brace block's
# assignments are harvested into a namespace and fed to ``params(**names)``.
# ---------------------------------------------------------------------------
def params(spec: Optional[Dict[str, PyTree]] = None, **named: PyTree) -> ParamDict:
    """Build a parameter pytree as a ``ParamDict``: ``{name: Param}``, with nested
    dicts/lists allowed and attribute access (``model.w`` as well as ``model["w"]``).

    ``params(w=0.1 * rng.standard_normal((2, 3)), b=np.zeros(3))`` makes both
    trainable; wrap a value in ``frozen(...)``/``frozen[...]`` to hold it fixed, or
    ``tied(key, ...)`` to share it by key. Inside the block, ``tied[w]`` ties a slot
    to the sibling parameter ``w`` by reference, reusing its initializer. Pass a
    mapping positionally instead of keywords if your names are not valid
    identifiers. The result flows through ``value_and_grad`` and ``sgd_update``
    unchanged; use ``param_values`` to recover the raw arrays for inference.
    """
    if spec is not None and named:
        raise TypeError("params() takes either a mapping or keyword args, not both")
    items = dict(spec) if spec is not None else dict(named)
    token = object()  # this block's identity, for the single-owner check

    # Map each top-level value's array identity to its declaring name, so a
    # ``tied[w]`` reference can find the sibling parameter it shares a weight with.
    declarer: Dict[int, str] = {}
    for name, value in items.items():
        arr = value.value if isinstance(value, Param) else value
        if isinstance(arr, np.ndarray):
            declarer.setdefault(id(arr), name)
    tie_keys: Dict[str, object] = {}  # target name -> shared tie sentinel

    def wrap(v: PyTree, key: Optional[str] = None) -> PyTree:
        if isinstance(v, _TieRef):
            arr = v.target.value if isinstance(v.target, Param) else v.target
            target = declarer.get(id(arr)) if isinstance(arr, np.ndarray) else None
            if target is None or target == key:
                raise ValueError(
                    "autodiff: tied[...] must reference another parameter declared "
                    "in the same params(...) call (by name)"
                )
            p = Param(np.asarray(cast(ArrayLike, arr), dtype=float))
            p.tie = tie_keys.setdefault(target, object())
            p.origin = token
            return p
        if isinstance(v, Param):
            if v.origin is None:
                v.origin = token  # adopt leaves from frozen()/tied() and nesting
            return v
        if isinstance(v, dict):
            return ParamDict({k: wrap(child) for k, child in v.items()})
        if isinstance(v, list):
            return [wrap(child) for child in v]
        if isinstance(v, tuple):
            return tuple(wrap(child) for child in v)
        if _is_numeric(v):
            p = Param(np.asarray(cast(ArrayLike, v), dtype=float))
            p.origin = token
            return p
        return v  # non-numeric leaf (e.g. a label): left alone, no gradient

    result = ParamDict({key: wrap(value, key) for key, value in items.items()})
    # Each referenced target must carry the same tie key as the slots tied to it.
    for target, sentinel in tie_keys.items():
        leaf = result.get(target)
        if isinstance(leaf, Param):
            leaf.tie = sentinel
    return result


def param_values(tree: PyTree) -> PyTree:
    """Strip ``Param`` wrappers back to raw values, preserving structure -- the
    inverse of what ``params`` adds, for running a model at inference time."""
    return tree_map(lambda p: p.value if isinstance(p, Param) else p, tree)


def register_pipescript_params_macro() -> None:
    """Wire the literal ``params{ w = ...; b = ... }`` brace surface into pipescript.

    pipescript's namespace-block mechanism runs the brace block, harvests its
    top-level assignments (``_``-prefixed names are local temporaries, excluded),
    and hands them to a builder; we point that builder at ``params(**names)``, so::

        model = params{
            w = 0.1 * rng.standard_normal((2, 16))
            b = np.zeros(16)
            g = frozen[np.ones(16)]
            out = tied[w]               # ties to w by name, reusing its init
        }

    builds the same ``{name: Param}`` pytree as the call form. ``frozen`` / ``tied``
    must be in scope inside the block. Requires pipescript; call once after loading
    it (e.g. in a notebook after ``%load_ext pipescript``)."""
    from pipescript.tracers.macro_tracer import register_namespace_macro

    register_namespace_macro(
        "params", lambda namespace: params(**namespace), call_form=params
    )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def resolve_call(func: Callable[..., object]) -> Callable[..., object]:
    """Autodiff-aware version of ``func`` (the logic ``before_call`` applies).

    Exposed so other call mechanisms can opt into interception -- e.g. wiring a
    pipescript pipe hook so ``x |> np.exp`` differentiates when ``x`` is a Var::

        from pipescript.tracers.pipeline_tracer import PipelineTracer
        PipelineTracer.application_hooks.append(
            lambda f, v: resolve_call(f) if isinstance(v, Var) else f
        )
    """
    return AutodiffTracer.instance().resolve_call(func)


def _is_numeric(x: object) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) or isinstance(
        x, np.ndarray
    )


# ``tracer.instrumented`` rebinds ``f.__code__`` (and is not idempotent -- a second
# call re-reads the now-relocated source), so build each function's runner once.
_INSTRUMENTED: Dict[Callable[..., object], Callable[..., object]] = {}


def _make_runner(f: Callable[..., object]) -> Callable[..., object]:
    """A callable that runs ``f`` with autodiff interception active.

    Normally that means instrumenting ``f``'s source so its calls emit before_call.
    But a function with no recoverable source -- notably a pipescript ``|>`` pipe
    lambda, which is synthesized via ``pyc.eval`` and is therefore *already* woven
    by its tracers -- can't be (re)instrumented; run it directly instead, with the
    autodiff tracer enabled so the tape is built as the pipe executes.
    """
    tracer = AutodiffTracer.instance()
    fname = getattr(getattr(f, "__code__", None), "co_filename", "") or ""
    if not fname.startswith("<"):  # a real source file -> weave before_call in
        try:
            return tracer.instrumented(f)
        except (OSError, TypeError, SyntaxError, ValueError):
            # ValueError: a closure (free vars) can't be recompiled standalone; run
            # it directly (its numpy must reach the tape via pipes or proxies then).
            pass

    @functools.wraps(f)
    def run_directly(*args: object, **kwargs: object) -> object:
        with tracer.tracing_enabled():
            return f(*args, **kwargs)

    return run_directly


def _dup_param_msg(p: "Param") -> str:
    where = f" (declared in params block {p.origin!r})" if p.origin is not None else ""
    return (
        f"autodiff: a parameter{where} appears more than once across the "
        "differentiated arguments; a weight must have a single owner -- declare it "
        "once, and use tied(key, value) to share one weight across positions"
    )


def _check_param_ownership(args: Tuple[PyTree, ...]) -> None:
    """Reject a ``params{...}``-declared weight that is *also* handed in as a
    separate ``Param`` leaf.

    A weight with two owners would be lifted onto the tape -- and stepped by the
    optimizer -- through both paths. We flag the same ``Param`` object reused in
    two leaf positions, and a distinct ``Param`` aliasing a block-owned weight's
    value array. (Sharing one weight across positions is what ``tied`` is for.)
    """
    seen_obj: Set[int] = set()
    val_owner: Dict[int, Param] = {}
    for a in args:
        for leaf in tree_leaves(a):
            if not isinstance(leaf, Param):
                continue
            if id(leaf) in seen_obj and leaf.origin is not None:
                raise ValueError(_dup_param_msg(leaf))
            seen_obj.add(id(leaf))
            other = val_owner.get(id(leaf.value))
            tied_pair = (
                other is not None and leaf.tie is not None and leaf.tie == other.tie
            )
            if (
                other is not None
                and other is not leaf
                and not tied_pair  # tied params legitimately share one weight
                and (leaf.origin is not None or other.origin is not None)
            ):
                raise ValueError(
                    _dup_param_msg(leaf if leaf.origin is not None else other)
                )
            val_owner.setdefault(id(leaf.value), leaf)


def _wrap_leaf(leaf: Leaf, tie_vars: Dict[object, Var]) -> Tuple[Optional[Var], Leaf]:
    """Lift one argument leaf onto the tape.

    Returns ``(var, call_value)``: ``var`` is the tape node whose ``.grad`` becomes
    this leaf's gradient (``None`` when the leaf is frozen or non-numeric, so its
    gradient comes back ``None``); ``call_value`` is what to pass into the
    differentiated function in this leaf's place. Trainable ``Param``s and bare
    numerics become fresh ``Var``s; a frozen ``Param`` passes its raw value
    through; ``Param``s sharing a ``tie`` key share a single ``Var``.
    """
    if isinstance(leaf, _TieRef):
        raise ValueError(
            "autodiff: tied[...] is only meaningful inside params(...), where it "
            "references a sibling parameter; it reached value_and_grad unresolved"
        )
    if isinstance(leaf, Param):
        if not leaf.trainable:
            return None, leaf.value
        if leaf.tie is not None:
            shared = tie_vars.get(leaf.tie)
            if shared is None:
                shared = Var(leaf.value)
                tie_vars[leaf.tie] = shared
            return shared, shared
        v = Var(leaf.value)
        return v, v
    if _is_numeric(leaf):
        v = Var(cast(ArrayLike, leaf))
        return v, v
    return None, leaf


def value_and_grad(
    f: Callable[..., object],
) -> Callable[..., Tuple[Array, Tuple[PyTree, ...]]]:
    """Wrap ``f`` so that calling it returns ``(value, grads)``.

    ``grads`` is a tuple with one entry per positional argument, holding the
    gradient of the (scalar) output w.r.t. that argument. Each argument may be a
    pytree (e.g. a dict of weights); its gradient comes back as a matching pytree,
    with ``None`` at any non-numeric or frozen leaf. A bare array/scalar is just a
    pytree with one leaf, so it yields a bare gradient (backward compatible).
    Leaves may be ``Param``s to opt into freezing (``frozen``) or tying (``tied``).
    ``f`` may be an ordinary function or a pipescript ``|>`` pipe lambda (run it
    under ``PipelineTracer``).
    """
    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    def wrapped(*args: PyTree) -> Tuple[Array, Tuple[PyTree, ...]]:
        call_args: List[PyTree] = []
        per_arg: List[Tuple[TreeDef, List[Tuple[Leaf, Optional[Var]]]]] = []
        _check_param_ownership(args)
        tie_vars: Dict[object, Var] = {}
        for a in args:
            leaves, treedef = tree_flatten(a)
            info: List[Tuple[Leaf, Optional[Var]]] = []
            wrapped_leaves: List[Leaf] = []
            for leaf in leaves:
                var, call_value = _wrap_leaf(leaf, tie_vars)
                info.append((leaf, var))
                wrapped_leaves.append(call_value)
            call_args.append(tree_unflatten(treedef, wrapped_leaves))
            per_arg.append((treedef, info))

        out = runner(*call_args)
        if isinstance(out, Var):
            out.backward()  # otherwise each leaf's grad stays at its init zeros
            value: Array = out.value
        else:
            value = np.asarray(out, dtype=float)

        grads = tuple(
            tree_unflatten(
                treedef,
                [None if v is None else _match_arg(orig, v.grad) for orig, v in info],
            )
            for treedef, info in per_arg
        )
        return value, grads

    return wrapped


def _match_arg(orig: Leaf, grad: Array) -> Operand:
    value = orig.value if isinstance(orig, Param) else orig
    return grad if isinstance(value, np.ndarray) else float(grad)


def grad(f: Callable[..., object]) -> Callable[..., Tuple[PyTree, ...]]:
    """Return a function computing just the gradient tuple of ``f`` (one entry per
    argument; each matches that argument's pytree structure)."""
    vg = value_and_grad(f)
    return lambda *args: vg(*args)[1]


def gradient_descent(
    loss_fn: Callable[..., object],
    init_params: Tuple[ArrayLike, ...],
    lr: float = 0.1,
    steps: int = 100,
) -> Tuple[Tuple[Array, ...], List[float]]:
    """Minimize ``loss_fn(*params)`` by gradient descent; return (params, history).

    Each step replaces ``p`` with ``p - lr * grad``; since a number/array minus an
    array is always an array, the returned params are ``Array``s (a scalar init
    like ``b=0.0`` is promoted on the first update). A positional ``frozen`` param
    is held fixed (its gradient is ``None``); ``Param`` wrappers are preserved.
    """
    vg = value_and_grad(loss_fn)
    params: List[Leaf] = [cast(Leaf, p) for p in init_params]
    history: List[float] = []
    for _ in range(steps):
        loss, grads = vg(*params)
        history.append(float(loss))
        params = [_sgd_step(p, cast(Leaf, g), lr) for p, g in zip(params, grads)]
    return tuple(cast(List[Array], params)), history


# ---------------------------------------------------------------------------
# Demo: logistic regression trained from scratch.
# ---------------------------------------------------------------------------
def _make_data() -> Tuple[Array, Array]:
    rng = np.random.default_rng(0)
    n = 50
    pos = rng.normal(loc=[2.0, 2.0], scale=0.6, size=(n, 2))
    neg = rng.normal(loc=[-2.0, -2.0], scale=0.6, size=(n, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return X, y


_X, _y = _make_data()


# The demo helper/loss functions below are typed with ``Tensor`` (Var-or-ndarray):
# numpy calls on a ``Var`` type-check because ``Var.__array__`` makes it statically
# array-like; scalar-returning losses are typed ``Operand``.
def sigmoid(z: Tensor) -> Tensor:
    # A plain helper with no autodiff rule -- differentiated by instrumenting it
    # on demand when ``logistic_loss`` calls it.
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss(w: Tensor, b: Tensor) -> Operand:
    z = _X @ w + b  # ndarray @ Var defers to Var.__rmatmul__; + b broadcasts
    p = sigmoid(z)  # helper differentiated transparently
    eps = 1e-12
    return -np.mean(_y * np.log(p + eps) + (1 - _y) * np.log(1 - p + eps))


def _accuracy(w: Array, b: ArrayLike) -> float:
    z = _X @ w + b
    return float(np.mean((z > 0).astype(float) == _y))


# Same logistic model, but parameters are a declarative ``params(...)`` block so
# we can freeze the bias: ``model["b"]`` is read by name and held fixed.
def logistic_param_loss(model: Dict[str, Tensor]) -> Operand:
    z = _X @ model["w"] + model["b"]
    p = sigmoid(z)
    eps = 1e-12
    return -np.mean(_y * np.log(p + eps) + (1 - _y) * np.log(1 - p + eps))


# ---------------------------------------------------------------------------
# Demo: a 2-layer feedforward net (ReLU hidden + softmax head), 3-way classifier.
# relu / softmax / cross_entropy / mlp_forward are plain helpers with no autodiff
# rules -- each is instrumented on demand so gradients flow through them.
# ---------------------------------------------------------------------------
def relu(z: Tensor) -> Tensor:
    return np.maximum(z, 0.0)


def softmax(z: Tensor) -> Tensor:
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy(probs: Tensor, y_onehot: Tensor) -> Operand:
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))


def mlp_forward(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
    hidden = relu(x @ w1 + b1)
    return softmax(hidden @ w2 + b2)


def _make_blobs() -> Tuple[Array, Array, Array]:
    rng = np.random.default_rng(1)
    n = 30
    centers = np.array([[2.0, 2.0], [-2.0, 2.0], [0.0, -2.5]])
    X = np.vstack([rng.normal(loc=c, scale=0.5, size=(n, 2)) for c in centers])
    labels = np.repeat(np.arange(3), n)
    onehot = np.eye(3)[labels]
    return X, labels, onehot


_Xc, _labels, _Yoh = _make_blobs()


def mlp_loss(w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Operand:
    return cross_entropy(mlp_forward(_Xc, w1, b1, w2, b2), _Yoh)


def _init_mlp(
    rng: np.random.Generator, n_in: int = 2, n_hidden: int = 16, n_out: int = 3
) -> Tuple[Array, ...]:
    return (
        0.1 * rng.standard_normal((n_in, n_hidden)),
        np.zeros(n_hidden),
        0.1 * rng.standard_normal((n_hidden, n_out)),
        np.zeros(n_out),
    )


def _mlp_accuracy(params: Tuple[Array, ...]) -> float:
    w1, b1, w2, b2 = params
    logits = np.maximum(_Xc @ w1 + b1, 0.0) @ w2 + b2
    return float(np.mean(np.argmax(logits, axis=1) == _labels))


# ---------------------------------------------------------------------------
# Demo: the same 2-layer MLP, but with parameters as a *pytree* -- one nested
# dict instead of four positional arrays. ``value_and_grad`` flattens the dict,
# differentiates, and hands back a gradient dict of the same shape, so the whole
# SGD update is the one-liner ``sgd_update`` (a ``tree_map``). This is how a real
# framework lets a model own a structured bag of named parameters.
# ---------------------------------------------------------------------------
# A param pytree: ``{"hidden": {"w", "b"}, "out": {"w", "b"}}``. Annotated as a
# concrete nested dict (not the open ``PyTree`` union) so the body indexes it; the
# leaves are arrays at init and ``Var``s once ``value_and_grad`` wraps them, both
# of which ``Tensor`` covers.
MLPParams = Dict[str, Dict[str, Tensor]]


def mlp_tree_loss(params: MLPParams) -> Operand:
    h, o = params["hidden"], params["out"]
    probs = mlp_forward(_Xc, h["w"], h["b"], o["w"], o["b"])
    return cross_entropy(probs, _Yoh)


def _init_mlp_tree(
    rng: np.random.Generator, n_in: int = 2, n_hidden: int = 16, n_out: int = 3
) -> PyTree:
    return {
        "hidden": {
            "w": 0.1 * rng.standard_normal((n_in, n_hidden)),
            "b": np.zeros(n_hidden),
        },
        "out": {
            "w": 0.1 * rng.standard_normal((n_hidden, n_out)),
            "b": np.zeros(n_out),
        },
    }


def _mlp_tree_accuracy(params: PyTree) -> float:
    p = cast(Dict[str, Dict[str, Array]], params)
    h, o = p["hidden"], p["out"]
    logits = np.maximum(_Xc @ h["w"] + h["b"], 0.0) @ o["w"] + o["b"]
    return float(np.mean(np.argmax(logits, axis=1) == _labels))


# ---------------------------------------------------------------------------
# Demo: the same classifier with a LayerNorm and a Dropout layer. These "fancy"
# layers are plain helpers -- LayerNorm is stateless (gamma/beta are just
# differentiated params), and Dropout's mask is a sampled constant the gradient
# routes through. train vs eval is a flag in the calling code (functional style).
# ---------------------------------------------------------------------------
_dropout_rng = np.random.default_rng(0)


def linear(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    return x @ w + b


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered * centered).mean(axis=-1, keepdims=True)
    return centered / (var + eps) ** 0.5 * gamma + beta


def dropout(x: Tensor, keep: float, training: bool) -> Tensor:
    if not training:
        return x
    mask = (_dropout_rng.random(x.shape) < keep) / keep  # x.shape via Var.shape
    return x * mask  # mask is a plain-array constant; grad routes through it


def deep_forward(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    g: Tensor,
    beta: Tensor,
    w2: Tensor,
    b2: Tensor,
    training: bool,
) -> Tensor:
    hidden = relu(layer_norm(linear(x, w1, b1), g, beta))
    hidden = dropout(hidden, 0.9, training)
    return softmax(linear(hidden, w2, b2))


def deep_loss(
    w1: Tensor, b1: Tensor, g: Tensor, beta: Tensor, w2: Tensor, b2: Tensor
) -> Operand:
    return cross_entropy(deep_forward(_Xc, w1, b1, g, beta, w2, b2, True), _Yoh)


def _init_deep(
    rng: np.random.Generator, n_in: int = 2, n_hidden: int = 16, n_out: int = 3
) -> Tuple[Array, ...]:
    return (
        0.3 * rng.standard_normal((n_in, n_hidden)),
        np.zeros(n_hidden),
        np.ones(n_hidden),  # layernorm gamma
        np.zeros(n_hidden),  # layernorm beta
        0.3 * rng.standard_normal((n_hidden, n_out)),
        np.zeros(n_out),
    )


def _deep_accuracy(params: Tuple[Array, ...]) -> float:
    w1, b1, g, beta, w2, b2 = params
    probs = deep_forward(_Xc, w1, b1, g, beta, w2, b2, training=False)  # dropout off
    return float(np.mean(np.argmax(probs, axis=1) == _labels))


# ---------------------------------------------------------------------------
# Demo: a single-head Transformer encoder block and a tiny sequence classifier.
# attention / softmax_last / transformer_block are plain helpers (instrumented on
# demand): matmul + a stable softmax + optional masking + residual + LayerNorm +
# feed-forward -- the whole block differentiates with no special rules.
# ---------------------------------------------------------------------------
def softmax_last(z: Tensor) -> Tensor:
    z = z - np.max(z, axis=-1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[np.ndarray] = None
) -> Tensor:
    scores = (q @ k.T) * (q.shape[-1] ** -0.5)  # scaled dot-product
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # masked positions get ~0 weight
    return softmax_last(scores) @ v


def transformer_block(
    x: Tensor,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    g1: Tensor,
    beta1: Tensor,
    w1: Tensor,
    bff1: Tensor,
    w2: Tensor,
    bff2: Tensor,
    g2: Tensor,
    beta2: Tensor,
) -> Tensor:
    attended = attention(x @ wq, x @ wk, x @ wv) @ wo
    x = layer_norm(x + attended, g1, beta1)  # residual + LayerNorm
    ff = relu(x @ w1 + bff1) @ w2 + bff2
    return layer_norm(x + ff, g2, beta2)  # residual + LayerNorm


def softmax_ce(logits: Tensor, onehot: Tensor) -> Operand:
    return -np.sum(onehot * np.log(softmax_last(logits) + 1e-12))


def _seq_logits(x: Tensor, *params: Tensor) -> Tensor:
    *block, wc, bc = params
    pooled = transformer_block(x, *block).mean(axis=0)  # mean-pool over positions
    return pooled @ wc + bc


def _make_sequences() -> Tuple[List[Array], Array, Array]:
    rng = np.random.default_rng(3)
    seq_len, d_model, n_per = 3, 4, 8
    seqs: List[Array] = []
    labels: List[int] = []
    for cls, shift in enumerate((-0.8, 0.8)):
        for _ in range(n_per):
            seqs.append(rng.normal(shift, 0.5, size=(seq_len, d_model)))
            labels.append(cls)
    return seqs, np.array(labels), np.eye(2)[labels]


_SEQS, _SEQ_LABELS, _SEQ_OH = _make_sequences()


def transformer_loss(*params: Tensor) -> Operand:
    total: Operand = 0.0
    for i in range(len(_SEQS)):
        total = total + softmax_ce(_seq_logits(_SEQS[i], *params), _SEQ_OH[i])
    return total / len(_SEQS)


def _init_transformer(
    rng: np.random.Generator, d_model: int = 4, d_ff: int = 8, n_classes: int = 2
) -> Tuple[Array, ...]:
    s = 0.3
    rn = rng.standard_normal
    return (
        s * rn((d_model, d_model)),  # wq
        s * rn((d_model, d_model)),  # wk
        s * rn((d_model, d_model)),  # wv
        s * rn((d_model, d_model)),  # wo
        np.ones(d_model),  # layernorm-1 gamma
        np.zeros(d_model),  # layernorm-1 beta
        s * rn((d_model, d_ff)),  # ffn in
        np.zeros(d_ff),
        s * rn((d_ff, d_model)),  # ffn out
        np.zeros(d_model),
        np.ones(d_model),  # layernorm-2 gamma
        np.zeros(d_model),  # layernorm-2 beta
        s * rn((d_model, n_classes)),  # classifier head
        np.zeros(n_classes),
    )


def _transformer_accuracy(params: Tuple[Array, ...]) -> float:
    preds = [int(np.argmax(_seq_logits(x, *params))) for x in _SEQS]
    return float(np.mean(np.array(preds) == _SEQ_LABELS))


def sin_sq(x: Tensor) -> Operand:
    return np.sum(np.sin(x * x))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    (w, b), history = gradient_descent(
        logistic_loss, (np.zeros(2), 0.0), lr=0.5, steps=200
    )
    logger.info(
        "logistic regression: loss %.4f -> %.4f over %d steps",
        history[0],
        history[-1],
        len(history),
    )
    logger.info("  learned w=%s b=%.3f", np.round(w, 3), b)
    logger.info("  train accuracy: %.3f", _accuracy(w, b))

    # Same model via the declarative params(...) builder, with the bias frozen:
    # SGD trains w and leaves b pinned at its initial value. Params read by
    # attribute (model.w) as well as by key (model["w"]).
    lr_model: ParamDict = params(w=np.zeros(2), b=frozen(0.0))
    lr_vg = value_and_grad(logistic_param_loss)
    for _step in range(200):
        _loss, (g,) = lr_vg(lr_model)
        lr_model = cast(ParamDict, sgd_update(lr_model, g, lr=0.5))
    lr_w = cast(Param, lr_model.w).value
    lr_b = cast(Param, lr_model.b).value
    logger.info(
        "logistic regression via params(...) with frozen bias: acc %.3f, b held at %g",
        _accuracy(lr_w, lr_b),
        float(lr_b),
    )

    mlp_params = _init_mlp(np.random.default_rng(1))
    mlp_params, mlp_hist = gradient_descent(mlp_loss, mlp_params, lr=0.5, steps=300)
    logger.info(
        "2-layer MLP (relu + softmax), 3 classes: loss %.4f -> %.4f over %d steps",
        mlp_hist[0],
        mlp_hist[-1],
        len(mlp_hist),
    )
    logger.info("  train accuracy: %.3f", _mlp_accuracy(mlp_params))

    # Same MLP, parameters as a nested-dict pytree; SGD via tree_map (sgd_update).
    tree_params = _init_mlp_tree(np.random.default_rng(1))
    vg = value_and_grad(mlp_tree_loss)
    tree_hist: List[float] = []
    for _ in range(300):
        loss, (tree_grads,) = vg(tree_params)
        tree_hist.append(float(loss))
        tree_params = sgd_update(tree_params, tree_grads, lr=0.5)
    logger.info(
        "Same MLP, dict-pytree params (tree_map SGD): loss %.4f -> %.4f over %d steps",
        tree_hist[0],
        tree_hist[-1],
        len(tree_hist),
    )
    logger.info("  train accuracy: %.3f", _mlp_tree_accuracy(tree_params))

    deep_params = _init_deep(np.random.default_rng(2))
    deep_params, deep_hist = gradient_descent(deep_loss, deep_params, lr=0.3, steps=400)
    logger.info(
        "MLP + LayerNorm + Dropout, 3 classes: loss %.4f -> %.4f over %d steps",
        deep_hist[0],
        deep_hist[-1],
        len(deep_hist),
    )
    logger.info("  train accuracy: %.3f", _deep_accuracy(deep_params))

    tparams = _init_transformer(np.random.default_rng(4))
    tparams, t_hist = gradient_descent(transformer_loss, tparams, lr=0.2, steps=300)
    logger.info(
        "Transformer encoder block, 2-class sequences: loss %.4f -> %.4f over %d steps",
        t_hist[0],
        t_hist[-1],
        len(t_hist),
    )
    logger.info("  train accuracy: %.3f", _transformer_accuracy(tparams))

    xv = np.array([0.5, 1.0, 1.5])
    val, (g,) = value_and_grad(sin_sq)(xv)
    logger.info("sum(sin(x*x)):")
    logger.info("  autodiff grad = %s", np.round(cast(Array, g), 4))
    logger.info("  analytic grad = %s", np.round(2 * xv * np.cos(xv * xv), 4))
