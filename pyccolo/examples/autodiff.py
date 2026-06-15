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
import sysconfig
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

import pyccolo as pyc

logger = logging.getLogger(__name__)

# Type aliases. ``Var`` is the tape node; an ``Operand`` is anything the ops and
# primitives accept -- a ``Var`` or a plain number/array that is lifted to one.
Scalar = Union[int, float]
Array = np.ndarray
ArrayLike = Union[Scalar, Array]
Operand = Union["Var", ArrayLike]
# A differentiable tensor value: a ``Var`` during tracing, or a plain ndarray
# when the same helper is run outside the tape (e.g. at eval time).
Tensor = Union["Var", np.ndarray]
Axis = Optional[Union[int, Tuple[int, ...]]]


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
    def __getitem__(self, key: Any) -> "Var":
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

    def reshape(self, *shape: Any) -> "Var":
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

    def __array__(self, *args: Any, **kwargs: Any) -> Array:
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
    return x if isinstance(x, Var) else Var(x)


def detach(x: Operand) -> Var:
    """A fresh leaf with the same value but no gradient history (stop-gradient)."""
    return Var(x.value if isinstance(x, Var) else x)


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
    dtype: Any = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: Any,
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
    dtype: Any = None,
    out: Any = None,
    ddof: int = 0,
    keepdims: bool = False,
    **_: Any,
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


def d_reshape(x: Operand, *shape: Any) -> Var:
    x = _lift(x)
    newshape = shape[0] if len(shape) == 1 else shape
    out = Var(np.reshape(x.value, newshape), _parents=(x,))

    def _backward() -> None:
        x.grad = x.grad + out.grad.reshape(x.value.shape)

    out._backward = _backward
    return out


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
_RULES: Dict[Callable[..., Any], Tuple[Any, ...]] = {
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
    d_concatenate: (np.concatenate,),
}

_INTERCEPT: Dict[Any, Callable[..., Any]] = {
    fn: impl for impl, fns in _RULES.items() for fn in fns
}


class AutodiffWarning(UserWarning):
    """Emitted when a ``Var`` flows into a numpy/math function we cannot differentiate."""


def _contains_var(values: Iterable[Any]) -> bool:
    for v in values:
        if isinstance(v, Var):
            return True
        if isinstance(v, (list, tuple)) and any(isinstance(x, Var) for x in v):
            return True
    return False


def _is_mathy(func: Callable[..., Any]) -> bool:
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


def _is_user_function(func: Callable[..., Any]) -> bool:
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


_WRAPPERS: Dict[Callable[..., Any], Callable[..., Any]] = {}


def _warn_wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
    wrapper = _WRAPPERS.get(func)
    if wrapper is not None:
        return wrapper
    name = getattr(func, "__name__", repr(func))

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
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
    _helpers: Dict[Callable[..., Any], Callable[..., Any]] = {}

    def _instrument_helper(self, func: Callable[..., Any]) -> Callable[..., Any]:
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

    def resolve_call(self, func: Callable[..., Any]) -> Callable[..., Any]:
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
        self, func: Callable[..., Any], node: Any, *_: Any, **__: Any
    ) -> Callable[..., Any]:
        return self.resolve_call(func)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def resolve_call(func: Callable[..., Any]) -> Callable[..., Any]:
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
_INSTRUMENTED: Dict[Callable[..., Any], Callable[..., Any]] = {}


def _make_runner(f: Callable[..., Any]) -> Callable[..., Any]:
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
        except (OSError, TypeError, SyntaxError):
            pass

    @functools.wraps(f)
    def run_directly(*args: Any, **kwargs: Any) -> Any:
        with tracer.tracing_enabled():
            return f(*args, **kwargs)

    return run_directly


def value_and_grad(
    f: Callable[..., Any],
) -> Callable[..., Tuple[Array, Tuple[ArrayLike, ...]]]:
    """Wrap ``f`` so that calling it returns ``(value, grads)``.

    ``grads`` is a tuple holding the gradient of the (scalar) output with respect
    to each numeric/ndarray positional argument, in order. ``f`` may be an ordinary
    function or a pipescript ``|>`` pipe lambda (run it under ``PipelineTracer``).
    """
    runner = _INSTRUMENTED.get(f)
    if runner is None:
        runner = _make_runner(f)
        _INSTRUMENTED[f] = runner

    def wrapped(*args: Any) -> Tuple[Array, Tuple[ArrayLike, ...]]:
        arg_vars: List[Tuple[ArrayLike, Var]] = []
        call_args: List[Any] = []
        for a in args:
            if _is_numeric(a):
                v = Var(a)
                arg_vars.append((a, v))
                call_args.append(v)
            else:
                call_args.append(a)
        out = runner(*call_args)
        if isinstance(out, Var):
            out.backward()
            grads = tuple(_match_arg(orig, v.grad) for orig, v in arg_vars)
            return out.value, grads
        # Output does not depend on the inputs.
        grads = tuple(_match_arg(orig, np.zeros_like(v.value)) for orig, v in arg_vars)
        return np.asarray(out, dtype=float), grads

    return wrapped


def _match_arg(orig: ArrayLike, grad: Array) -> ArrayLike:
    return grad if isinstance(orig, np.ndarray) else float(grad)


def grad(f: Callable[..., Any]) -> Callable[..., Tuple[ArrayLike, ...]]:
    """Return a function computing just the gradient tuple of ``f``."""
    vg = value_and_grad(f)
    return lambda *args: vg(*args)[1]


def gradient_descent(
    loss_fn: Callable[..., Any],
    init_params: Tuple[ArrayLike, ...],
    lr: float = 0.1,
    steps: int = 100,
) -> Tuple[Tuple[Any, ...], List[float]]:
    """Minimize ``loss_fn(*params)`` by gradient descent; return (params, history)."""
    vg = value_and_grad(loss_fn)
    params = list(init_params)
    history: List[float] = []
    for _ in range(steps):
        loss, grads = vg(*params)
        history.append(float(loss))
        params = [p - lr * g for p, g in zip(params, grads)]
    return tuple(params), history


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

    params = _init_mlp(np.random.default_rng(1))
    params, mlp_hist = gradient_descent(mlp_loss, params, lr=0.5, steps=300)
    logger.info(
        "2-layer MLP (relu + softmax), 3 classes: loss %.4f -> %.4f over %d steps",
        mlp_hist[0],
        mlp_hist[-1],
        len(mlp_hist),
    )
    logger.info("  train accuracy: %.3f", _mlp_accuracy(params))

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
    logger.info("  autodiff grad = %s", np.round(g, 4))
    logger.info("  analytic grad = %s", np.round(2 * xv * np.cos(xv * xv), 4))
