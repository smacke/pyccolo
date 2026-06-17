# -*- coding: utf-8 -*-
"""
Tests for the reverse-mode autodiff example (``pyccolo/examples/autodiff.py``).
"""
import math
import warnings

import pytest

np = pytest.importorskip("numpy")

import pyccolo.examples.autodiff as ad  # noqa: E402
from pyccolo.examples.autodiff import (  # noqa: E402
    AutodiffWarning,
    Param,
    ParamDict,
    Var,
    Weight,
    _accuracy,
    _deep_accuracy,
    _init_deep,
    _init_mlp,
    _init_mlp_tree,
    _init_transformer,
    _mlp_accuracy,
    _mlp_tree_accuracy,
    _transformer_accuracy,
    attention,
    cross_entropy,
    deep_loss,
    detach,
    dropout,
    frozen,
    grad,
    gradient_descent,
    layer_norm,
    logistic_loss,
    mlp_loss,
    mlp_tree_loss,
    param_values,
    params,
    softmax,
    tied,
    transformer_block,
    transformer_loss,
    value_and_grad,
)

# Module-level data so target functions reference globals (not closures), which
# keeps them recompilable by ``tracer.instrumented`` (getsource + recompile).
_M = np.array([[1.0, 2.0, -1.0], [0.5, -2.0, 1.0]])  # (2, 3)


# --- target functions (module level => inspect.getsource works) -------------
def poly(x):
    return x * x + 3.0 * x - 2.0  # d/dx = 2x + 3


def ratio(x, y):
    return (x * y + 1.0) / (x + y)


def powfn(x):
    return x**3 + 2.0 * x**2  # d/dx = 3x^2 + 4x


def transcendental(x):
    return np.exp(np.sin(x) + 1.0) / (1.0 + np.log(x * x + 1.0))


def matmul_fn(w):  # w shape (3,);  _M @ w -> (2,)
    return np.sum(_M @ w)


def broadcast_fn(w, b):  # w (3,), scalar b broadcast over (2,)
    return np.sum(_M @ w + b)


def mean_fn(w):
    return np.mean(_M @ w)


def relu_fn(x):
    return np.sum(np.maximum(x, 0.0))


def just_np_exp(x):
    return np.exp(x)


def just_math_exp(x):
    return math.exp(x)


def uses_unsupported(x):
    return np.tan(x)  # no differentiation rule registered for np.tan


# --- user helper functions (no autodiff rules; differentiated via instrument-
# on-demand at the call site) ------------------------------------------------
def softplus(z):
    return np.log(1.0 + np.exp(z))


def through_helper(w):
    return np.sum(softplus(_M @ w))


def through_nested_helpers(w, b):
    return np.sum(softplus(_M @ w + b))


def bad_helper(z):
    return np.tan(z)  # unsupported inside a helper


def through_bad_helper(x):
    return bad_helper(x)


# --- finite-difference oracle ----------------------------------------------
def finite_diff(f, args, h=1e-6):
    # Reverse-mode backward seeds ones, so it yields the gradient of sum(output);
    # scalarize with np.sum so the oracle matches for scalar- and array-output f.
    def s(*a):
        return float(np.sum(f(*a)))

    base = [
        np.array(a, dtype=float) if isinstance(a, np.ndarray) else float(a)
        for a in args
    ]
    grads = []
    for i, a in enumerate(base):
        if isinstance(a, np.ndarray):
            g = np.zeros_like(a)
            for idx in np.ndindex(a.shape):
                up = [x.copy() if isinstance(x, np.ndarray) else x for x in base]
                dn = [x.copy() if isinstance(x, np.ndarray) else x for x in base]
                up[i][idx] += h
                dn[i][idx] -= h
                g[idx] = (s(*up) - s(*dn)) / (2 * h)
            grads.append(g)
        else:
            up, dn = list(base), list(base)
            up[i], dn[i] = a + h, a - h
            grads.append((s(*up) - s(*dn)) / (2 * h))
    return tuple(grads)


def _assert_grads_match(f, args, atol=1e-5):
    _, ad = value_and_grad(f)(*args)
    fd = finite_diff(f, args)
    assert len(ad) == len(fd)
    for g_ad, g_fd in zip(ad, fd):
        assert np.allclose(g_ad, g_fd, atol=atol), (g_ad, g_fd)


# --- scalar ops -------------------------------------------------------------
def test_polynomial():
    val, (g,) = value_and_grad(poly)(4.0)
    assert math.isclose(val, 4 * 4 + 3 * 4 - 2)
    assert math.isclose(g, 2 * 4 + 3)


def test_division_and_power():
    _assert_grads_match(ratio, (2.0, 3.0))
    _assert_grads_match(powfn, (1.5,))


# --- transcendentals via call interception ----------------------------------
def test_transcendental_chain_rule():
    _assert_grads_match(transcendental, (0.7,))


def test_transcendental_on_array():
    _assert_grads_match(transcendental, (np.array([0.3, 0.7, 1.2]),))


# --- linear algebra / reductions / broadcasting -----------------------------
def test_matmul_gradient():
    _assert_grads_match(matmul_fn, (np.array([0.5, -1.0, 2.0]),))
    # analytic: d/dw sum(M @ w) = M.T @ ones = column sums of M
    _, (g,) = value_and_grad(matmul_fn)(np.array([0.5, -1.0, 2.0]))
    assert np.allclose(g, _M.sum(axis=0))


def test_broadcasting_unbroadcast():
    w = np.array([0.5, -1.0, 2.0])
    _assert_grads_match(broadcast_fn, (w, 0.3))
    _, (gw, gb) = value_and_grad(broadcast_fn)(w, 0.3)
    assert gw.shape == w.shape
    assert np.isscalar(gb) or np.ndim(gb) == 0
    assert math.isclose(float(gb), 2.0)  # scalar b broadcast over 2 rows


def test_mean_gradient():
    _assert_grads_match(mean_fn, (np.array([1.0, -1.0, 0.5]),))


def test_relu_subgradient():
    x = np.array([-2.0, 3.0, -0.5, 4.0])
    _, (g,) = value_and_grad(relu_fn)(x)
    assert np.allclose(g, (x > 0).astype(float))


# --- arbitrary shapes / broadcasting / extra primitives ---------------------
_RNG = np.random.default_rng(7)
_A31 = _RNG.standard_normal((3, 1))
_B14 = _RNG.standard_normal((1, 4))
_T = _RNG.standard_normal((2, 3, 4))
_BL = _RNG.standard_normal((4, 5))


def broadcast_elemwise(a, b):  # (3,1) and (1,4) -> (3,4)
    return np.sum((a * b + a) ** 2)


def batched_matmul(t):  # (2,3,4) @ (4,5) -> (2,3,5)
    return np.sum(t @ _BL)


def reduce_max(t):
    return np.sum(np.max(t, axis=1))


def softmax_stable(z):  # uses np.max for numerical stability
    shifted = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(shifted)
    return np.sum(np.log(np.sum(e, axis=1)))


def clip_where(x):
    pos = np.where(x > 0, x, np.clip(x, -0.5, 0.0))
    return np.sum(pos * pos)


def gather(x):  # fancy indexing with a duplicate row -> scatter-add backward
    return np.sum(x[[0, 0, 2]] ** 2)


def concat_fn(a, b):
    return np.sum(np.concatenate([a, b], axis=0) ** 2)


def test_broadcasting_arbitrary_rank():
    _assert_grads_match(broadcast_elemwise, (_A31, _B14))


def test_batched_matmul():
    _assert_grads_match(batched_matmul, (_T,))


def test_max_reduction_subgradient():
    _assert_grads_match(reduce_max, (_RNG.standard_normal((4, 5)),))


def test_stable_softmax_with_max():
    _assert_grads_match(softmax_stable, (_RNG.standard_normal((3, 4)),))


def test_where_and_clip():
    _assert_grads_match(clip_where, (_RNG.standard_normal((3, 4)),), atol=1e-4)


def test_fancy_index_scatter_add():
    _assert_grads_match(gather, (_RNG.standard_normal((3, 2)),))


def test_concatenate_split_backward():
    _assert_grads_match(
        concat_fn, (_RNG.standard_normal((2, 3)), _RNG.standard_normal((4, 3)))
    )


# --- the stack family (composed from concatenate + reshape) -----------------
def stack_fn(a, b):
    return np.sum(np.stack([a, b]) ** 2)  # new axis 0


def stack_axis1_fn(a, b):
    return np.sum(np.stack([a, b], axis=1) * np.arange(2).reshape(1, 2, 1))


def vstack_fn(a, b):  # 1-D rows -> (2, n)
    return np.sum(np.vstack([a, b]) ** 2)


def vstack_2d_fn(a, b):  # 2-D -> concatenate axis 0
    return np.sum(np.vstack([a, b]) * 3.0)


def hstack_1d_fn(a, b):  # 1-D -> concatenate axis 0
    return np.sum(np.hstack([a, b]) ** 2)


def hstack_2d_fn(a, b):  # 2-D -> concatenate axis 1
    return np.sum(np.hstack([a, b]) ** 2)


def column_stack_fn(a, b):  # 1-D columns -> (n, 2)
    return np.sum(np.column_stack([a, b]) ** 2)


def dstack_fn(a, b):  # 2-D -> (m, n, 2)
    return np.sum(np.dstack([a, b]) ** 2)


def test_stack_new_axis():
    _assert_grads_match(stack_fn, (_RNG.standard_normal(4), _RNG.standard_normal(4)))
    _assert_grads_match(
        stack_axis1_fn, (_RNG.standard_normal((3, 2)), _RNG.standard_normal((3, 2)))
    )


def test_vstack():
    _assert_grads_match(vstack_fn, (_RNG.standard_normal(4), _RNG.standard_normal(4)))
    _assert_grads_match(
        vstack_2d_fn, (_RNG.standard_normal((2, 3)), _RNG.standard_normal((4, 3)))
    )


def test_hstack():
    _assert_grads_match(
        hstack_1d_fn, (_RNG.standard_normal(3), _RNG.standard_normal(5))
    )
    _assert_grads_match(
        hstack_2d_fn, (_RNG.standard_normal((2, 3)), _RNG.standard_normal((2, 4)))
    )


def test_column_stack():
    _assert_grads_match(
        column_stack_fn, (_RNG.standard_normal(4), _RNG.standard_normal(4))
    )


def test_dstack():
    _assert_grads_match(
        dstack_fn, (_RNG.standard_normal((2, 3)), _RNG.standard_normal((2, 3)))
    )


def test_stack_values_match_numpy():
    # the differentiable versions must also compute the same values as numpy
    a, b = _RNG.standard_normal((2, 3)), _RNG.standard_normal((2, 3))
    va, vb = Var(a), Var(b)
    assert np.allclose(ad.d_stack([va, vb], axis=1).value, np.stack([a, b], axis=1))
    assert np.allclose(ad.d_vstack([va, vb]).value, np.vstack([a, b]))
    assert np.allclose(ad.d_hstack([va, vb]).value, np.hstack([a, b]))
    assert np.allclose(ad.d_dstack([va, vb]).value, np.dstack([a, b]))
    c, dd = _RNG.standard_normal(4), _RNG.standard_normal(4)
    assert np.allclose(
        ad.d_column_stack([Var(c), Var(dd)]).value, np.column_stack([c, dd])
    )


# --- the pyccolo-justifying behaviour ---------------------------------------
def test_interception_is_required():
    # Without interception, a Var hitting a numpy ufunc / math C-func fails LOUD
    # (we deliberately omit __array__ / __float__ to avoid silent wrong gradients).
    with pytest.raises(TypeError):
        np.exp(Var(1.0))
    with pytest.raises(TypeError):
        math.exp(Var(1.0))

    # Through value_and_grad, before_call swaps the function and it differentiates.
    val, (g,) = value_and_grad(just_np_exp)(2.0)
    assert math.isclose(val, math.exp(2.0), rel_tol=1e-9)
    assert math.isclose(g, math.exp(2.0), rel_tol=1e-9)

    # The same interception routes scalar math.exp through the numpy primitive.
    val, (g,) = value_and_grad(just_math_exp)(2.0)
    assert math.isclose(val, math.exp(2.0), rel_tol=1e-9)
    assert math.isclose(g, math.exp(2.0), rel_tol=1e-9)


def test_grad_helper():
    (g,) = grad(poly)(4.0)
    assert math.isclose(g, 2 * 4 + 3)


def test_warns_on_unsupported_function():
    # A Var reaching an un-ruled numpy function warns (and then fails loud, since
    # np.tan is a ufunc and Var opts out of ufuncs). Record so the warning is
    # captured even though a TypeError follows.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):
            value_and_grad(uses_unsupported)(0.5)
    assert any(issubclass(w.category, AutodiffWarning) for w in rec), [
        str(w.message) for w in rec
    ]


def test_no_spurious_warnings_on_supported_code():
    # Differentiating code that only uses intercepted funcs + operators must be
    # warning-free (no false positives from builtins / dunder dispatch).
    with warnings.catch_warnings():
        warnings.simplefilter("error", AutodiffWarning)
        value_and_grad(transcendental)(0.7)
        gradient_descent(logistic_loss, (np.zeros(2), 0.0), lr=0.5, steps=3)


# --- differentiating through user helper functions --------------------------
def test_differentiates_through_helper():
    # softplus has no rule; it is instrumented on demand so the tape flows through.
    _assert_grads_match(through_helper, (np.array([0.3, -0.2, 0.1]),))


def test_differentiates_through_nested_helpers():
    _assert_grads_match(through_nested_helpers, (np.array([0.3, -0.2, 0.1]), 0.4))


def test_no_warning_through_supported_helper():
    with warnings.catch_warnings():
        warnings.simplefilter("error", AutodiffWarning)
        value_and_grad(through_helper)(np.array([0.3, -0.2, 0.1]))


def test_warning_propagates_into_helpers():
    # Interception/warning still applies inside an instrumented helper.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):
            value_and_grad(through_bad_helper)(0.5)
    assert any(issubclass(w.category, AutodiffWarning) for w in rec)


# --- 2-layer MLP (relu hidden + softmax head) -------------------------------
def test_mlp_gradients_match_finite_diff():
    # Small architecture; gradients flow through relu/softmax/cross_entropy
    # helpers. ReLU kinks make finite differences slightly noisier -> atol=1e-4.
    rng = np.random.default_rng(0)
    w1 = 0.3 * rng.standard_normal((2, 4))
    b1 = 0.1 * rng.standard_normal(4)
    w2 = 0.3 * rng.standard_normal((4, 3))
    b2 = 0.1 * rng.standard_normal(3)
    _assert_grads_match(mlp_loss, (w1, b1, w2, b2), atol=1e-4)


def test_mlp_trains_to_high_accuracy():
    params = _init_mlp(np.random.default_rng(1))
    params, history = gradient_descent(mlp_loss, params, lr=0.5, steps=300)
    assert history[-1] < history[0]
    assert history[-1] < 0.05
    assert _mlp_accuracy(params) >= 0.97


# --- conveniences: shape metadata, detach, var/std --------------------------
def test_var_shape_metadata():
    v = Var(np.zeros((2, 3)))
    assert v.shape == (2, 3)
    assert v.ndim == 2
    assert v.size == 6


def detach_fn(x):
    # d/dx sum(x * stop_grad(x)) == stop_grad(x) == x, NOT 2x
    return np.sum(x * detach(x))


def test_detach_stops_gradient():
    x = np.array([1.0, 2.0, 3.0])
    _, (g,) = value_and_grad(detach_fn)(x)
    assert np.allclose(g, x)  # would be 2x if the gradient flowed through detach


def var_fn(x):
    return np.var(x, axis=1)


def std_fn(x):
    return np.std(x)


def test_var_std_gradients():
    rng = np.random.default_rng(0)
    _assert_grads_match(var_fn, (rng.standard_normal((3, 4)),))
    _assert_grads_match(std_fn, (rng.standard_normal((6,)),), atol=1e-4)


# --- "fancy" layers: LayerNorm + Dropout ------------------------------------
def test_layer_norm_gradients():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5))
    gamma = rng.standard_normal((5,))
    beta = rng.standard_normal((5,))
    _assert_grads_match(layer_norm, (x, gamma, beta), atol=1e-4)


def dropout_eval(x):
    return dropout(x, 0.5, training=False)


def test_dropout_eval_is_identity():
    x = np.array([1.0, -2.0, 3.0])
    val, (g,) = value_and_grad(dropout_eval)(x)
    assert np.allclose(val, x)
    assert np.allclose(g, np.ones_like(x))


def test_dropout_train_routes_gradient_through_mask():
    x = Var(np.array([1.0, 2.0, 3.0, 4.0]))
    ad._dropout_rng = np.random.default_rng(0)
    out = dropout(x, 0.8, training=True)
    out.backward()
    mask = (np.random.default_rng(0).random((4,)) < 0.8) / 0.8
    assert np.allclose(out.value, x.value * mask)
    assert np.allclose(x.grad, mask)  # gradient zero on dropped units


def test_deep_mlp_with_layernorm_dropout_trains():
    ad._dropout_rng = np.random.default_rng(0)  # deterministic masks
    params = _init_deep(np.random.default_rng(2))
    params, history = gradient_descent(deep_loss, params, lr=0.3, steps=400)
    assert history[-1] < history[0]
    assert history[-1] < 0.1
    assert _deep_accuracy(params) >= 0.97


# --- attention / transformer ------------------------------------------------
def attn(q, k, v):
    return attention(q, k, v)


def test_attention_gradients():
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((3, 4)) for _ in range(3))
    _assert_grads_match(attn, (q, k, v), atol=1e-4)


_AQ = np.random.default_rng(1).standard_normal((3, 4))
_AK = np.random.default_rng(2).standard_normal((3, 4))
_CAUSAL = np.tril(np.ones((3, 3))).astype(bool)


def attn_first_row(v):
    # query/key fixed; differentiate the first output row wrt the values
    return attention(_AQ, _AK, v, _CAUSAL)[0]


def test_attention_causal_mask_blocks_future():
    v = np.random.default_rng(3).standard_normal((3, 4))
    _, (g,) = value_and_grad(attn_first_row)(v)
    # position 0 may only attend to position 0 -> no gradient to later values
    assert np.allclose(g[1:], 0.0)
    assert not np.allclose(g[0], 0.0)


def block_out(x, *block):
    return transformer_block(x, *block)


def test_transformer_block_gradients():
    block = _init_transformer(np.random.default_rng(5))[:12]  # block params only
    x = np.random.default_rng(6).standard_normal((3, 4))
    _assert_grads_match(block_out, (x, *block), atol=1e-3)


def test_transformer_classifier_trains():
    params = _init_transformer(np.random.default_rng(4))
    params, history = gradient_descent(transformer_loss, params, lr=0.2, steps=250)
    assert history[-1] < history[0]
    assert _transformer_accuracy(params) >= 0.9


# --- end-to-end: logistic regression ----------------------------------------
def test_logistic_regression_trains():
    (w, b), history = gradient_descent(
        logistic_loss, (np.zeros(2), 0.0), lr=0.5, steps=200
    )
    assert history[-1] < history[0]
    assert history[-1] < 0.05
    assert _accuracy(w, b) == 1.0


def test_logistic_gradient_matches_finite_diff():
    _assert_grads_match(logistic_loss, (np.zeros(2), 0.0))
    _assert_grads_match(logistic_loss, (np.array([0.3, -0.4]), 0.1))


# --- pytrees ----------------------------------------------------------------
# Target functions live at module scope (a real file) so ``value_and_grad`` can
# getsource + recompile them.
def sum_sq(x):
    return np.sum(x * x)


def dict_quadratic(p):  # ignores p["tag"] -> its gradient leaf comes back None
    return np.sum(p["x"] * p["x"]) + 3.0 * np.sum(p["y"])


def _trees_equal(a, b):
    la, ta = ad.tree_flatten(a)
    lb, tb = ad.tree_flatten(b)
    return (
        ta == tb
        and len(la) == len(lb)
        and all(np.array_equal(x, y) for x, y in zip(la, lb))
    )


def _tree_finite_diff(f, tree, h=1e-6):
    """Central-difference gradient of ``sum(f(tree))`` w.r.t. each numeric leaf."""
    leaves, treedef = ad.tree_flatten(tree)

    def loss(ls):
        return float(np.sum(f(ad.tree_unflatten(treedef, ls))))

    grads = []
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, np.ndarray):
            g = np.zeros_like(leaf)
            for idx in np.ndindex(leaf.shape):
                up = [x.copy() if isinstance(x, np.ndarray) else x for x in leaves]
                dn = [x.copy() if isinstance(x, np.ndarray) else x for x in leaves]
                up[i][idx] += h
                dn[i][idx] -= h
                g[idx] = (loss(up) - loss(dn)) / (2 * h)
            grads.append(g)
        elif isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            up, dn = list(leaves), list(leaves)
            up[i], dn[i] = leaf + h, leaf - h
            grads.append((loss(up) - loss(dn)) / (2 * h))
        else:
            grads.append(None)  # non-numeric leaf: no gradient
    return grads


def test_tree_flatten_unflatten_roundtrip():
    tree = {"a": [1.0, np.array([2.0, 3.0])], "b": (4.0,), "c": {"d": 5.0}}
    leaves, treedef = ad.tree_flatten(tree)
    rebuilt = ad.tree_unflatten(treedef, leaves)
    assert ad.tree_structure(rebuilt) == treedef
    assert _trees_equal(rebuilt, tree)


def test_tree_structure_equality_by_shape():
    a = {"w": np.zeros(3), "b": 0.0}
    b = {"w": np.ones(5), "b": 1.0}  # same shape, different values/dtype
    c = {"w": np.zeros(3)}  # missing a key -> different shape
    d = [np.zeros(3), 0.0]  # list, not dict
    assert ad.tree_structure(a) == ad.tree_structure(b)
    assert ad.tree_structure(a) != ad.tree_structure(c)
    assert ad.tree_structure(a) != ad.tree_structure(d)


def test_tree_leaves_order_is_sorted_by_key():
    leaves = ad.tree_leaves({"b": 2.0, "a": 1.0, "c": 3.0})
    assert leaves == [1.0, 2.0, 3.0]


def test_tree_map_leafwise_multiarg():
    a = {"x": np.array([1.0, 2.0]), "y": 3.0}
    b = {"x": np.array([10.0, 20.0]), "y": 30.0}
    out = ad.tree_map(lambda p, q: p + q, a, b)
    assert np.allclose(out["x"], [11.0, 22.0])
    assert out["y"] == 33.0


def test_sgd_update_matches_manual_and_preserves_structure():
    params = {"w": np.array([1.0, 2.0, 3.0]), "b": 0.5}
    grads = {"w": np.array([0.1, 0.2, 0.3]), "b": 1.0}
    out = ad.sgd_update(params, grads, lr=0.1)
    assert ad.tree_structure(out) == ad.tree_structure(params)
    assert np.allclose(out["w"], params["w"] - 0.1 * grads["w"])
    assert np.isclose(out["b"], 0.5 - 0.1 * 1.0)


def test_bare_arg_is_single_leaf_pytree():
    # A bare array is a one-leaf pytree, so its gradient comes back bare (this is
    # what keeps the positional API backward compatible).
    _, (g,) = value_and_grad(sum_sq)(np.array([1.0, 2.0, 3.0]))
    assert isinstance(g, np.ndarray)
    assert np.allclose(g, 2 * np.array([1.0, 2.0, 3.0]))


def test_grad_pytree_matches_structure_and_finite_diff():
    p = {"x": np.array([0.5, -1.0, 2.0]), "y": np.array([1.0, 3.0]), "tag": "ignored"}
    _, (gtree,) = value_and_grad(dict_quadratic)(p)
    assert ad.tree_structure(gtree) == ad.tree_structure(p)
    assert gtree["tag"] is None  # non-numeric leaf -> no gradient
    g_leaves = ad.tree_leaves(gtree)
    fd_leaves = _tree_finite_diff(dict_quadratic, p)
    for g, fd in zip(g_leaves, fd_leaves):
        if fd is None:
            assert g is None
        else:
            assert np.allclose(g, fd, atol=1e-5)


def test_value_and_grad_over_dict_mlp_matches_finite_diff():
    params = _init_mlp_tree(np.random.default_rng(0), n_hidden=4)
    _, (gtree,) = value_and_grad(mlp_tree_loss)(params)
    assert ad.tree_structure(gtree) == ad.tree_structure(params)
    g_leaves = ad.tree_leaves(gtree)
    fd_leaves = _tree_finite_diff(mlp_tree_loss, params)
    for g, fd in zip(g_leaves, fd_leaves):
        assert np.allclose(g, fd, atol=1e-4)


def test_dict_pytree_mlp_trains():
    params = _init_mlp_tree(np.random.default_rng(1))
    vg = value_and_grad(mlp_tree_loss)
    first = last = None
    for _ in range(200):
        loss, (g,) = vg(params)
        first = float(loss) if first is None else first
        last = float(loss)
        params = ad.sgd_update(params, g, lr=0.5)
    assert last < first
    assert _mlp_tree_accuracy(params) >= 0.9


# --- Param leaves: frozen / tied / ownership --------------------------------
# Target functions at module scope so value_and_grad can recompile them.
def param_quadratic(p):  # p["w"] trainable, p["b"] frozen
    return np.sum(p["w"] * p["w"]) + np.sum(p["b"])


def tied_quadratic(p):  # a and b are the same tied weight, used twice
    return np.sum(p["a"] * p["a"]) + np.sum(p["b"] * p["b"])


def param_sum(p):  # body irrelevant: the ownership guard fires before the call
    return np.sum(p["w"])


def reg_loss(p):  # minimize over w only; b is frozen
    pred = _M @ p["w"] + p["b"]
    return np.sum(pred * pred)


def test_trainable_param_matches_bare_array():
    # A bare array and a default Param around it differentiate identically.
    x = np.array([1.0, 2.0, 3.0])
    _, (g_bare,) = value_and_grad(sum_sq)(x)
    _, (g_param,) = value_and_grad(sum_sq)(Param(x))
    assert np.allclose(g_bare, g_param)


def test_frozen_param_has_no_gradient():
    p = {"w": Param(np.array([1.0, 2.0, 3.0])), "b": frozen(np.array([1.0, 2.0]))}
    _, (g,) = value_and_grad(param_quadratic)(p)
    assert ad.tree_structure(g) == ad.tree_structure(p)
    assert np.allclose(g["w"], 2 * np.array([1.0, 2.0, 3.0]))
    assert g["b"] is None  # frozen leaf -> no gradient


def test_tied_params_share_one_weight():
    # a and b tied -> one weight used twice: loss = 2*sum(w^2), dL/dw = 4w at both.
    w = np.array([1.0, 2.0])
    p = {"a": tied("shared", w), "b": tied("shared", w)}
    val, (g,) = value_and_grad(tied_quadratic)(p)
    assert np.allclose(val, 2 * np.sum(w * w))
    assert np.allclose(g["a"], 4 * w)
    assert np.allclose(g["b"], 4 * w)


def test_sgd_update_preserves_param_and_holds_frozen_fixed():
    p = {"w": Param(np.array([1.0, 2.0])), "b": frozen(np.array([5.0, 6.0]))}
    g = {"w": np.array([1.0, 1.0]), "b": None}
    out = ad.sgd_update(p, g, lr=0.1)
    assert isinstance(out["w"], Param) and out["w"].trainable
    assert np.allclose(out["w"].value, [0.9, 1.9])
    assert isinstance(out["b"], Param) and not out["b"].trainable
    assert np.allclose(out["b"].value, [5.0, 6.0])  # frozen leaf untouched


def test_frozen_param_unchanged_during_training():
    p = {"w": Param(np.array([1.0, -1.0, 0.5])), "b": frozen(np.array([2.0, -2.0]))}
    b0 = p["b"].value.copy()
    vg = value_and_grad(reg_loss)
    first = last = None
    for _ in range(50):
        loss, (g,) = vg(p)
        first = float(loss) if first is None else first
        last = float(loss)
        p = ad.sgd_update(p, g, lr=0.01)
    assert last < first  # trainable w drives the loss down
    assert isinstance(p["b"], Param) and not p["b"].trainable
    assert np.allclose(p["b"].value, b0)  # frozen bias never moved


def test_tied_params_stay_equal_during_training():
    w = np.array([0.5, -0.5])
    p = {"a": tied("k", w), "b": tied("k", w)}
    vg = value_and_grad(tied_quadratic)
    for _ in range(20):
        _, (g,) = vg(p)
        p = ad.sgd_update(p, g, lr=0.1)
    assert np.allclose(p["a"].value, p["b"].value)


def test_block_owned_param_reused_as_leaf_raises():
    # Simulate a params{...} block stamping ``origin``; reusing that Param object
    # as a second leaf is a double-owned weight.
    shared = Param(np.array([1.0, 2.0]))
    shared.origin = "blockA"
    with pytest.raises(ValueError, match="single owner"):
        value_and_grad(param_sum)({"w": shared, "extra": shared})


def test_block_owned_value_aliased_by_handwritten_param_raises():
    w = np.array([1.0, 2.0])
    block = Param(w)
    block.origin = "blockA"
    hand = Param(w)  # distinct Param, same underlying value array
    with pytest.raises(ValueError, match="single owner"):
        value_and_grad(param_sum)({"a": block, "b": hand})


def test_plain_duplicate_param_without_origin_is_allowed():
    # Two hand Params sharing a value but with no block origin: not a guard error
    # (only block-declared weights are policed). Both differentiate independently.
    w = np.array([1.0, 2.0])
    p = {"a": Param(w), "b": Param(w)}
    val, (g,) = value_and_grad(tied_quadratic)(p)
    assert np.allclose(val, 2 * np.sum(w * w))
    assert np.allclose(g["a"], 2 * w)  # independent leaves -> 2w each, not 4w
    assert np.allclose(g["b"], 2 * w)


# --- the declarative params(...) builder ------------------------------------
def two_arg_sum(p, q):  # body irrelevant: the ownership guard fires first
    return np.sum(p["w"])


def test_params_wraps_bare_numerics_as_trainable():
    w, b = np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.5])
    model = params(w=w, b=b)
    assert all(isinstance(model[k], Param) and model[k].trainable for k in model)
    # differentiates exactly like a hand-built dict of bare arrays
    _, (g_model,) = value_and_grad(reg_loss)(model)
    _, (g_plain,) = value_and_grad(reg_loss)({"w": w, "b": b})
    assert np.allclose(g_model["w"], g_plain["w"])
    assert np.allclose(g_model["b"], g_plain["b"])


def test_params_marks_frozen_and_tied():
    w0 = np.array([1.0, 2.0])
    model = params(
        w=np.array([1.0, 2.0, 3.0]),
        b=frozen(np.zeros(2)),
        a=tied("k", w0),
        c=tied("k", w0),
    )
    assert model["w"].trainable
    assert not model["b"].trainable
    assert model["a"].tie == "k" and model["c"].tie == "k"
    # every leaf carries this block's single (shared) origin token
    origins = {id(model[k].origin) for k in model}
    assert len(origins) == 1 and model["w"].origin is not None


def test_params_nests():
    model = params(enc={"w": np.zeros((2, 2)), "b": frozen(np.zeros(2))}, scale=2.0)
    assert isinstance(model["enc"]["w"], Param) and model["enc"]["w"].trainable
    assert not model["enc"]["b"].trainable
    assert isinstance(model["scale"], Param)


def test_param_values_strips_wrappers():
    model = params(w=np.array([1.0, 2.0]), nested={"u": np.array([3.0])})
    raw = param_values(model)
    assert type(raw["w"]) is np.ndarray and np.allclose(raw["w"], [1.0, 2.0])
    assert np.allclose(raw["nested"]["u"], [3.0])


def test_params_rejects_mapping_and_kwargs_together():
    with pytest.raises(TypeError):
        params({"w": np.zeros(2)}, b=np.zeros(2))


def test_params_accepts_positional_mapping():
    model = params({"weird key": np.zeros(2)})  # non-identifier name
    assert isinstance(model["weird key"], Param)


def test_params_block_weight_reused_as_leaf_raises():
    model = params(w=np.array([1.0, 2.0, 3.0]), b=np.zeros(3))
    # hand the same declared weight back as a separate leaf -> two owners
    with pytest.raises(ValueError, match="single owner"):
        value_and_grad(two_arg_sum)(model, {"alias": model["w"]})


def test_frozen_bracket_matches_call_form():
    x = np.array([1.0, 2.0, 3.0])
    a, b = frozen[x], frozen(x)
    assert isinstance(a, Param) and not a.trainable
    assert isinstance(b, Param) and not b.trainable
    assert np.allclose(a.value, x)


def test_tied_bracket_ties_to_sibling_by_reference():
    # tied[w] ties to the sibling `w` -- name only, no restated initializer.
    w = np.array([1.0, 2.0])
    model = params(a=w, b=tied[w])
    assert model["a"].tie is not None
    assert model["a"].tie == model["b"].tie  # one shared weight
    # one weight used twice: loss = 2*sum(w^2), dL/dw = 4w reported at both slots
    val, (g,) = value_and_grad(tied_quadratic)(model)
    assert np.allclose(val, 2 * np.sum(w * w))
    assert np.allclose(g["a"], 4 * w)
    assert np.allclose(g["b"], 4 * w)


def test_tied_bracket_unknown_reference_raises():
    stranger = np.array([1.0, 2.0])  # not declared in this params() call
    with pytest.raises(ValueError, match="reference another parameter"):
        params(a=np.zeros(2), b=tied[stranger])


def test_tie_ref_outside_params_raises():
    w = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="tied"):
        value_and_grad(sum_sq)(tied[w])


def test_params_supports_attribute_access():
    model = params(w=np.array([1.0, 2.0]), b=frozen(np.zeros(2)))
    assert isinstance(model, ad.ParamDict)
    assert model.w is model["w"]  # attribute access mirrors the item
    assert isinstance(model.b, Param) and not model.b.trainable
    with pytest.raises(AttributeError):
        model.nonexistent


def test_params_attribute_access_is_nested():
    model = params(enc={"w": np.zeros((2, 2))}, scale=2.0)
    assert isinstance(model.enc, ad.ParamDict)
    assert model.enc.w is model["enc"]["w"]


def test_attribute_access_survives_value_and_grad_and_sgd_update():
    model = params(w=np.array([1.0, -1.0, 0.5]), b=frozen(np.array([2.0, -2.0])))
    vg = value_and_grad(reg_loss)
    for _ in range(5):
        _, (g,) = vg(model)
        assert isinstance(g, ad.ParamDict)  # gradient pytree is attribute-accessible
        assert g.b is None  # frozen leaf -> no gradient, reached by attribute
        model = ad.sgd_update(model, g, lr=0.01)
        assert isinstance(model, ad.ParamDict)  # type preserved across the step
    assert np.allclose(model.b.value, [2.0, -2.0])  # frozen, read by attribute
    assert isinstance(model.w, Param)


def test_params_trains_with_frozen_layer():
    model = params(w=np.array([1.0, -1.0, 0.5]), b=frozen(np.array([2.0, -2.0])))
    b0 = model["b"].value.copy()
    vg = value_and_grad(reg_loss)
    first = last = None
    for _ in range(50):
        loss, (g,) = vg(model)
        first = float(loss) if first is None else first
        last = float(loss)
        model = ad.sgd_update(model, g, lr=0.01)
    assert last < first
    assert np.allclose(model["b"].value, b0)  # frozen leaf stayed put


# --- ambient weights: `with weights:` proxy injection + tape-style training ----
# Forwards reference the weight names unqualified; `with weights:` injects a Weight
# proxy for each into this module's globals, so these resolve at call time.
_PX = np.random.default_rng(5).standard_normal((6, 3))
_PY = np.eye(2)[np.random.default_rng(6).integers(0, 2, size=6)]
_PW = 0.1 * np.random.default_rng(7).standard_normal((3, 2))
_PB = np.zeros(2)


def proxy_forward(x):
    return softmax(x @ w + b)  # noqa: F821  (w, b injected by `with weights:`)


def proxy_loss():
    return cross_entropy(proxy_forward(_PX), _PY)


def _softmax_ce_fd(W, B, h=1e-6):
    def raw(Wm, Bm):
        z = _PX @ Wm + Bm
        e = np.exp(z - z.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        return float(-np.mean(np.sum(_PY * np.log(p + 1e-12), axis=1)))

    gW = np.zeros_like(W)
    for i in np.ndindex(W.shape):
        a, c = W.copy(), W.copy()
        a[i] += h
        c[i] -= h
        gW[i] = (raw(a, B) - raw(c, B)) / (2 * h)
    gB = np.zeros_like(B)
    for i in np.ndindex(B.shape):
        a, c = B.copy(), B.copy()
        a[i] += h
        c[i] -= h
        gB[i] = (raw(W, a) - raw(W, c)) / (2 * h)
    return gW, gB


def test_proxy_operators_forward_to_live_value():
    weights = params(w=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    wp = Weight(weights, "w")
    x = np.array([1.0, -1.0, 2.0])
    # inference: live value is the array -> plain numpy result, no Var
    assert type(x @ wp) is np.ndarray
    assert np.allclose(x @ wp, x @ weights["w"].value)
    assert np.allclose((wp + 1.0), weights["w"].value + 1.0)
    assert np.allclose(wp.T, weights["w"].value.T)
    assert wp.shape == (3, 2)


def test_proxy_ufunc_np_exp_both_modes():
    weights = params(w=np.array([0.5, 1.0, 1.5]))
    wp = Weight(weights, "w")
    out = np.exp(wp)  # inference: bare proxy straight into a ufunc
    assert type(out) is np.ndarray and np.allclose(out, np.exp([0.5, 1.0, 1.5]))
    v = Var(np.array([0.5, 1.0, 1.5]))
    weights._live = {"w": v}  # training: live value is a Var
    out2 = np.exp(wp)
    out2.backward()
    assert isinstance(out2, Var) and np.allclose(v.grad, np.exp([0.5, 1.0, 1.5]))
    weights._live = None


def test_proxy_array_function_np_sum_both_modes():
    weights = params(w=np.array([1.0, 2.0, 3.0]))
    wp = Weight(weights, "w")
    assert np.allclose(np.sum(wp), 6.0)  # inference
    v = Var(np.array([1.0, 2.0, 3.0]))
    weights._live = {"w": v}  # training
    s = np.sum(wp)
    s.backward()
    assert isinstance(s, Var) and np.allclose(v.grad, np.ones(3))
    weights._live = None


def test_with_weights_injects_and_removes_globals():
    weights = params(w=np.zeros(2), b=np.zeros(3))
    assert "w" not in globals() and "b" not in globals()
    with weights:
        assert isinstance(globals()["w"], Weight)
        assert isinstance(globals()["b"], Weight)
        assert globals()["w"]._owner is weights
    assert "w" not in globals() and "b" not in globals()  # cleaned up on exit


def test_with_weights_collision_raises():
    weights = params(taken=np.zeros(2))
    globals()["taken"] = 123  # a pre-existing, unrelated global
    try:
        with pytest.raises(ValueError, match="already exists"):
            with weights:
                pass
        assert globals()["taken"] == 123  # untouched
    finally:
        del globals()["taken"]


def test_with_weights_grad_matches_finite_diff():
    weights = params(w=_PW.copy(), b=_PB.copy())
    with weights:
        value, grads = weights.grad(proxy_loss)
    assert isinstance(grads, ParamDict)
    gW, gB = _softmax_ce_fd(_PW, _PB)
    assert np.allclose(grads.w, gW, atol=1e-5)  # gradient by attribute
    assert np.allclose(grads["b"], gB, atol=1e-5)


def test_step_updates_in_place_and_skips_frozen():
    weights = params(w=np.array([1.0, 2.0]), b=frozen(np.array([5.0, 6.0])))
    grads = ParamDict({"w": np.array([1.0, 1.0]), "b": None})
    weights.step(grads, lr=0.1)
    assert np.allclose(weights["w"].value, [0.9, 1.9])
    assert np.allclose(weights["b"].value, [5.0, 6.0])  # frozen untouched


def test_with_weights_inference_is_clean_and_training_updates():
    weights = params(w=_PW.copy(), b=_PB.copy())
    with weights:
        preds = proxy_forward(_PX)  # inference: arrays in
        assert type(preds) is np.ndarray and np.allclose(preds.sum(1), 1.0)
        first = last = None
        for _ in range(40):
            value, grads = weights.grad(proxy_loss)
            first = float(value) if first is None else first
            last = float(value)
            weights.step(grads, 0.5)
    assert last < first  # the same definition trained the ambient weights


def tied_obj():
    return np.sum(a * a) + np.sum(c * c)  # noqa: F821  (a, c injected; one weight)


def test_grad_with_tied_weights_shares_gradient():
    w0 = np.array([1.0, 2.0])
    weights = params(a=tied("k", w0), c=tied("k", w0))  # both tied -> one weight
    with weights:
        value, grads = weights.grad(tied_obj)
    assert np.allclose(value, 2 * np.sum(w0 * w0))
    assert np.allclose(grads["a"], 4 * w0)  # 4w: shared weight
    assert np.allclose(grads["c"], 4 * w0)
