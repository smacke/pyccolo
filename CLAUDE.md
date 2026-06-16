# pyccolo — coding conventions

## Typing

- **Avoid `Any`.** Prefer precise types: concrete classes, `Union`s, recursive
  type aliases, `TypeVar`s, or `object` (the type-safe top type — it forces
  callers to narrow, whereas `Any` silently disables checking). Narrow with
  `isinstance` or `typing.cast` rather than reaching for `Any`.
- Reserve `Any` for genuine interop escape hatches where a precise type fights
  the third-party stubs — e.g. NumPy index types (`x[key]`), the variadic shape
  args to `np.reshape`, and ignored `**kwargs` / passthrough numpy params. Keep
  those uses localized and obvious.
- New / changed code must pass `mypy` (`make typecheck`) and the formatters
  (`make black` / isort).
