Installation
============

Pyccolo is a pure-Python package with a small dependency footprint
(``traitlets`` and ``typing_extensions``). Grab it from PyPI:

.. code-block:: console

   $ pip install pyccolo

Installing the package registers two equivalent console commands — ``pyc`` and
``pyccolo`` — for running scripts and modules with instrumentation enabled (see
:doc:`cli`).

To live on the bleeding edge, install the latest revision straight from GitHub:

.. code-block:: console

   $ pip install git+https://github.com/smacke/pyccolo@master

Supported Python versions
-------------------------

Pyccolo targets **Python 3.6 through 3.14**. Because instrumentation is embedded
at the level of *source code* rather than bytecode, the same tracer code is
portable across that whole range, with a couple of feature-specific exceptions:

- **Syntax augmentation** (custom surface syntax; see
  :doc:`syntax_augmentation`) requires **Python >= 3.8**.
- **Opcode-level** ``sys`` tracing (the ``opcode`` event; see
  :doc:`sys_settrace`) requires **Python >= 3.7**.

Everything else — AST events, ``sys.settrace`` events, source-to-source
transforms — works across the full supported range.

Installing for development
--------------------------

If you are hacking on Pyccolo itself, install it in editable mode with the
``test`` extra (linters, type checker, and test runner):

.. code-block:: console

   $ pip install -e '.[test]'

or the ``dev`` extra, which additionally pulls in the build/release tooling.
Building these docs locally needs the docs requirements instead:

.. code-block:: console

   $ pip install -e '.[docs]'
   $ make -C docs html
