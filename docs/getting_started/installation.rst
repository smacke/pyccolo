Installation
============

Pyccolo is published on PyPI and installs with a single command:

.. code-block:: console

   $ pip install pyccolo

This pulls in only two runtime dependencies — `traitlets
<https://traitlets.readthedocs.io/>`_ and `typing_extensions
<https://pypi.org/project/typing-extensions/>`_ — and installs two console
scripts, ``pyc`` and ``pyccolo`` (they are aliases), for running scripts under
instrumentation from the command line. See :doc:`/reference/cli` for details.

To install the latest development version straight from GitHub:

.. code-block:: console

   $ pip install git+https://github.com/smacke/pyccolo.git

Supported Python versions
-------------------------

Because Pyccolo embeds instrumentation at the level of *source code* rather than
bytecode, the same tracer code runs across **Python 3.6 through 3.14**, with two
feature caveats:

- **Syntax augmentation** (defining new surface syntax) requires Python **3.8+**.
- **Opcode tracing** (the ``opcode`` ``sys.settrace`` event) requires Python
  **3.7+**.

Everything else — AST events, handlers, guards, composition, source-to-source —
works on the full range.

Working from a source checkout
------------------------------

If you are hacking on Pyccolo itself, clone the repository and install it in
editable mode with the relevant extras:

.. code-block:: console

   $ git clone https://github.com/smacke/pyccolo.git
   $ cd pyccolo
   $ pip install -e '.[test]'   # test/lint/type-check toolchain
   $ pip install -e '.[docs]'   # Sphinx toolchain for building these docs

You can then build this documentation locally:

.. code-block:: console

   $ make -C docs html

The rendered site lands in ``docs/_build/html``.

Next steps
----------

With Pyccolo installed, head to :doc:`/getting_started/first_tracer` to build
something that runs.
