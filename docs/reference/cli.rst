Command-line interface
======================

You can execute arbitrary scripts with instrumentation enabled using the ``pyc``
command-line tool (installed as both ``pyc`` and ``pyccolo``). For example, to run
some script ``bar.py`` under the optional-chaining example tracer:

.. code-block:: text

   # bar.py
   bar = None
   # prints `None` since bar?.foo coalesces to `None`
   print(bar?.foo)

.. code-block:: console

   $ pyc bar.py -t pyccolo.examples.optional_chaining.ScriptOptionalChainer

You can also run ``bar`` as a module (indeed, ``pyc`` does this internally when
given a file):

.. code-block:: console

   $ pyc -m bar -t pyccolo.examples.optional_chaining.ScriptOptionalChainer

The ``-t`` value is a fully-qualified reference to a tracer class. Note that we
use ``ScriptOptionalChainer`` rather than the bare ``OptionalChainer``: because
``pyc`` runs your file through Pyccolo's import machinery, the tracer must opt that
file in by overriding :meth:`~pyccolo.BaseTracer.should_instrument_file` (which
``ScriptOptionalChainer`` does, and ``OptionalChainer`` does not — see
:doc:`/guides/tracing_real_programs`).

You can specify multiple tracer classes after ``-t``; in case you were not already
aware, Pyccolo is :doc:`composable </concepts/composition>`!

Options
-------

.. argparse::
   :module: pyccolo.__main__
   :func: make_parser
   :prog: pyc
