Guards and performance
======================

Instrumentation is not free: every emitted event is a function call into your
handler. Often you only need that call *once* — the first time a loop body runs,
the first time a function executes, the first time a particular name is loaded —
and after that the instrumentation is pure overhead. **Guards** let a handler
switch its own instrumentation off for a region, so subsequent runs execute
native, uninstrumented code. This guide covers the two flavors of guard, plus the
:class:`~pyccolo.TraceStack` helper for handlers that need per-call state, and the
class-level switches that trade safety for speed.

Region guards: run once, then step aside
-----------------------------------------

Several events are **guarded**: they hand your handler a ``guard`` keyword
argument naming the region they belong to (a loop body, a function). Call
:meth:`~pyccolo.BaseTracer.activate_guard` with it and Pyccolo stops emitting that
region's events; later iterations or calls run uninstrumented. Here a handler
fires on the first loop iteration and then guards itself off:

.. testcode::

   import pyccolo as pyc

   iterations_traced = []


   class TraceFirstIteration(pyc.BaseTracer):
       @pyc.register_raw_handler(pyc.after_for_loop_iter)
       def once(self, *_, guard, **__):
           iterations_traced.append(1)
           self.activate_guard(guard)


   with TraceFirstIteration:
       pyc.exec("for i in range(1000):\n    pass")
   assert iterations_traced == [1]     # only the first iteration was traced

The same pattern applies to ``after_while_loop_iter`` and
``after_function_execution``. To turn a region's instrumentation back on, call
:meth:`~pyccolo.BaseTracer.deactivate_guard` with the same guard. (We use
``register_raw_handler`` here because we only need the ``guard`` keyword, not the
resolved node — see :doc:`/reference/registration`.)

Local guards: run once *per node*
---------------------------------

Region guards are coarse — they gate a whole loop or function. A **local guard**
is finer: you supply a ``guard=`` function that maps an AST node to a stable guard
*name*, and Pyccolo gives you a boolean flag per name that you flip once you've
seen what you need. This is how you say "handle the first load of each variable,
then never again." Because the flag lives in the module globals, you initialize
each one to ``False`` in an ``init_module`` handler:

.. testcode::

   from collections import Counter


   class FirstLoadOfEachName(pyc.BaseTracer):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.loads = Counter()

       @pyc.init_module
       def init(self, _ret, node, frame, *_, **__):
           for guard in self.local_guards_by_module_id.get(id(node), []):
               frame.f_globals[guard] = False

       @pyc.load_name(guard=lambda node: f"{pyc.PYCCOLO_BUILTIN_PREFIX}_{node.id}")
       def on_load(self, _ret, node, frame, _evt, guard, *_, **__):
           self.loads[node.id] += 1
           frame.f_globals[guard] = True     # activate this name's guard


   with FirstLoadOfEachName.instance():
       pyc.exec("w = 0\nx = w + 1\ny = w + x + 1\nz = w + x + y + 1")
   assert FirstLoadOfEachName.instance().loads == Counter(w=1, x=1, y=1)

``w``, ``x``, and ``y`` are each loaded several times, but the guard keyed on the
name means the handler fires only on the *first* load of each. The
``local_guards_by_module_id`` table is populated during the rewrite, which is why
``init_module`` — fired once as the module starts — is the right place to
zero-initialize the flags. The bundled ``lazy_imports.py`` example uses exactly
this technique to resolve each imported name once, on first use.

Per-call state with ``TraceStack``
-----------------------------------

Guards shed overhead; :class:`~pyccolo.TraceStack` manages *state* that must
follow the call stack. :meth:`~pyccolo.BaseTracer.make_stack` returns a stack, and
attributes assigned inside its
:meth:`~pyccolo.TraceStack.register_stack_state` block become per-frame: pushing
saves them, popping restores them. Pairing a push in ``before_call`` with a pop in
``after_call`` gives you the live call depth as ``len(stack)``:

.. testcode::

   class CallDepth(pyc.BaseTracer):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.stack = self.make_stack()
           with self.stack.register_stack_state():
               self.name = ""
           self.max_depth = 0

       @pyc.register_handler(pyc.before_call)
       def enter(self, fun, *_, **__):
           with self.stack.push():
               self.name = getattr(fun, "__name__", "?")
           self.max_depth = max(self.max_depth, len(self.stack))
           return fun

       @pyc.register_handler(pyc.after_call)
       def leave(self, ret, *_, **__):
           self.stack.pop()
           return ret


   with CallDepth:
       env = pyc.exec(
           "def fact(n):\n"
           "    return 1 if n <= 1 else n * fact(n - 1)\n"
           "result = fact(4)"
       )
   assert env["result"] == 24
   assert CallDepth.instance().max_depth == 4

Each stack frame carries its own ``name``, and the deepest the stack ever got
during ``fact(4)`` was four frames. See :doc:`/reference/tracer` for the full
``TraceStack`` surface (``get_field``, ``needing_manual_initialization``, and
friends).

Class-level switches
--------------------

A few class attributes trade a safety guarantee for speed; set them on your
tracer subclass.

- ``bytecode_caching_allowed`` — Pyccolo can cache the instrumented bytecode it
  compiles, a welcome speedup for most tracers. A tool that must observe *every*
  run from scratch (a coverage counter, say) should set this to ``False`` so a
  stale cache never hides a statement.
- ``global_guards_enabled`` — set ``False`` on a tracer that only rewrites source
  and emits no runtime events, so it adds no per-node guard machinery (see
  :doc:`/guides/compose_tracers`).
- ``instrument_lambdas`` — whether lambda bodies are instrumented; leaving it off
  avoids overhead when you don't care about them.

The complete list, with defaults, is in :doc:`/reference/config`.
