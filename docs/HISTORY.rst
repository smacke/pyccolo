History
=======

0.0.20 (2022-02-14)
------------------
* Provide non-context manager variants of tracing-related contexts;

0.0.19 (2022-02-14)
------------------
* Add 'exit_module' event;
* Use deferred evaluation variants for all 'before expr' events;
* Improve AST bookkeeping;
* Add FutureTracer under pyccolo.examples;
* Fix bug where starred expressions weren't traced if used as literal elements;

0.0.17 (2022-02-03)
------------------
* Fix packaging issue after new configuration;

0.0.14 (2022-02-02)
------------------
* Move configuration out of setup.py;

0.0.13 (2022-01-31)
------------------
* Default to all tracers in stack for package-level tracing enabled / disabled context managers;
* Omit instrumentating the AST of statements underneath "with pyc.tracing_disabled()" bocks;
* Add SkipAll return value;
* Improve reentrancy for sys events;

0.0.12 (2022-01-30)
------------------
* Expose logic for resolving tracer class based on module path;

0.0.11 (2022-01-30)
------------------
* Expanded predicate functionality;
* New events for after if / while test, after expr stmts, after lambda body, before / after augassign rhs;
* Disambiguate between user and generated lambdas (e.g. used for before expr events);

0.0.10 (2022-01-26)
------------------
* Simplify binop events;
* Add compare events;

0.0.9 (2022-01-24)
------------------
* Allow per-handler reentrancy;

0.0.8 (2022-01-23)
------------------
* Add eval helper;
* Add syntactic macro examples (quasiquotes and quick lambdas);
* Add support for conditional handlers;

0.0.7 (2022-01-06)
------------------
* Add cli;
* Add basic readme documentation;
* Allow returning pyc.Skip for skipping subsequent handlers for same event;
* Misc improvements to file filter hooks;
* Allow returning lambdas for before_expr events;

0.0.6 (2022-01-06)
------------------
* Misc ergonomics improvements;
* Enable for Python 3.10;
* Enable linting and fix package-level imports;

0.0.5 (2021-12-29)
------------------
* Get rid of future-annotations dependency;
* Fix memory leak in sandbox exec;

0.0.4 (2021-12-26)
------------------
* Misc composability improvements and fixes;
* Improve file filter handling;

0.0.3 (2021-12-23)
------------------
* Misc ergonomics improvements;
* Misc composability improvements and fixes;

0.0.2 (2021-12-22)
------------------
* Initial internal release;

0.0.1 (2020-10-25)
------------------
* Initial placeholder release;
