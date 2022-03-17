# -*- coding: utf-8 -*-
import ast
from typing import Any, Callable, NamedTuple, Optional

from pyccolo.predicate import Predicate


class HandlerSpec(NamedTuple):
    handler: Callable[..., Any]
    use_raw_node_id: bool
    reentrant: bool
    predicate: Predicate
    guard: Optional[Callable[[ast.AST], str]]

    @classmethod
    def empty(cls):
        return cls(None, False, False, Predicate(lambda *_: True), None)  # type: ignore
