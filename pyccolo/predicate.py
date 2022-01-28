# -*- coding: utf-8 -*-
import ast
from typing import overload, Callable, Optional, Sequence, Union
from typing_extensions import Literal


class Predicate:
    TRUE: "Predicate" = None
    FALSE: "Predicate" = None

    @overload
    def __init__(
        self,
        condition: Callable[[ast.AST], bool],
        use_raw_node_id: Literal[False],
        static: bool = True,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        condition: Callable[[int], bool],
        use_raw_node_id: Literal[True],
        static: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        condition: Callable[..., bool],
        use_raw_node_id: bool = False,
        static: bool = False,
    ) -> None:
        ...

    def __init__(
        self,
        condition: Callable[..., bool],
        use_raw_node_id: bool = False,
        static: bool = False,
    ) -> None:
        self.condition = condition
        self.use_raw_node_id = use_raw_node_id
        self.static = static

    def __call__(self, node: Union[ast.AST, int]) -> bool:
        node_or_id = (
            id(node) if self.use_raw_node_id and isinstance(node, ast.AST) else node
        )
        return self.condition(node_or_id)  # type: ignore

    def dynamic_call(self, node: Union[ast.AST, int]) -> bool:
        return True if self.static else self(node)


Predicate.TRUE = Predicate(lambda *_: True)
Predicate.FALSE = Predicate(lambda *_: False)


class CompositePredicate(Predicate):
    def __init__(self, base_predicates: Sequence[Predicate], reducer=any) -> None:
        self.base_predicates = list(base_predicates)
        self.dynamic_base_predicates = [
            pred for pred in base_predicates if not pred.static
        ]
        self.static = len(self.dynamic_base_predicates) == 0
        self.use_raw_node_id = all(pred.use_raw_node_id for pred in base_predicates)
        self.reducer = reducer

    def __call__(
        self,
        node: Union[ast.AST, int],
        predicates: Optional[Sequence[Predicate]] = None,
    ) -> bool:
        predicates = self.base_predicates if predicates is None else predicates
        assert len(predicates) > 0
        return self.reducer(pred(node) for pred in predicates)

    def dynamic_call(self, node: Union[ast.AST, int]) -> bool:
        return (
            True if self.static else self(node, predicates=self.dynamic_base_predicates)
        )

    @classmethod
    def _create(cls, base_predicates: Sequence[Predicate], reducer) -> Predicate:
        assert len(base_predicates) > 0
        return cls(base_predicates, reducer=reducer)

    @classmethod
    def any(cls, base_predicates: Sequence[Predicate]) -> Predicate:
        if len(base_predicates) == 0 or any(
            pred is Predicate.TRUE for pred in base_predicates
        ):
            return Predicate.TRUE
        if all(pred is Predicate.FALSE for pred in base_predicates):
            return Predicate.FALSE
        return cls._create(base_predicates, reducer=any)

    @classmethod
    def all(cls, base_predicates: Sequence[Predicate]) -> Predicate:
        if any(pred is Predicate.FALSE for pred in base_predicates):
            return Predicate.FALSE
        if all(pred is Predicate.TRUE for pred in base_predicates):
            return Predicate.TRUE
        return cls._create(base_predicates, reducer=all)
