# -*- coding: future_annotations -*-
import ast
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from pyccolo.tracing.syntax_augmentation import AugmentationSpec
from pyccolo.tracing.expr_rewriter import ExprRewriter
from pyccolo.tracing.stmt_inserter import StatementInserter
from pyccolo.tracing.stmt_mapper import StatementMapper

if TYPE_CHECKING:
    from typing import Dict, Optional, Set, Tuple
    from pyccolo.tracing.tracer import SingletonTracerStateMachine


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class AstRewriter(ast.NodeTransformer):
    def __init__(self, tracer: SingletonTracerStateMachine, module_id: Optional[int] = None):
        self._tracer = tracer
        self._module_id: Optional[int] = module_id
        self._augmented_positions_by_spec: Dict[AugmentationSpec, Set[Tuple[int, int]]] = defaultdict(set)
        self.orig_to_copy_mapping: Optional[Dict[int, ast.AST]] = None

    def register_augmented_position(self, aug_spec: AugmentationSpec, lineno: int, col_offset: int) -> None:
        self._augmented_positions_by_spec[aug_spec].add((lineno, col_offset))

    def visit(self, node: ast.AST):
        assert isinstance(node, ast.Module)
        mapper = StatementMapper(
            self._tracer.line_to_stmt_by_module_id[id(node) if self._module_id is None else self._module_id],
            self._tracer,
            self._augmented_positions_by_spec,
        )
        orig_to_copy_mapping = mapper(node)
        self.orig_to_copy_mapping = orig_to_copy_mapping
        # very important that the eavesdropper does not create new ast nodes for ast.stmt (but just
        # modifies existing ones), since StatementInserter relies on being able to map these
        events_with_handlers = self._tracer.events_with_registered_handlers
        node = ExprRewriter(orig_to_copy_mapping, events_with_handlers, self._tracer.guards).visit(node)
        node = StatementInserter(orig_to_copy_mapping, events_with_handlers, self._tracer.guards).visit(node)
        return node
