"""Optimisation passes."""

from .algebra import AlgebraicSimplification
from .base import ScopedTransformer
from .comprehension import ComprehensionToMap
from .condnarrow import ConditionNarrowing
from .cse import CommonSubexpressionElimination
from .deadcode import DeadCodeElimination
from .dedynamize import DeDynamize
from .folding import ConstantFolding
from .inline import FunctionInlining
from .licm import LoopInvariantMotion
from .localize import GlobalLocalization
from .loopfold import LoopFolding
from .looptocomp import LoopToComprehension
from .outline import ModuleLoopOutlining
from .propagate import ConstantPropagation
from .rangeiter import RangeToIteration
from .tailcall import TailRecursion
from .unused import UnusedElimination

__all__ = [
    "ScopedTransformer",
    "DeDynamize",
    "ConstantFolding",
    "ConstantPropagation",
    "AlgebraicSimplification",
    "LoopFolding",
    "ConditionNarrowing",
    "DeadCodeElimination",
    "LoopInvariantMotion",
    "CommonSubexpressionElimination",
    "UnusedElimination",
    "FunctionInlining",
    "TailRecursion",
    "RangeToIteration",
    "LoopToComprehension",
    "ComprehensionToMap",
    "ModuleLoopOutlining",
    "GlobalLocalization",
]
