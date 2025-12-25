"""
ConstrainedDiffusionLM: Constraint-Preserving Diffusion Language Modeling

A diffusion-based language model that enables global text generation and editing
under hard token-level constraints.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from constrained_diffusion_lm.utils.seed import set_seed

__all__ = [
    "__version__",
    "set_seed",
]

