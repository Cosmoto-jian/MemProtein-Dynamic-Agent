"""MemProtein: membrane-protein force-deformation simulation & analysis.

Typical use:
    from memprotein.preprocess import build_inputs
    from memprotein.simulate import run_simulation
    from memprotein import analysis
    from memprotein.pipeline import run

See README for the command-line interface (cli.py).
"""

from .preprocess import build_inputs  # noqa: F401
