from .release import __version__

# the package is imported during installation (to obtain the version)
# however installation happens in an isolated build environment
# where no dependencies are installed.

try:
    from . import core
    from . import blocks
    from .core import *
    from .blocks import *
except ImportError:
    import os
    if "PIP_BUILD_TRACKER" in os.environ:
        pass
    else:
        # raise the original exception
        raise
