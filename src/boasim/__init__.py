
# this try-except clause is necessary during installation
try:
    from .core import *
    from .blocks import *

except ImportError:
    pass

from .release import __version__
