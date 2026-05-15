__version__ = '0.1.0'
__author__ = 'Manuel Blanco Valentin'
__url__ = 'https://manuelblancovalentin.github.io/ENABOL/'

from . import utils
from . import dtypes
from .dataset import AffineDataset
from .nn import LinearBlockModel
from .precision import PrecisionDict


# Print a welcome message when the package is imported
print(f'[INFO] - ENABOL imported successfully! Version: {__version__}, URL: {__url__}')
