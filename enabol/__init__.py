__version__ = '0.1.0'
__author__ = 'Manuel Blanco Valentin'
__url__ = 'https://manuelblancovalentin.github.io/ENABOL/'

from . import utils
from . import dtypes
from .dataset import AffineDataset
from .nn import (
    BaseModel,
    LinearBlockModel,
    Controller,
    BaseController,
    NoController,
    GlobalThrottleOrder0Controller,
    GlobalThrottleOrder1Controller,
    GlobalThrottleOrder2Controller,
    QuantizationAwareOrder2Controller,
    BaseUpdateRule,
    SGDUpdateRule,
    CurvatureSensor,
    HistoryRecorder,
    MetricsConfig,
    InstrumentedTrainer,
    InstrumentationConfig,
    UpdateApplier,
)
from .precision import PrecisionDict
from .compile import compile
from .history import FitHistory


# Print a welcome message when the package is imported
print(f'[INFO] - ENABOL imported successfully! Version: {__version__}, URL: {__url__}')
