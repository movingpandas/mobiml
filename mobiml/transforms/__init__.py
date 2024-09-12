import warnings

from .trip_extractor import TripExtractor  # noqa F401
from .delta_dataset_creator import DeltaDatasetCreator  # noqa F401
from .mover_splitter import MoverSplitter  # noqa F401

try:
    from .temporal_splitter import (  # noqa F401
        TemporalSplitter,
    )  # requires torch
except ImportError as e:
    warnings.warn(e.msg, UserWarning)

try:
    from .traj_aggregator import (  # noqa F401
        TrajectoryAggregator,
        traj_to_h3_sequence,
    )  # requires h3
except ImportError as e:
    warnings.warn(e.msg, UserWarning)

try:
    from .od_aggregator import ODAggregator  # requires h3  # noqa F401
except ImportError as e:
    warnings.warn(e.msg, UserWarning)
