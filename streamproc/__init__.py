"""Event-time stream processing from scratch: watermarks, windows, panes, lateness, checkpoints.

The one-line version: aggregating by arrival time gives an answer that depends on your infrastructure's
mood; aggregating by event time gives an answer about the world, and the whole of this package is the
machinery for deciding when such an answer may be published.
"""

from .aggregate import (
    AGGREGATORS,
    Count,
    DistinctExact,
    ExactTopK,
    Mean,
    MinMax,
    SpaceSaving,
    Sum,
    merge_all,
    topk_agreement,
)
from .events import Event, StreamSpec, generate, sessions_stream, skewed_stream
from .pipeline import Result, Stats, WindowedProcessor, totals_by_window, tumbling_count
from .windows import (
    BoundedOutOfOrderness,
    PercentileWatermark,
    SessionWindows,
    SlidingWindows,
    TumblingWindows,
    Window,
    panes_for,
    panes_in_window,
)

__all__ = [
    "AGGREGATORS",
    "BoundedOutOfOrderness",
    "Count",
    "DistinctExact",
    "Event",
    "ExactTopK",
    "Mean",
    "MinMax",
    "PercentileWatermark",
    "Result",
    "SessionWindows",
    "SlidingWindows",
    "SpaceSaving",
    "Stats",
    "StreamSpec",
    "Sum",
    "TumblingWindows",
    "Window",
    "WindowedProcessor",
    "generate",
    "merge_all",
    "panes_for",
    "panes_in_window",
    "sessions_stream",
    "skewed_stream",
    "topk_agreement",
    "totals_by_window",
    "tumbling_count",
]
__version__ = "1.0.0"
