"""Windows and watermarks: how a stream with no end is cut into answers that can be reported.

**Windows.** Three assigners cover almost everything:

* *Tumbling* -- fixed, non-overlapping. Each event belongs to exactly one.
* *Sliding* -- fixed size, advancing by a smaller step, so each event belongs to ``size/step`` windows at
  once. That multiplicity is the cost, and ``panes_for`` is how it is avoided.
* *Session* -- data-driven: a window per burst of activity, closed by an idle gap. Sessions have no fixed
  boundaries, so a late event can *merge two existing sessions into one*, which is the hardest case in
  windowing and is implemented rather than skipped.

**Watermarks.** The pipeline must decide when a window is complete, and completeness is unknowable in a
stream. A watermark is a claim: "no event with event time below W will arrive from now on". Everything about
this claim is a trade:

    W = max event time seen - allowed_lag

A large lag means late results and few dropped events. A small lag means fast results and more drops. There is
no setting that gives both, and any documentation that suggests otherwise is selling something.
``BoundedOutOfOrderness`` implements this; ``PercentileWatermark`` estimates the lag from the observed delay
distribution instead of taking it as a constant, which adapts to a lagging consumer and is *still* only a
quantile of a distribution with no maximum.

The watermark is **monotonic by construction**. It never moves backwards, even when a very late event arrives,
because a window that has already fired cannot be un-fired -- downstream has seen the result. That is why late
events need their own path (`allowed lateness` and a side output) rather than simply being folded back in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import Event


@dataclass(frozen=True, order=True)
class Window:
    """A half-open interval ``[start, end)`` in event time.

    Half-open is not a detail: with closed intervals an event at a boundary lands in two adjacent tumbling
    windows and every count is inflated at exactly the timestamps most likely to be tested by hand.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"window [{self.start}, {self.end}) is empty or inverted")

    def contains(self, time: int) -> bool:
        return self.start <= time < self.end

    @property
    def size(self) -> int:
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.start},{self.end})"


class TumblingWindows:
    """Fixed, non-overlapping windows of ``size``, aligned to zero."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("window size must be positive")
        self.size = size

    def assign(self, time: int) -> "list[Window]":
        start = (time // self.size) * self.size
        return [Window(start, start + self.size)]

    def __repr__(self) -> str:
        return f"TumblingWindows(size={self.size})"


class SlidingWindows:
    """Windows of ``size`` every ``step``, so each event lands in ``ceil(size/step)`` of them."""

    def __init__(self, size: int, step: int) -> None:
        if size <= 0 or step <= 0:
            raise ValueError("size and step must be positive")
        if step > size:
            raise ValueError(
                "step above size would leave gaps between windows, which is a hopping window and "
                "means some events belong to no window at all -- say so explicitly if that is wanted"
            )
        self.size = size
        self.step = step

    def assign(self, time: int) -> "list[Window]":
        """Every window containing ``time``.

        The first window that can contain ``time`` starts at the largest multiple of ``step`` at or below it;
        from there, walk backwards while the window still covers ``time``.
        """
        last_start = (time // self.step) * self.step
        windows = []
        start = last_start
        while start > time - self.size:
            if start >= 0:
                windows.append(Window(start, start + self.size))
            start -= self.step
        return sorted(windows)

    @property
    def multiplicity(self) -> int:
        return -(-self.size // self.step)

    def __repr__(self) -> str:
        return f"SlidingWindows(size={self.size}, step={self.step})"


def panes_for(size: int, step: int) -> int:
    """Pane width for a sliding window: ``gcd(size, step)``.

    A sliding window of size 60 stepping every 10 makes every event part of six windows, so a naive
    implementation stores and aggregates it six times. Instead, aggregate once into non-overlapping **panes**
    of width ``gcd(size, step)``, then compose each window from its panes -- for combinable aggregates (count,
    sum, min, max) the work per event drops to one update regardless of how many windows overlap.

    The saving is exactly ``size/step``, and it is not available for non-combinable aggregates: an exact
    median cannot be composed from pane medians, which is why percentile aggregations over long sliding
    windows are expensive in every stream processor and why sketches exist.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    from math import gcd

    return gcd(size, step)


def panes_in_window(window: Window, pane: int) -> "list[Window]":
    """The panes composing a window. ``window.size`` must be a whole number of panes."""
    if window.size % pane != 0:
        raise ValueError(f"window {window} is not a whole number of panes of {pane}")
    return [Window(start, start + pane) for start in range(window.start, window.end, pane)]


# ---------------------------------------------------------------------------------------------
# session windows
# ---------------------------------------------------------------------------------------------


@dataclass
class SessionWindows:
    """Data-driven windows separated by an idle ``gap``.

    Assignment is per key and stateful: each event provisionally creates the window
    ``[event_time, event_time + gap)``, and overlapping windows for the same key are merged. Merging is the
    part that matters -- an event arriving between two existing sessions joins them, so a session's identity
    is not stable until its key has been idle for ``gap``. Any downstream consumer that keyed on a session id
    before then was keying on a guess.
    """

    gap: int
    sessions: "dict[str, list[Window]]" = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gap <= 0:
            raise ValueError("session gap must be positive")

    def add(self, event: Event) -> "tuple[Window, list[Window]]":
        """Add an event; return its (possibly merged) window and any windows the merge replaced."""
        proposed = Window(event.event_time, event.event_time + self.gap)
        existing = self.sessions.setdefault(event.key, [])
        overlapping = [
            window
            for window in existing
            if window.start <= proposed.end and proposed.start <= window.end
        ]
        if overlapping:
            merged = Window(
                min(proposed.start, min(window.start for window in overlapping)),
                max(proposed.end, max(window.end for window in overlapping)),
            )
        else:
            merged = proposed
        for window in overlapping:
            existing.remove(window)
        existing.append(merged)
        existing.sort()
        return merged, overlapping

    def closed(self, watermark: int) -> "list[tuple[str, Window]]":
        """Sessions whose idle gap has fully elapsed under the current watermark."""
        output = []
        for key, windows in self.sessions.items():
            for window in windows:
                if window.end <= watermark:
                    output.append((key, window))
        return output

    def discard(self, key: str, window: Window) -> None:
        if window in self.sessions.get(key, []):
            self.sessions[key].remove(window)


# ---------------------------------------------------------------------------------------------
# watermarks
# ---------------------------------------------------------------------------------------------


class BoundedOutOfOrderness:
    """``W = max event time seen - lag``, monotonic.

    The standard watermark, and the standard trap: ``lag`` is a bet on the delay distribution's tail. Set it
    to the observed maximum delay in a test dataset and production will exceed it the first time a mobile
    client reconnects after a flight.
    """

    def __init__(self, lag: int) -> None:
        if lag < 0:
            raise ValueError("lag cannot be negative")
        self.lag = lag
        self._max_seen = -1
        self._watermark = -1

    def observe(self, event: Event) -> int:
        self._max_seen = max(self._max_seen, event.event_time)
        # max(): a watermark must never regress, whatever arrives.
        self._watermark = max(self._watermark, self._max_seen - self.lag)
        return self._watermark

    @property
    def current(self) -> int:
        return self._watermark

    def state(self) -> dict:
        return {"kind": "bounded", "lag": self.lag, "max_seen": self._max_seen, "watermark": self._watermark}

    def restore(self, state: dict) -> None:
        self.lag = state["lag"]
        self._max_seen = state["max_seen"]
        self._watermark = state["watermark"]

    def __repr__(self) -> str:
        return f"BoundedOutOfOrderness(lag={self.lag})"


class PercentileWatermark:
    """Estimate the lag from the observed delay distribution instead of fixing it in advance.

    Keeps a reservoir of recent delays and uses a high quantile as the lag, so a lagging consumer widens the
    window automatically and a healthy stream reports promptly. It adapts, and it is still a quantile: by
    construction it expects to drop about ``1 - percentile`` of events, and that is the honest reading of the
    parameter -- not "safe", just "wrong this often".
    """

    def __init__(self, percentile: float = 0.99, window: int = 2000, minimum_lag: int = 1) -> None:
        if not 0.5 <= percentile < 1.0:
            raise ValueError("percentile must lie in [0.5, 1)")
        self.percentile = percentile
        self.window = window
        self.minimum_lag = minimum_lag
        self._delays: list[int] = []
        self._max_seen = -1
        self._watermark = -1

    def observe(self, event: Event) -> int:
        self._delays.append(event.delay)
        if len(self._delays) > self.window:
            self._delays.pop(0)
        self._max_seen = max(self._max_seen, event.event_time)
        ordered = sorted(self._delays)
        index = min(int(self.percentile * len(ordered)), len(ordered) - 1)
        lag = max(ordered[index], self.minimum_lag)
        self._watermark = max(self._watermark, self._max_seen - lag)
        return self._watermark

    @property
    def current(self) -> int:
        return self._watermark

    @property
    def estimated_lag(self) -> int:
        if not self._delays:
            return self.minimum_lag
        ordered = sorted(self._delays)
        index = min(int(self.percentile * len(ordered)), len(ordered) - 1)
        return max(ordered[index], self.minimum_lag)

    def state(self) -> dict:
        return {
            "kind": "percentile",
            "percentile": self.percentile,
            "window": self.window,
            "minimum_lag": self.minimum_lag,
            "delays": list(self._delays),
            "max_seen": self._max_seen,
            "watermark": self._watermark,
        }

    def restore(self, state: dict) -> None:
        self.percentile = state["percentile"]
        self.window = state["window"]
        self.minimum_lag = state["minimum_lag"]
        self._delays = list(state["delays"])
        self._max_seen = state["max_seen"]
        self._watermark = state["watermark"]


def restore_watermark(state: dict):
    """Rebuild a watermark from a checkpoint, dispatching on its recorded kind."""
    if state["kind"] == "bounded":
        watermark = BoundedOutOfOrderness(state["lag"])
    elif state["kind"] == "percentile":
        watermark = PercentileWatermark(state["percentile"], state["window"], state["minimum_lag"])
    else:
        raise ValueError(f"unknown watermark kind {state['kind']!r}")
    watermark.restore(state)
    return watermark
