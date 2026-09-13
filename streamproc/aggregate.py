"""Aggregators, and the property that decides whether a window is cheap or expensive.

An aggregator is **combinable** when partial results can be merged:

    merge(agg(A), agg(B)) == agg(A + B)

Count, sum, min and max are. Mean is, if you carry the count alongside the sum rather than the mean itself.
An exact median is not, and no amount of engineering makes it so -- which is why pane decomposition works for
the first group and why percentile aggregations over long sliding windows need a sketch instead.

Combinability is what makes three separate things possible, all of them load-bearing:

1. **Pane decomposition.** Aggregate each event once into a small pane, then compose windows from panes. A
   sliding window of 60 stepping by 10 costs one update per event instead of six.
2. **Session merging.** When a late event joins two sessions, their accumulated states merge without
   revisiting the original events -- which is the only reason session windows are affordable at all.
3. **Checkpointing.** State that merges is state that can be serialised, split and reassigned across workers.

``ExactTopK`` and ``SpaceSaving`` sit at the end. Exact top-k over a stream needs a counter per distinct key,
which on a Zipf-distributed key space means state proportional to the tail -- mostly keys seen once. Metwally,
Agrawal and El Abbadi's Space-Saving keeps ``capacity`` counters in bounded memory and comes with a real
guarantee, stated here rather than hand-waved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from .events import Event


class Aggregator(Protocol):
    """The interface a windowed aggregate must satisfy to be usable with panes and sessions."""

    def add(self, event: Event) -> None: ...
    def merge(self, other: "Aggregator") -> "Aggregator": ...
    def result(self) -> float: ...
    def state(self) -> dict: ...


@dataclass
class Count:
    count: int = 0

    def add(self, event: Event) -> None:
        self.count += 1

    def merge(self, other: "Count") -> "Count":
        return Count(self.count + other.count)

    def result(self) -> float:
        return float(self.count)

    def state(self) -> dict:
        return {"kind": "count", "count": self.count}


@dataclass
class Sum:
    total: float = 0.0

    def add(self, event: Event) -> None:
        self.total += event.value

    def merge(self, other: "Sum") -> "Sum":
        return Sum(self.total + other.total)

    def result(self) -> float:
        return self.total

    def state(self) -> dict:
        return {"kind": "sum", "total": self.total}


@dataclass
class Mean:
    """Carries sum and count, never the running mean.

    Merging two means requires their counts, so a state holding only the mean is not combinable. Storing the
    pair also avoids the numerically worse incremental-mean update; for the magnitudes seen here either is
    fine, but the pair is what makes the aggregate mergeable, and that is the reason.
    """

    total: float = 0.0
    count: int = 0

    def add(self, event: Event) -> None:
        self.total += event.value
        self.count += 1

    def merge(self, other: "Mean") -> "Mean":
        return Mean(self.total + other.total, self.count + other.count)

    def result(self) -> float:
        return self.total / self.count if self.count else 0.0

    def state(self) -> dict:
        return {"kind": "mean", "total": self.total, "count": self.count}


@dataclass
class MinMax:
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, event: Event) -> None:
        self.minimum = min(self.minimum, event.value)
        self.maximum = max(self.maximum, event.value)

    def merge(self, other: "MinMax") -> "MinMax":
        return MinMax(min(self.minimum, other.minimum), max(self.maximum, other.maximum))

    def result(self) -> float:
        """The range. Empty state gives 0.0 rather than inf - -inf, which is nan."""
        if self.minimum > self.maximum:
            return 0.0
        return self.maximum - self.minimum

    def state(self) -> dict:
        return {"kind": "minmax", "minimum": self.minimum, "maximum": self.maximum}


@dataclass
class DistinctExact:
    """Exact distinct count by keeping every key seen.

    Combinable (set union), and unbounded: state grows with cardinality, without limit. Included as the
    reference an approximate sketch would be measured against, and as an honest statement of the cost --
    HyperLogLog exists because this class is what the alternative looks like.
    """

    keys: set = field(default_factory=set)

    def add(self, event: Event) -> None:
        self.keys.add(event.key)

    def merge(self, other: "DistinctExact") -> "DistinctExact":
        return DistinctExact(self.keys | other.keys)

    def result(self) -> float:
        return float(len(self.keys))

    def state(self) -> dict:
        return {"kind": "distinct", "keys": sorted(self.keys)}


AGGREGATORS = {
    "count": Count,
    "sum": Sum,
    "mean": Mean,
    "minmax": MinMax,
    "distinct": DistinctExact,
}


def restore_aggregator(state: dict) -> Aggregator:
    kind = state["kind"]
    if kind == "count":
        return Count(state["count"])
    if kind == "sum":
        return Sum(state["total"])
    if kind == "mean":
        return Mean(state["total"], state["count"])
    if kind == "minmax":
        return MinMax(state["minimum"], state["maximum"])
    if kind == "distinct":
        return DistinctExact(set(state["keys"]))
    raise ValueError(f"unknown aggregator kind {kind!r}")


def merge_all(states: "list[Aggregator]") -> Aggregator:
    """Fold a list of partial aggregates into one. Panes and session merges both come through here."""
    if not states:
        raise ValueError("nothing to merge")
    merged = states[0]
    for state in states[1:]:
        merged = merged.merge(state)
    return merged


# ---------------------------------------------------------------------------------------------
# heavy hitters
# ---------------------------------------------------------------------------------------------


@dataclass
class ExactTopK:
    """A counter per distinct key. Correct, and unbounded -- the reference to measure error against."""

    counts: "dict[str, float]" = field(default_factory=dict)

    def add(self, event: Event) -> None:
        self.counts[event.key] = self.counts.get(event.key, 0.0) + 1.0

    def top(self, k: int) -> "list[tuple[str, float]]":
        return sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))[:k]

    @property
    def distinct_keys(self) -> int:
        return len(self.counts)


class SpaceSaving:
    """Metwally, Agrawal & El Abbadi (2005): top-k in ``capacity`` counters, with a provable error bound.

    The algorithm, in three cases:

    * key already monitored -> increment it;
    * spare capacity -> start monitoring it at 1;
    * otherwise -> **take over the smallest counter**: evict its key, adopt the new one, keep the count and
      add one, and record the evicted count as this entry's ``error``.

    The takeover step is the whole idea. A key that arrives once inherits the minimum count, which is an
    over-estimate -- so every count is an upper bound, and the true count lies in
    ``[count - error, count]``. Never an under-estimate, which is the direction that matters: a genuinely
    frequent key cannot be missed, while an infrequent one may be reported with an inflated count.

    Guarantee: with ``capacity = 1/eps`` counters, every key's over-estimate is at most ``eps * N`` after N
    events, and every key with true frequency above ``eps * N`` is guaranteed to be monitored. So a top-k
    report is trustworthy exactly when the k-th count comfortably exceeds ``min_count`` -- and
    ``guaranteed_top`` filters on that instead of leaving the caller to assume it.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.counts: dict[str, float] = {}
        self.errors: dict[str, float] = {}
        self.total = 0.0

    def add(self, event: Event) -> None:
        self.observe(event.key)

    def observe(self, key: str) -> None:
        self.total += 1.0
        if key in self.counts:
            self.counts[key] += 1.0
            return
        if len(self.counts) < self.capacity:
            self.counts[key] = 1.0
            self.errors[key] = 0.0
            return
        victim = min(self.counts.items(), key=lambda item: (item[1], item[0]))[0]
        inherited = self.counts.pop(victim)
        self.errors.pop(victim, None)
        self.counts[key] = inherited + 1.0
        self.errors[key] = inherited  # the count this key did not earn

    def top(self, k: int) -> "list[tuple[str, float, float]]":
        """``(key, count, error)`` -- the estimate and how much of it may be borrowed."""
        ranked = sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [(key, count, self.errors.get(key, 0.0)) for key, count in ranked]

    @property
    def epsilon(self) -> float:
        return 1.0 / self.capacity

    @property
    def min_count(self) -> float:
        """``eps * N``: the frequency above which a key is guaranteed to be monitored."""
        return self.epsilon * self.total

    def guaranteed_top(self, k: int) -> "list[tuple[str, float, float]]":
        """Only entries whose lower bound clears the guarantee threshold.

        An entry whose ``count - error`` sits below ``eps * N`` may owe its whole rank to a takeover. Dropping
        those turns "here is a top-k list" into "here is a top-k list I can defend".
        """
        return [
            (key, count, error)
            for key, count, error in self.top(k)
            if count - error >= self.min_count
        ]

    def state(self) -> dict:
        return {
            "capacity": self.capacity,
            "counts": dict(self.counts),
            "errors": dict(self.errors),
            "total": self.total,
        }

    @classmethod
    def restore(cls, state: dict) -> "SpaceSaving":
        sketch = cls(state["capacity"])
        sketch.counts = dict(state["counts"])
        sketch.errors = dict(state["errors"])
        sketch.total = state["total"]
        return sketch

    def __repr__(self) -> str:
        return f"SpaceSaving(capacity={self.capacity}, monitored={len(self.counts)}, n={self.total:.0f})"


def topk_agreement(
    exact: ExactTopK, sketch: SpaceSaving, k: int
) -> "tuple[float, float, int]":
    """Compare a sketch against the truth: rank overlap, worst relative count error, and state ratio.

    Overlap on the *set* of top-k keys is the metric that matters -- the exact ordering of ranks 8 and 9 is
    rarely worth memory, while missing a key from the list entirely is a different kind of failure.
    """
    true_top = [key for key, _ in exact.top(k)]
    sketch_top = [key for key, _, _ in sketch.top(k)]
    overlap = len(set(true_top) & set(sketch_top)) / max(len(true_top), 1)
    worst = 0.0
    for key, count, _ in sketch.top(k):
        truth = exact.counts.get(key, 0.0)
        if truth > 0:
            worst = max(worst, abs(count - truth) / truth)
    return overlap, worst, exact.distinct_keys
