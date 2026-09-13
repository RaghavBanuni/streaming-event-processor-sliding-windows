"""Events, and the three ways a real stream differs from the list you tested with.

An event has **two timestamps** and confusing them is the defining error of stream processing:

* ``event_time`` -- when the thing happened. A phone recorded it, possibly while offline in a tunnel.
* ``ingest_time`` -- when the pipeline saw it. Minutes later, after a retry, a queue backlog and a reconnect.

Aggregate by ingest time and the answer depends on your infrastructure's mood. The 09:00-09:05 revenue figure
changes when a consumer lags, and it never converges to anything, because it was never a question about the
world. Aggregate by event time and the answer is a property of what happened -- at the cost of never knowing
for certain that you have seen everything.

This module generates streams with the three properties that make that cost real:

**Out of order.** Events arrive with their event times shuffled within some delay distribution. Anything that
assumes monotonically increasing timestamps produces silently wrong output rather than an error.

**Late.** A small fraction arrives long after the rest -- the phone that was in a tunnel for an hour. Since a
window must close eventually, these are the events a pipeline must consciously decide to drop, buffer, or
re-emit a correction for. There is no fourth option and no setting that makes them free.

**Duplicated.** At-least-once delivery is what queues actually guarantee, so the same event arrives twice.
Any counting aggregate is wrong until duplicates are removed, and dedup state cannot grow forever.

Times are unitless integers, read as seconds. The generators return events in **ingest order**, which is the
only order a pipeline ever gets to see.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class Event:
    """One record. ``sort_index`` first so that sorting a list of events sorts by event time."""

    sort_index: int = field(init=False, repr=False)
    event_time: int
    key: str
    value: float = 1.0
    event_id: str = ""
    ingest_time: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_index", self.event_time)
        if self.event_time < 0:
            raise ValueError("event_time cannot be negative")
        if self.ingest_time and self.ingest_time < self.event_time:
            raise ValueError(
                f"event {self.event_id!r} was ingested at {self.ingest_time} before it happened at "
                f"{self.event_time}: clocks disagree, and that is a data problem rather than a late event"
            )

    @property
    def delay(self) -> int:
        """How far behind the event was when it arrived. Zero for a perfectly punctual stream."""
        return max(self.ingest_time - self.event_time, 0)


@dataclass(frozen=True)
class StreamSpec:
    """The truth about a generated stream, so results can be checked rather than eyeballed."""

    events: "tuple[Event, ...]"
    duplicate_ids: "frozenset[str]"
    late_ids: "frozenset[str]"
    max_delay: int
    description: str

    @property
    def unique_events(self) -> "list[Event]":
        """One event per id, which is what any correct pipeline must count."""
        seen: dict[str, Event] = {}
        for event in self.events:
            seen.setdefault(event.event_id, event)
        return sorted(seen.values())

    def true_total(self, key: str | None = None) -> float:
        return sum(
            event.value for event in self.unique_events if key is None or event.key == key
        )

    def summary(self) -> str:
        delays = [event.delay for event in self.events]
        return (
            f"{len(self.events)} arrivals, {len(self.unique_events)} distinct events, "
            f"{len(self.duplicate_ids)} duplicated, {len(self.late_ids)} late\n"
            f"delay: median {sorted(delays)[len(delays) // 2]}, max {max(delays)} "
            f"(generator bound {self.max_delay})\n{self.description}"
        )


def generate(
    n: int = 4000,
    keys: "tuple[str, ...]" = ("checkout", "search", "signup", "refund"),
    horizon: int = 600,
    typical_delay: int = 4,
    max_delay: int = 20,
    late_fraction: float = 0.02,
    late_delay: int = 400,
    duplicate_fraction: float = 0.03,
    seed: int = 0,
) -> StreamSpec:
    """A stream that is out of order, occasionally very late, and sometimes duplicated.

    ``typical_delay`` shapes the bulk of the distribution (exponential, so most events are nearly punctual
    and a few are not), while ``max_delay`` bounds it -- which is exactly the assumption a bounded-delay
    watermark makes. ``late_fraction`` then breaks that assumption on purpose, because in production it is
    always broken: a delay distribution has no maximum, only a quantile you chose to believe in.

    Duplicates are appended at a later ingest time than the original, since a retry is by definition after
    the first attempt.
    """
    if not 0.0 <= late_fraction < 1.0:
        raise ValueError("late_fraction must lie in [0, 1)")
    rng = random.Random(seed)
    arrivals: list[Event] = []
    duplicates: set[str] = set()
    late: set[str] = set()

    for index in range(n):
        event_time = rng.randrange(horizon)
        key = rng.choice(keys)
        event_id = f"e{index:06d}"
        if rng.random() < late_fraction:
            delay = late_delay + rng.randrange(late_delay)
            late.add(event_id)
        else:
            delay = min(int(rng.expovariate(1.0 / max(typical_delay, 1))), max_delay)
        value = round(rng.uniform(1.0, 50.0), 2) if key != "refund" else -round(rng.uniform(1.0, 30.0), 2)
        arrivals.append(
            Event(
                event_time=event_time,
                key=key,
                value=value,
                event_id=event_id,
                ingest_time=event_time + delay,
            )
        )
        if rng.random() < duplicate_fraction:
            duplicates.add(event_id)
            arrivals.append(
                Event(
                    event_time=event_time,
                    key=key,
                    value=value,
                    event_id=event_id,
                    ingest_time=event_time + delay + rng.randrange(1, 30),
                )
            )

    arrivals.sort(key=lambda event: (event.ingest_time, event.event_id))
    description = (
        f"delays are exponential with mean {typical_delay} truncated at {max_delay}, except "
        f"{late_fraction:.0%} of events which arrive around {late_delay} late -- the tail that no "
        "watermark can cover without waiting forever."
    )
    return StreamSpec(
        events=tuple(arrivals),
        duplicate_ids=frozenset(duplicates),
        late_ids=frozenset(late),
        max_delay=max_delay,
        description=description,
    )


def sessions_stream(
    users: int = 40, horizon: int = 3600, gap: int = 60, seed: int = 0
) -> StreamSpec:
    """Bursts of activity per user separated by idle gaps: what session windows are for.

    Each user has a few bursts; within a burst events are seconds apart, and bursts are separated by more
    than ``gap``. The correct session count is therefore known by construction, which is what makes the
    session-window tests meaningful rather than decorative.
    """
    rng = random.Random(seed)
    arrivals: list[Event] = []
    index = 0
    for user in range(users):
        time = rng.randrange(120)
        for _ in range(rng.randint(1, 4)):
            burst_length = rng.randint(2, 8)
            for _ in range(burst_length):
                delay = min(int(rng.expovariate(0.5)), 5)
                arrivals.append(
                    Event(
                        event_time=time,
                        key=f"user_{user:03d}",
                        value=1.0,
                        event_id=f"s{index:06d}",
                        ingest_time=time + delay,
                    )
                )
                index += 1
                time += rng.randint(1, max(gap // 4, 2))
            time += gap + rng.randint(5, gap)  # an idle gap, so the next burst is a new session
            if time > horizon:
                break
    arrivals.sort(key=lambda event: (event.ingest_time, event.event_id))
    return StreamSpec(
        events=tuple(arrivals),
        duplicate_ids=frozenset(),
        late_ids=frozenset(),
        max_delay=5,
        description=f"bursty per-user activity with idle gaps above {gap}: session windows should split on them",
    )


def skewed_stream(n: int = 20000, distinct_keys: int = 5000, seed: int = 0) -> StreamSpec:
    """A Zipf-like key distribution: a few keys dominate and a long tail appears once each.

    This is the shape that makes exact top-k expensive -- the state is proportional to the number of distinct
    keys, most of which will never matter -- and it is the shape that Space-Saving is designed for.
    """
    rng = random.Random(seed)
    weights = [1.0 / (rank + 1) ** 1.1 for rank in range(distinct_keys)]
    total = sum(weights)
    cumulative: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)

    arrivals: list[Event] = []
    for index in range(n):
        draw = rng.random()
        low, high = 0, distinct_keys - 1
        while low < high:  # binary search on the cumulative distribution
            middle = (low + high) // 2
            if cumulative[middle] < draw:
                low = middle + 1
            else:
                high = middle
        arrivals.append(
            Event(
                event_time=index,
                key=f"k{low:05d}",
                value=1.0,
                event_id=f"z{index:06d}",
                ingest_time=index,
            )
        )
    return StreamSpec(
        events=tuple(arrivals),
        duplicate_ids=frozenset(),
        late_ids=frozenset(),
        max_delay=0,
        description=f"Zipf(1.1) over {distinct_keys} keys: a handful dominate, most appear once",
    )


STREAMS = {"mixed": generate, "sessions": sessions_stream, "skewed": skewed_stream}
