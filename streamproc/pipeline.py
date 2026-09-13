"""The processor: dedup, windowing, watermark-driven firing, allowed lateness, panes, checkpoints.

Everything in this file exists to answer one question -- *when is a result final?* -- and the answer is that it
never quite is. What a pipeline can offer instead is a stated policy, which here is four explicit decisions:

1. **Deduplicate** on ``event_id`` within a bounded time-to-live. At-least-once delivery is what queues
   guarantee, so counting without dedup is counting retries. The TTL is the uncomfortable part and it is
   uncomfortable in the code too: dedup state cannot grow forever, so what is implemented is *exactly-once
   within a bounded window of time*, and a duplicate arriving after the TTL will be counted twice. Every
   production "exactly-once" claim contains this asterisk; here it is measurable via ``Stats.dedup_expired``.

2. **Fire** a window when the watermark passes its end. The result is published downstream, which is why the
   watermark may never regress.

3. **Retain** the window's state for ``allowed_lateness`` after firing, so an event arriving in that grace
   period produces a *revised* result rather than being lost. Downstream must then be able to accept a
   correction -- a revision is worse than useless if the consumer only knows how to append.

4. **Divert** anything later than that to a side output. Not silently dropped: counted, and available. A
   pipeline that discards data without telling you is one whose numbers cannot be reconciled with a batch job,
   and that reconciliation is the only real test a streaming aggregate ever gets.

The pane path is the same processor with a different state layout: aggregate each event once into a pane of
``gcd(size, step)`` and compose windows on firing. ``Stats.aggregator_updates`` counts the actual work, so the
saving is reported as a measurement rather than a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .aggregate import Aggregator, merge_all, restore_aggregator
from .events import Event
from .windows import (
    SessionWindows,
    SlidingWindows,
    TumblingWindows,
    Window,
    panes_for,
    panes_in_window,
    restore_watermark,
)


@dataclass(frozen=True)
class Result:
    """One emitted window result. ``revision`` marks a correction to something already published."""

    key: str
    window: Window
    value: float
    events: int
    watermark: int
    revision: bool = False

    def __str__(self) -> str:
        mark = " (revised)" if self.revision else ""
        return f"{self.key} {self.window} = {self.value:.2f} from {self.events} events{mark}"


@dataclass
class Stats:
    """What the pipeline did, including everything it threw away."""

    processed: int = 0
    duplicates_dropped: int = 0
    dedup_expired: int = 0
    late_but_accepted: int = 0
    dropped_too_late: int = 0
    windows_fired: int = 0
    revisions: int = 0
    aggregator_updates: int = 0
    peak_state_entries: int = 0

    def report(self) -> str:
        return (
            f"processed          {self.processed}\n"
            f"duplicates dropped {self.duplicates_dropped}\n"
            f"dedup TTL misses   {self.dedup_expired}   <- counted twice; the asterisk on 'exactly once'\n"
            f"late, accepted     {self.late_but_accepted}\n"
            f"dropped, too late  {self.dropped_too_late}\n"
            f"windows fired      {self.windows_fired}\n"
            f"revisions emitted  {self.revisions}\n"
            f"aggregator updates {self.aggregator_updates}\n"
            f"peak state entries {self.peak_state_entries}"
        )


class WindowedProcessor:
    """A keyed, windowed, event-time aggregation with an explicit late-data policy.

    Parameters
    ----------
    assigner:
        ``TumblingWindows``, ``SlidingWindows`` or ``SessionWindows``.
    aggregator_factory:
        Zero-argument callable returning a fresh combinable aggregator.
    watermark:
        ``BoundedOutOfOrderness`` or ``PercentileWatermark``.
    allowed_lateness:
        Grace period after firing during which a late event produces a revision.
    dedup_ttl:
        How long event ids are remembered, in event time. ``None`` disables dedup, which is the honest way to
        represent "we do not handle duplicates" rather than pretending an unbounded set is a design.
    use_panes:
        Sliding windows only. Aggregate into ``gcd(size, step)`` panes instead of into every overlapping
        window.
    time_domain:
        ``"event"`` (default) or ``"ingest"``. The latter is not a feature; it exists so the demo can measure
        how wrong processing-time windowing is on the same data.
    """

    def __init__(
        self,
        assigner,
        aggregator_factory,
        watermark,
        allowed_lateness: int = 0,
        dedup_ttl: int | None = None,
        use_panes: bool = False,
        time_domain: str = "event",
    ) -> None:
        if time_domain not in {"event", "ingest"}:
            raise ValueError("time_domain must be 'event' or 'ingest'")
        if use_panes and not isinstance(assigner, SlidingWindows):
            raise ValueError("panes apply to sliding windows; other assigners have no overlap to exploit")
        if allowed_lateness < 0:
            raise ValueError("allowed_lateness cannot be negative")

        self.assigner = assigner
        self.factory = aggregator_factory
        self.watermark = watermark
        self.allowed_lateness = allowed_lateness
        self.dedup_ttl = dedup_ttl
        self.use_panes = use_panes
        self.time_domain = time_domain
        self.pane = panes_for(assigner.size, assigner.step) if use_panes else None

        self.state: dict[tuple[str, Window], Aggregator] = {}
        self.counts: dict[tuple[str, Window], int] = {}
        self.fired: set[tuple[str, Window]] = set()
        self.seen: dict[str, int] = {}  # event id -> event time, for dedup within the TTL
        self.side_output: list[Event] = []
        self.stats = Stats()

    # -- ingestion ----------------------------------------------------------------------------

    def _timestamp(self, event: Event) -> int:
        return event.event_time if self.time_domain == "event" else event.ingest_time

    def _is_duplicate(self, event: Event) -> bool:
        if self.dedup_ttl is None or not event.event_id:
            return False
        if event.event_id in self.seen:
            self.stats.duplicates_dropped += 1
            return True
        self.seen[event.event_id] = self._timestamp(event)
        return False

    def _purge_dedup(self) -> None:
        """Forget ids older than the TTL, and count what that costs.

        A duplicate of a forgotten id will be counted twice. That is the bounded-memory price of dedup, and
        counting it is more useful than a docstring promising it will not happen.
        """
        if self.dedup_ttl is None:
            return
        cutoff = self.watermark.current - self.dedup_ttl
        if cutoff < 0:
            return
        expired = [key for key, time in self.seen.items() if time < cutoff]
        for key in expired:
            del self.seen[key]
            self.stats.dedup_expired += 1

    def _add(self, key: str, window: Window, event: Event) -> None:
        entry = (key, window)
        if entry not in self.state:
            self.state[entry] = self.factory()
            self.counts[entry] = 0
        self.state[entry].add(event)
        self.counts[entry] += 1
        self.stats.aggregator_updates += 1

    def process(self, event: Event) -> "list[Result]":
        """Ingest one event and return whatever became final as a consequence."""
        if self._is_duplicate(event):
            return []
        self.stats.processed += 1
        self.watermark.observe(event)
        timestamp = self._timestamp(event)
        results: list[Result] = []

        if isinstance(self.assigner, SessionWindows):
            results.extend(self._process_session(event))
        else:
            for window in self.assigner.assign(timestamp):
                if window.end + self.allowed_lateness <= self.watermark.current:
                    # beyond the grace period: the state is gone and the result is published
                    self.stats.dropped_too_late += 1
                    self.side_output.append(event)
                    continue
                late = window.end <= self.watermark.current
                target = (
                    self._pane_of(timestamp) if self.use_panes else window
                )
                self._add(event.key, target, event)
                if late:
                    self.stats.late_but_accepted += 1
                    if (event.key, window) in self.fired:
                        results.append(self._emit(event.key, window, revision=True))
                if self.use_panes:
                    break  # one pane update covers every window containing this event

        results.extend(self._fire_ready())
        self._purge_dedup()
        self._purge_state()
        self.stats.peak_state_entries = max(self.stats.peak_state_entries, len(self.state))
        return results

    def _pane_of(self, timestamp: int) -> Window:
        assert self.pane is not None
        start = (timestamp // self.pane) * self.pane
        return Window(start, start + self.pane)

    def _process_session(self, event: Event) -> "list[Result]":
        """Sessions: merge the aggregates of any windows the new event joined together.

        This is where combinability pays for itself. Two sessions bridged by one event become a single session
        whose aggregate is the merge of theirs -- no reprocessing of the original events, which the pipeline
        no longer has.
        """
        merged, replaced = self.assigner.add(event)
        parts: list[Aggregator] = []
        total = 0
        for window in replaced:
            entry = (event.key, window)
            if entry in self.state:
                parts.append(self.state.pop(entry))
                total += self.counts.pop(entry, 0)
                self.fired.discard(entry)
        entry = (event.key, merged)
        if parts:
            self.state[entry] = merge_all(parts)
            self.counts[entry] = total
        self._add(event.key, merged, event)
        return []

    def _emit(self, key: str, window: Window, revision: bool = False) -> Result:
        if self.use_panes:
            parts = [
                self.state[(key, pane)]
                for pane in panes_in_window(window, self.pane)  # type: ignore[arg-type]
                if (key, pane) in self.state
            ]
            value = merge_all(parts).result() if parts else 0.0
            events = sum(
                self.counts.get((key, pane), 0)
                for pane in panes_in_window(window, self.pane)  # type: ignore[arg-type]
            )
        else:
            aggregator = self.state.get((key, window))
            value = aggregator.result() if aggregator else 0.0
            events = self.counts.get((key, window), 0)
        if revision:
            self.stats.revisions += 1
        else:
            self.stats.windows_fired += 1
            self.fired.add((key, window))
        return Result(key, window, value, events, self.watermark.current, revision)

    def _fire_ready(self) -> "list[Result]":
        """Emit every window the watermark has passed and that has not fired yet."""
        results = []
        if isinstance(self.assigner, SessionWindows):
            for key, window in self.assigner.closed(self.watermark.current):
                if (key, window) not in self.fired and (key, window) in self.state:
                    results.append(self._emit(key, window))
        elif self.use_panes:
            for key, window in self._pending_pane_windows():
                results.append(self._emit(key, window))
        else:
            for key, window in list(self.state):
                if window.end <= self.watermark.current and (key, window) not in self.fired:
                    results.append(self._emit(key, window))
        return sorted(results, key=lambda result: (result.window, result.key))

    def _pending_pane_windows(self) -> "list[tuple[str, Window]]":
        """Windows implied by the panes currently held, that the watermark has passed."""
        assert isinstance(self.assigner, SlidingWindows) and self.pane is not None
        pending = set()
        for key, pane in list(self.state):
            for window in self.assigner.assign(pane.start):
                if (
                    window.end <= self.watermark.current
                    and (key, window) not in self.fired
                    and window.start >= 0
                ):
                    pending.add((key, window))
        return sorted(pending, key=lambda item: (item[1], item[0]))

    def _purge_state(self) -> None:
        """Drop state whose grace period has elapsed. Without this, a stream leaks until it dies."""
        cutoff = self.watermark.current
        for key, window in list(self.state):
            if self.use_panes:
                # a pane is needed until every window containing it has passed its grace period
                last_window_end = window.start + self.assigner.size
                if last_window_end + self.allowed_lateness <= cutoff:
                    del self.state[(key, window)]
                    self.counts.pop((key, window), None)
            elif window.end + self.allowed_lateness <= cutoff:
                del self.state[(key, window)]
                self.counts.pop((key, window), None)
                if isinstance(self.assigner, SessionWindows):
                    self.assigner.discard(key, window)

    def close(self) -> "list[Result]":
        """Flush at end of stream: emit every window still held.

        A real unbounded stream never reaches this, which is worth stating -- results that only appear at
        ``close()`` are results a live pipeline would never have published.
        """
        results = []
        entries = (
            self._all_pane_windows() if self.use_panes else list(self.state)
        )
        for key, window in entries:
            if (key, window) not in self.fired:
                results.append(self._emit(key, window))
        return sorted(results, key=lambda result: (result.window, result.key))

    def _all_pane_windows(self) -> "list[tuple[str, Window]]":
        assert isinstance(self.assigner, SlidingWindows)
        windows = set()
        for key, pane in list(self.state):
            for window in self.assigner.assign(pane.start):
                if window.start >= 0:
                    windows.add((key, window))
        return sorted(windows, key=lambda item: (item[1], item[0]))

    def run(self, events) -> "tuple[list[Result], Stats]":
        """Process a whole stream in ingest order, then flush."""
        results: list[Result] = []
        for event in events:
            results.extend(self.process(event))
        results.extend(self.close())
        return results, self.stats

    # -- checkpointing ------------------------------------------------------------------------

    def checkpoint(self) -> dict:
        """A serialisable snapshot: window state, fired markers, dedup table, watermark, counters.

        Restoring must reproduce exactly what an uninterrupted run would have produced -- that equivalence is
        the definition of a correct checkpoint, and it is asserted in the tests rather than assumed.
        """
        return {
            "state": [
                {"key": key, "start": window.start, "end": window.end, "agg": aggregator.state(),
                 "count": self.counts.get((key, window), 0)}
                for (key, window), aggregator in self.state.items()
            ],
            "fired": [[key, window.start, window.end] for key, window in self.fired],
            "seen": dict(self.seen),
            "watermark": self.watermark.state(),
            "stats": self.stats.__dict__.copy(),
            "sessions": (
                {
                    key: [[window.start, window.end] for window in windows]
                    for key, windows in self.assigner.sessions.items()
                }
                if isinstance(self.assigner, SessionWindows)
                else None
            ),
        }

    def restore(self, snapshot: dict) -> None:
        self.state = {}
        self.counts = {}
        for entry in snapshot["state"]:
            window = Window(entry["start"], entry["end"])
            self.state[(entry["key"], window)] = restore_aggregator(entry["agg"])
            self.counts[(entry["key"], window)] = entry["count"]
        self.fired = {(key, Window(start, end)) for key, start, end in snapshot["fired"]}
        self.seen = dict(snapshot["seen"])
        self.watermark = restore_watermark(snapshot["watermark"])
        self.stats = Stats(**snapshot["stats"])
        if isinstance(self.assigner, SessionWindows) and snapshot["sessions"] is not None:
            self.assigner.sessions = {
                key: [Window(start, end) for start, end in windows]
                for key, windows in snapshot["sessions"].items()
            }


def tumbling_count(size: int, lag: int, **kwargs) -> WindowedProcessor:
    """The common case, in one call, so demos and tests do not repeat the wiring."""
    from .aggregate import Count
    from .windows import BoundedOutOfOrderness

    return WindowedProcessor(
        TumblingWindows(size), Count, BoundedOutOfOrderness(lag), **kwargs
    )


def totals_by_window(results: "list[Result]") -> "dict[Window, float]":
    """Collapse per-key results into a per-window total, taking the latest revision of each key.

    Revisions make this less obvious than it looks: summing every emitted result double-counts any window that
    was corrected, which is the mistake that makes a revision-aware pipeline report worse numbers than one
    without revisions at all.
    """
    latest: dict[tuple[str, Window], float] = {}
    for result in results:
        latest[(result.key, result.window)] = result.value
    totals: dict[Window, float] = {}
    for (_, window), value in latest.items():
        totals[window] = totals.get(window, 0.0) + value
    return dict(sorted(totals.items()))
