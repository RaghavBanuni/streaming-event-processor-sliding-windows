"""Demos. Each one measures a claim from the README against a stream whose truth is known.

    python -m streamproc.cli eventtime     processing time vs event time, on identical data
    python -m streamproc.cli watermark     the lag trade: completeness against latency
    python -m streamproc.cli lateness      allowed lateness, revisions, and the side output
    python -m streamproc.cli panes         pane decomposition: same answers, a sixth of the work
    python -m streamproc.cli sessions      session windows and merges, against a known session count
    python -m streamproc.cli heavyhitters  Space-Saving vs exact counting on a Zipf key space
    python -m streamproc.cli restart       checkpoint and restore reproduce an uninterrupted run
    python -m streamproc.cli all
"""

from __future__ import annotations

import sys

from .aggregate import Count, ExactTopK, SpaceSaving, Sum, topk_agreement
from .events import generate, sessions_stream, skewed_stream
from .pipeline import WindowedProcessor, totals_by_window
from .windows import (
    BoundedOutOfOrderness,
    PercentileWatermark,
    SessionWindows,
    SlidingWindows,
    TumblingWindows,
    panes_for,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def demo_eventtime() -> None:
    rule("PROCESSING TIME VS EVENT TIME -- the same data, two different answers")
    spec = generate(n=4000, seed=0)
    print(spec.summary())

    windows = {}
    for domain in ("event", "ingest"):
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(20), dedup_ttl=600, time_domain=domain
        )
        results, _ = processor.run(spec.events)
        windows[domain] = totals_by_window(results)

    truth: dict[int, int] = {}
    for event in spec.unique_events:
        start = (event.event_time // 60) * 60
        truth[start] = truth.get(start, 0) + 1

    print("\nwindow        truth   event-time   ingest-time    ingest error")
    for start in sorted(truth):
        by_event = next(
            (value for window, value in windows["event"].items() if window.start == start), 0.0
        )
        by_ingest = next(
            (value for window, value in windows["ingest"].items() if window.start == start), 0.0
        )
        error = (by_ingest - truth[start]) / truth[start] * 100.0
        print(
            f"[{start:>4},{start + 60:>4})  {truth[start]:>5}   {by_event:>10.0f}   "
            f"{by_ingest:>11.0f}    {error:>+8.1f}%"
        )
    print(
        "\nEvent time reconstructs the world, up to the events too late for the watermark.\n"
        "Ingest time answers a question about the pipeline's plumbing -- and would answer it\n"
        "differently on a slower day, from the same input."
    )


def demo_watermark() -> None:
    rule("THE WATERMARK LAG TRADE -- completeness bought with latency, in both directions")
    spec = generate(n=4000, seed=1)
    print(f"{spec.summary()}\n")
    print("lag    dropped   dropped %   mean firing delay   state peak")
    for lag in (0, 5, 10, 20, 60, 200):
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(lag), dedup_ttl=600
        )
        results, stats = processor.run(spec.events)
        delays = [result.watermark - result.window.end for result in results if not result.revision]
        mean_delay = sum(delays) / len(delays) if delays else 0.0
        share = stats.dropped_too_late / max(stats.processed, 1) * 100.0
        print(
            f"{lag:>3}    {stats.dropped_too_late:>7}   {share:>8.2f}%   {mean_delay:>17.1f}   "
            f"{stats.peak_state_entries:>10}"
        )

    adaptive = WindowedProcessor(
        TumblingWindows(60), Count, PercentileWatermark(0.99), dedup_ttl=600
    )
    _, stats = adaptive.run(spec.events)
    print(
        f"\nPercentileWatermark(0.99): dropped {stats.dropped_too_late} "
        f"({stats.dropped_too_late / max(stats.processed, 1):.2%})"
    )
    print(
        "It adapts the lag to the observed delays, and it is still a quantile: it expects to be\n"
        "wrong about 1% of the time, which is what choosing 0.99 means."
    )


def demo_lateness() -> None:
    rule("ALLOWED LATENESS -- a revision, or a counted side output")
    spec = generate(n=3000, late_fraction=0.05, seed=2)
    print(f"{spec.summary()}\n")
    print("lateness   revisions   dropped   recovered value    state peak")
    for lateness in (0, 30, 120, 600):
        processor = WindowedProcessor(
            TumblingWindows(60),
            Sum,
            BoundedOutOfOrderness(20),
            allowed_lateness=lateness,
            dedup_ttl=900,
        )
        results, stats = processor.run(spec.events)
        recovered = sum(abs(result.value) for result in results if result.revision)
        print(
            f"{lateness:>8}   {stats.revisions:>9}   {stats.dropped_too_late:>7}   "
            f"{recovered:>15.2f}    {stats.peak_state_entries:>10}"
        )
    print(
        "\nLateness buys correctness with memory: state for every open window is retained for the\n"
        "whole grace period. And a revision is only worth emitting if the consumer can accept a\n"
        "correction -- appended blindly, it double-counts."
    )


def demo_panes() -> None:
    rule("PANE DECOMPOSITION -- identical answers, a fraction of the work")
    spec = generate(n=4000, seed=3)
    assigner = SlidingWindows(size=60, step=10)
    print(
        f"sliding {assigner.size} step {assigner.step}: each event belongs to "
        f"{assigner.multiplicity} windows; pane width gcd = {panes_for(60, 10)}\n"
    )
    outputs = {}
    for use_panes in (False, True):
        processor = WindowedProcessor(
            assigner if not use_panes else SlidingWindows(60, 10),
            Count,
            BoundedOutOfOrderness(20),
            dedup_ttl=600,
            use_panes=use_panes,
        )
        results, stats = processor.run(spec.events)
        label = "panes" if use_panes else "naive"
        outputs[label] = totals_by_window(results)
        print(
            f"{label:>6}: {stats.aggregator_updates:>7} aggregator updates, "
            f"peak state {stats.peak_state_entries:>5}, {len(results):>5} results"
        )

    shared = set(outputs["naive"]) & set(outputs["panes"])
    differences = [
        abs(outputs["naive"][window] - outputs["panes"][window]) for window in shared
    ]
    print(
        f"\n{len(shared)} windows in common, largest disagreement {max(differences) if differences else 0.0:.6f}"
    )
    print(
        "The saving is exactly size/step, and it is only available because count is combinable.\n"
        "An exact median cannot be composed from pane medians at any price."
    )


def demo_sessions() -> None:
    rule("SESSION WINDOWS -- data-driven boundaries, and merges")
    spec = sessions_stream(users=40, gap=60, seed=0)
    processor = WindowedProcessor(SessionWindows(gap=60), Count, BoundedOutOfOrderness(10))
    results, stats = processor.run(spec.events)
    lengths = sorted(result.events for result in results)
    print(f"{len(spec.events)} events over {len({event.key for event in spec.events})} users")
    print(f"{len(results)} sessions closed")
    print(
        f"events per session: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}"
    )
    print(f"\n{stats.report()}")
    print(
        "\nA session's identity is provisional: an event landing in the gap merges two sessions into\n"
        "one. Anything downstream that keyed on a session id before the gap elapsed keyed on a guess."
    )


def demo_heavyhitters() -> None:
    rule("HEAVY HITTERS -- Space-Saving against exact counting")
    spec = skewed_stream(n=20000, distinct_keys=5000, seed=0)
    exact = ExactTopK()
    for event in spec.events:
        exact.add(event)

    print(f"{len(spec.events)} events, {exact.distinct_keys} distinct keys\n")
    print("capacity   eps      top-10 overlap   worst count error   guaranteed entries   state ratio")
    for capacity in (16, 64, 256, 1024):
        sketch = SpaceSaving(capacity=capacity)
        for event in spec.events:
            sketch.add(event)
        overlap, worst, distinct = topk_agreement(exact, sketch, k=10)
        print(
            f"{capacity:>8}   {sketch.epsilon:<7.4f}  {overlap:>13.0%}   {worst:>17.2%}   "
            f"{len(sketch.guaranteed_top(10)):>18}   {capacity / distinct:>10.1%}"
        )
    print("\ntrue top 5:", ", ".join(f"{key}={count:.0f}" for key, count in exact.top(5)))
    sketch = SpaceSaving(capacity=64)
    for event in spec.events:
        sketch.add(event)
    print(
        "sketch top 5:",
        ", ".join(f"{key}={count:.0f}+-{error:.0f}" for key, count, error in sketch.top(5)),
    )
    print(
        f"\nEvery count is an upper bound, so a frequent key cannot be missed. Entries below\n"
        f"eps*N = {sketch.min_count:.1f} may owe their rank entirely to a takeover, and guaranteed_top\n"
        "drops them."
    )


def demo_restart() -> None:
    rule("CHECKPOINT AND RESTORE -- a restart must be invisible in the output")
    spec = generate(n=3000, seed=4)
    events = list(spec.events)
    midpoint = len(events) // 2

    straight = WindowedProcessor(
        TumblingWindows(60), Sum, BoundedOutOfOrderness(20), allowed_lateness=60, dedup_ttl=600
    )
    expected, _ = straight.run(events)

    first = WindowedProcessor(
        TumblingWindows(60), Sum, BoundedOutOfOrderness(20), allowed_lateness=60, dedup_ttl=600
    )
    early: list = []
    for event in events[:midpoint]:
        early.extend(first.process(event))
    snapshot = first.checkpoint()

    resumed = WindowedProcessor(
        TumblingWindows(60), Sum, BoundedOutOfOrderness(20), allowed_lateness=60, dedup_ttl=600
    )
    resumed.restore(snapshot)
    late: list = []
    for event in events[midpoint:]:
        late.extend(resumed.process(event))
    late.extend(resumed.close())

    combined = early + late
    same_length = len(combined) == len(expected)
    same_values = all(
        left.key == right.key and left.window == right.window and abs(left.value - right.value) < 1e-9
        for left, right in zip(combined, expected)
    )
    print(f"uninterrupted: {len(expected)} results")
    print(f"checkpointed:  {len(combined)} results  ({midpoint} events, snapshot, then the rest)")
    print(f"snapshot held {len(snapshot['state'])} window states and {len(snapshot['seen'])} dedup ids")
    print(f"\nidentical output: {same_length and same_values}")
    print(
        "That equivalence is what a checkpoint means. The dedup table has to be in the snapshot --\n"
        "without it, every event replayed after the restart is counted a second time."
    )


DEMOS = {
    "eventtime": demo_eventtime,
    "watermark": demo_watermark,
    "lateness": demo_lateness,
    "panes": demo_panes,
    "sessions": demo_sessions,
    "heavyhitters": demo_heavyhitters,
    "restart": demo_restart,
}


def main(argv: "list[str] | None" = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    choice = arguments[0] if arguments else "all"
    if choice == "all":
        for demo in DEMOS.values():
            demo()
        return 0
    if choice not in DEMOS:
        print(f"unknown demo {choice!r}\navailable: {', '.join(DEMOS)}, all")
        return 2
    DEMOS[choice]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
