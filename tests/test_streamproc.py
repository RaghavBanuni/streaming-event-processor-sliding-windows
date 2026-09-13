"""Tests. Properties first, then the two equivalences that define correctness here.

The load-bearing tests are:

* ``test_combinability_holds_for_every_combinable_aggregator`` -- the algebraic property that panes, session
  merging and checkpointing all rest on. If it fails, every other result in this repository is coincidence.
* ``test_panes_and_naive_sliding_agree_exactly`` -- the optimisation must not change the answer.
* ``test_a_checkpointed_run_reproduces_an_uninterrupted_one`` -- the definition of a correct checkpoint.
* ``test_space_saving_never_underestimates`` -- the direction of the error is what makes the sketch usable.

Ground truth comes from the generators, so window contents are compared against what was actually emitted
rather than against a previous run of the same code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamproc.aggregate import (  # noqa: E402
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
from streamproc.events import Event, generate, sessions_stream, skewed_stream  # noqa: E402
from streamproc.pipeline import WindowedProcessor, totals_by_window  # noqa: E402
from streamproc.windows import (  # noqa: E402
    BoundedOutOfOrderness,
    PercentileWatermark,
    SessionWindows,
    SlidingWindows,
    TumblingWindows,
    Window,
    panes_for,
    panes_in_window,
    restore_watermark,
)


def event(time: int, key: str = "k", value: float = 1.0, ident: str = "", delay: int = 0) -> Event:
    return Event(
        event_time=time, key=key, value=value, event_id=ident or f"id{time}", ingest_time=time + delay
    )


def truth_by_window(spec, size: int) -> "dict[int, int]":
    """Per-window event counts computed directly from the deduplicated stream."""
    counts: dict[int, int] = {}
    for item in spec.unique_events:
        start = (item.event_time // size) * size
        counts[start] = counts.get(start, 0) + 1
    return counts


class TestEvents:
    def test_ingest_before_event_time_is_rejected(self):
        with pytest.raises(ValueError, match="clocks disagree"):
            Event(event_time=100, key="k", ingest_time=50)

    def test_events_sort_by_event_time(self):
        assert sorted([event(5), event(1), event(3)])[0].event_time == 1

    def test_arrivals_are_in_ingest_order(self):
        spec = generate(n=500, seed=0)
        times = [item.ingest_time for item in spec.events]
        assert times == sorted(times)

    def test_the_stream_is_actually_out_of_order_in_event_time(self):
        """Without this the whole exercise is vacuous: a sorted stream hides every bug in it."""
        spec = generate(n=500, seed=0)
        event_times = [item.event_time for item in spec.events]
        assert event_times != sorted(event_times)

    def test_duplicates_share_an_id_and_arrive_later(self):
        spec = generate(n=800, duplicate_fraction=0.1, seed=1)
        assert spec.duplicate_ids
        assert len(spec.events) > len(spec.unique_events)
        by_id: dict[str, list[Event]] = {}
        for item in spec.events:
            by_id.setdefault(item.event_id, []).append(item)
        for ident in spec.duplicate_ids:
            arrivals = sorted(by_id[ident], key=lambda item: item.ingest_time)
            assert arrivals[1].ingest_time > arrivals[0].ingest_time
            assert arrivals[1].event_time == arrivals[0].event_time

    def test_late_events_exceed_the_stated_delay_bound(self):
        spec = generate(n=2000, late_fraction=0.05, seed=2)
        late = [item for item in spec.events if item.event_id in spec.late_ids]
        assert late and all(item.delay > spec.max_delay for item in late)


class TestWindows:
    def test_tumbling_windows_are_half_open_at_the_boundary(self):
        assigner = TumblingWindows(60)
        assert assigner.assign(59) == [Window(0, 60)]
        assert assigner.assign(60) == [Window(60, 120)]  # not in both

    def test_every_sliding_window_returned_contains_the_timestamp(self):
        assigner = SlidingWindows(size=60, step=10)
        for time in (0, 7, 55, 60, 137, 1000):
            for window in assigner.assign(time):
                assert window.contains(time)

    def test_sliding_multiplicity_is_size_over_step(self):
        assigner = SlidingWindows(size=60, step=10)
        assert len(assigner.assign(500)) == assigner.multiplicity == 6

    def test_early_timestamps_get_fewer_windows_not_negative_ones(self):
        assigner = SlidingWindows(size=60, step=10)
        assert all(window.start >= 0 for window in assigner.assign(5))

    def test_a_step_above_size_is_rejected(self):
        with pytest.raises(ValueError, match="gaps"):
            SlidingWindows(size=10, step=60)

    def test_empty_window_is_rejected(self):
        with pytest.raises(ValueError, match="empty or inverted"):
            Window(60, 60)

    def test_pane_width_is_the_gcd_and_tiles_the_window(self):
        assert panes_for(60, 10) == 10
        assert panes_for(60, 25) == 5
        panes = panes_in_window(Window(0, 60), 10)
        assert len(panes) == 6
        assert panes[0].start == 0 and panes[-1].end == 60


class TestWatermarks:
    def test_the_watermark_never_regresses(self):
        watermark = BoundedOutOfOrderness(lag=10)
        watermark.observe(event(1000))
        high = watermark.current
        watermark.observe(event(5))  # extremely late
        assert watermark.current == high

    def test_the_watermark_is_max_seen_minus_lag(self):
        watermark = BoundedOutOfOrderness(lag=10)
        watermark.observe(event(100))
        assert watermark.current == 90

    def test_percentile_watermark_adapts_to_the_delays_it_sees(self):
        punctual = PercentileWatermark(percentile=0.9, window=100)
        for time in range(200):
            punctual.observe(event(time, delay=1))
        lagging = PercentileWatermark(percentile=0.9, window=100)
        for time in range(200):
            lagging.observe(event(time, delay=50))
        assert lagging.estimated_lag > punctual.estimated_lag

    def test_a_watermark_survives_a_round_trip_through_a_checkpoint(self):
        watermark = PercentileWatermark(percentile=0.95)
        for time in range(50):
            watermark.observe(event(time, delay=time % 7))
        restored = restore_watermark(watermark.state())
        assert restored.current == watermark.current
        assert restored.estimated_lag == watermark.estimated_lag


class TestAggregators:
    @pytest.mark.parametrize("factory", [Count, Sum, Mean, MinMax, DistinctExact])
    def test_combinability_holds_for_every_combinable_aggregator(self, factory):
        """merge(agg(A), agg(B)) == agg(A + B). Panes, session merges and checkpoints all need this."""
        left_events = [event(index, key=f"k{index % 3}", value=float(index)) for index in range(7)]
        right_events = [event(index, key=f"k{index % 5}", value=-float(index)) for index in range(7, 15)]

        left, right, whole = factory(), factory(), factory()
        for item in left_events:
            left.add(item)
        for item in right_events:
            right.add(item)
        for item in left_events + right_events:
            whole.add(item)

        assert left.merge(right).result() == pytest.approx(whole.result())

    def test_mean_merges_correctly_which_requires_carrying_the_count(self):
        one, two = Mean(), Mean()
        for value in (1.0, 2.0, 3.0):
            one.add(event(0, value=value))
        for value in (100.0,):
            two.add(event(0, value=value))
        # not (2 + 100)/2 = 51: the correct answer weights by count
        assert one.merge(two).result() == pytest.approx(106.0 / 4.0)

    def test_empty_minmax_reports_zero_rather_than_nan(self):
        assert MinMax().result() == 0.0

    def test_merge_all_folds_a_list(self):
        parts = []
        for index in range(5):
            part = Count()
            for _ in range(index):
                part.add(event(0))
            parts.append(part)
        assert merge_all(parts).result() == pytest.approx(10.0)

    def test_merging_nothing_is_an_error_not_a_zero(self):
        with pytest.raises(ValueError, match="nothing to merge"):
            merge_all([])


class TestEventTimeVersusIngestTime:
    def test_event_time_recovers_the_truth_when_the_lag_covers_the_delays(self):
        spec = generate(n=2000, late_fraction=0.0, max_delay=15, seed=3)
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(lag=15), dedup_ttl=600
        )
        results, stats = processor.run(spec.events)
        totals = {window.start: value for window, value in totals_by_window(results).items()}
        assert stats.dropped_too_late == 0
        assert totals == pytest.approx(
            {start: float(count) for start, count in truth_by_window(spec, 60).items()}
        )

    def test_ingest_time_windows_disagree_with_the_truth(self):
        """The same data, the same code path, a different timestamp -- and wrong answers."""
        spec = generate(n=2000, late_fraction=0.0, max_delay=15, seed=3)
        processor = WindowedProcessor(
            TumblingWindows(60),
            Count,
            BoundedOutOfOrderness(lag=15),
            dedup_ttl=600,
            time_domain="ingest",
        )
        results, _ = processor.run(spec.events)
        totals = {window.start: value for window, value in totals_by_window(results).items()}
        truth = truth_by_window(spec, 60)
        assert any(
            abs(totals.get(start, 0.0) - count) > 0.5 for start, count in truth.items()
        )

    def test_total_events_are_conserved_in_both_domains(self):
        """Both assign every event to some window; only event time assigns them to the right one."""
        spec = generate(n=1500, late_fraction=0.0, max_delay=15, seed=4)
        for domain in ("event", "ingest"):
            processor = WindowedProcessor(
                TumblingWindows(60),
                Count,
                BoundedOutOfOrderness(lag=15),
                dedup_ttl=600,
                time_domain=domain,
            )
            results, _ = processor.run(spec.events)
            assert sum(totals_by_window(results).values()) == pytest.approx(
                float(len(spec.unique_events))
            )


class TestDeduplication:
    def test_duplicates_are_dropped_and_the_count_matches_the_unique_events(self):
        spec = generate(n=1500, duplicate_fraction=0.2, late_fraction=0.0, max_delay=10, seed=5)
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(10), dedup_ttl=10_000
        )
        results, stats = processor.run(spec.events)
        assert stats.duplicates_dropped == len(spec.events) - len(spec.unique_events)
        assert sum(totals_by_window(results).values()) == pytest.approx(float(len(spec.unique_events)))

    def test_without_dedup_the_count_is_inflated_by_exactly_the_retries(self):
        spec = generate(n=1000, duplicate_fraction=0.2, late_fraction=0.0, max_delay=10, seed=5)
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(10), dedup_ttl=None
        )
        results, _ = processor.run(spec.events)
        assert sum(totals_by_window(results).values()) == pytest.approx(float(len(spec.events)))

    def test_an_expiring_ttl_is_counted_rather_than_hidden(self):
        """Bounded dedup state means bounded-time exactly-once, and the shortfall is measurable."""
        spec = generate(n=1000, duplicate_fraction=0.2, late_fraction=0.0, max_delay=10, seed=6)
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(10), dedup_ttl=5
        )
        _, stats = processor.run(spec.events)
        assert stats.dedup_expired > 0
        assert len(processor.seen) < len(spec.unique_events)  # state stayed bounded


class TestLateness:
    def test_a_late_event_within_the_grace_period_produces_a_revision(self):
        """Hand-built: window [0,60) fires, then an event for it arrives inside the grace period."""
        processor = WindowedProcessor(
            TumblingWindows(60),
            Count,
            BoundedOutOfOrderness(lag=0),
            allowed_lateness=60,
        )
        for time in (10, 20, 100):  # the 100 pushes the watermark past 60 and fires the window
            processor.process(event(time, ident=f"a{time}"))
        assert processor.stats.windows_fired == 1
        revisions = processor.process(event(30, ident="late"))
        assert len(revisions) == 1
        assert revisions[0].revision is True
        assert revisions[0].value == pytest.approx(3.0)

    def test_beyond_the_grace_period_the_event_goes_to_the_side_output(self):
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(lag=0), allowed_lateness=0
        )
        for time in (10, 20, 200):
            processor.process(event(time, ident=f"b{time}"))
        assert processor.process(event(30, ident="way-late")) == []
        assert processor.stats.dropped_too_late == 1
        assert [item.event_id for item in processor.side_output] == ["way-late"]

    def test_more_lateness_recovers_more_and_holds_more_state(self):
        spec = generate(n=2000, late_fraction=0.05, seed=7)
        dropped, peaks = [], []
        for lateness in (0, 600):
            processor = WindowedProcessor(
                TumblingWindows(60),
                Count,
                BoundedOutOfOrderness(20),
                allowed_lateness=lateness,
                dedup_ttl=900,
            )
            _, stats = processor.run(spec.events)
            dropped.append(stats.dropped_too_late)
            peaks.append(stats.peak_state_entries)
        assert dropped[1] < dropped[0]  # fewer events lost
        assert peaks[1] >= peaks[0]  # paid for with retained state

    def test_totals_by_window_does_not_double_count_a_revision(self):
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(0), allowed_lateness=60
        )
        results = []
        for time in (10, 20, 100):
            results.extend(processor.process(event(time, ident=f"c{time}")))
        results.extend(processor.process(event(30, ident="late")))
        results.extend(processor.close())
        totals = totals_by_window(results)
        assert totals[Window(0, 60)] == pytest.approx(3.0)  # not 2 + 3


class TestPanes:
    def test_panes_and_naive_sliding_agree_exactly(self):
        spec = generate(n=2000, late_fraction=0.0, max_delay=10, seed=8)
        outputs = {}
        updates = {}
        for use_panes in (False, True):
            processor = WindowedProcessor(
                SlidingWindows(60, 10),
                Count,
                BoundedOutOfOrderness(10),
                dedup_ttl=600,
                use_panes=use_panes,
            )
            results, stats = processor.run(spec.events)
            outputs[use_panes] = totals_by_window(results)
            updates[use_panes] = stats.aggregator_updates
        common = set(outputs[True]) & set(outputs[False])
        assert len(common) > 10
        for window in common:
            assert outputs[True][window] == pytest.approx(outputs[False][window])

    def test_panes_do_one_update_per_event_instead_of_six(self):
        spec = generate(n=1000, late_fraction=0.0, max_delay=10, seed=8)
        counts = {}
        for use_panes in (False, True):
            processor = WindowedProcessor(
                SlidingWindows(60, 10),
                Count,
                BoundedOutOfOrderness(10),
                dedup_ttl=600,
                use_panes=use_panes,
            )
            _, stats = processor.run(spec.events)
            counts[use_panes] = stats.aggregator_updates
        assert counts[False] > 4 * counts[True]

    def test_panes_are_rejected_for_assigners_without_overlap(self):
        with pytest.raises(ValueError, match="panes apply to sliding windows"):
            WindowedProcessor(
                TumblingWindows(60), Count, BoundedOutOfOrderness(0), use_panes=True
            )


class TestSessions:
    def test_an_event_in_the_gap_merges_two_sessions(self):
        """The hard case, built by hand: [0,100) and [150,250) joined by an event at 90."""
        assigner = SessionWindows(gap=100)
        assigner.add(event(0, key="u"))
        assigner.add(event(150, key="u"))
        assert len(assigner.sessions["u"]) == 2
        merged, replaced = assigner.add(event(90, key="u"))
        assert len(replaced) == 2
        assert merged == Window(0, 250)
        assert assigner.sessions["u"] == [Window(0, 250)]

    def test_a_merge_combines_the_aggregates_without_reprocessing(self):
        processor = WindowedProcessor(SessionWindows(gap=100), Count, BoundedOutOfOrderness(0))
        for time in (0, 10, 150, 160):
            processor.process(event(time, key="u", ident=f"s{time}"))
        processor.process(event(90, key="u", ident="bridge"))
        entries = [entry for entry in processor.state if entry[0] == "u"]
        assert len(entries) == 1
        assert processor.counts[entries[0]] == 5  # all five events, none reprocessed

    def test_distant_bursts_stay_separate(self):
        processor = WindowedProcessor(SessionWindows(gap=60), Count, BoundedOutOfOrderness(0))
        results = []
        for time in (0, 5, 1000, 1005, 5000):
            results.extend(processor.process(event(time, key="u", ident=f"t{time}")))
        results.extend(processor.close())
        assert len(results) == 3

    def test_sessions_close_and_every_event_is_accounted_for(self):
        spec = sessions_stream(users=20, gap=60, seed=0)
        processor = WindowedProcessor(SessionWindows(gap=60), Count, BoundedOutOfOrderness(10))
        results, _ = processor.run(spec.events)
        assert sum(result.value for result in results) == pytest.approx(float(len(spec.events)))


class TestCheckpointing:
    def test_a_checkpointed_run_reproduces_an_uninterrupted_one(self):
        spec = generate(n=1500, duplicate_fraction=0.1, seed=9)
        events = list(spec.events)
        settings = dict(allowed_lateness=60, dedup_ttl=600)

        straight = WindowedProcessor(
            TumblingWindows(60), Sum, BoundedOutOfOrderness(20), **settings
        )
        expected, _ = straight.run(events)

        first = WindowedProcessor(TumblingWindows(60), Sum, BoundedOutOfOrderness(20), **settings)
        produced = []
        for item in events[: len(events) // 2]:
            produced.extend(first.process(item))
        snapshot = first.checkpoint()

        resumed = WindowedProcessor(TumblingWindows(60), Sum, BoundedOutOfOrderness(20), **settings)
        resumed.restore(snapshot)
        for item in events[len(events) // 2 :]:
            produced.extend(resumed.process(item))
        produced.extend(resumed.close())

        assert len(produced) == len(expected)
        for left, right in zip(produced, expected):
            assert (left.key, left.window, left.revision) == (right.key, right.window, right.revision)
            assert left.value == pytest.approx(right.value)

    def test_the_snapshot_carries_the_dedup_table(self):
        """Without it, everything replayed after a restart is counted twice."""
        spec = generate(n=400, duplicate_fraction=0.3, seed=10)
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(10), dedup_ttl=10_000
        )
        for item in spec.events[:200]:
            processor.process(item)
        assert processor.checkpoint()["seen"]

    def test_session_state_survives_a_checkpoint(self):
        processor = WindowedProcessor(SessionWindows(gap=100), Count, BoundedOutOfOrderness(0))
        for time in (0, 10, 500):
            processor.process(event(time, key="u", ident=f"v{time}"))
        snapshot = processor.checkpoint()
        resumed = WindowedProcessor(SessionWindows(gap=100), Count, BoundedOutOfOrderness(0))
        resumed.restore(snapshot)
        assert resumed.assigner.sessions == processor.assigner.sessions


class TestStateGrowth:
    def test_state_does_not_grow_without_bound(self):
        """A pipeline that never purges dies in production, and the death is slow enough to ship."""
        spec = generate(n=6000, horizon=3000, late_fraction=0.0, max_delay=10, seed=11)
        processor = WindowedProcessor(
            TumblingWindows(60), Count, BoundedOutOfOrderness(10), dedup_ttl=120
        )
        processor.run(spec.events)
        assert len(processor.state) <= 8  # only recent windows remain open
        assert len(processor.seen) < 400


class TestHeavyHitters:
    def test_space_saving_by_hand_takes_over_the_smallest_counter(self):
        sketch = SpaceSaving(capacity=2)
        for key in ("a", "a", "b", "c"):
            sketch.observe(key)
        assert sketch.counts["a"] == 2.0
        # 'c' evicts 'b' (count 1), inherits it, and records the inherited count as error
        assert sketch.counts["c"] == 2.0
        assert sketch.errors["c"] == 1.0
        assert "b" not in sketch.counts

    def test_space_saving_never_underestimates(self):
        """The direction of the error is the property that makes the sketch usable."""
        spec = skewed_stream(n=5000, distinct_keys=800, seed=0)
        exact, sketch = ExactTopK(), SpaceSaving(capacity=32)
        for item in spec.events:
            exact.add(item)
            sketch.add(item)
        for key, count in sketch.counts.items():
            assert count >= exact.counts[key] - 1e-9

    def test_the_true_count_lies_within_the_error_bound(self):
        spec = skewed_stream(n=5000, distinct_keys=800, seed=1)
        exact, sketch = ExactTopK(), SpaceSaving(capacity=32)
        for item in spec.events:
            exact.add(item)
            sketch.add(item)
        for key, count, error in sketch.top(32):
            assert count - error - 1e-9 <= exact.counts[key] <= count + 1e-9

    def test_frequent_keys_are_always_monitored(self):
        """Anything above eps*N is guaranteed present; that is the theorem, checked on data."""
        spec = skewed_stream(n=8000, distinct_keys=2000, seed=2)
        exact, sketch = ExactTopK(), SpaceSaving(capacity=64)
        for item in spec.events:
            exact.add(item)
            sketch.add(item)
        threshold = sketch.min_count
        for key, count in exact.counts.items():
            if count > threshold:
                assert key in sketch.counts

    def test_more_capacity_recovers_more_of_the_true_top_k(self):
        spec = skewed_stream(n=10000, distinct_keys=3000, seed=3)
        exact = ExactTopK()
        for item in spec.events:
            exact.add(item)
        overlaps = []
        for capacity in (8, 256):
            sketch = SpaceSaving(capacity=capacity)
            for item in spec.events:
                sketch.add(item)
            overlaps.append(topk_agreement(exact, sketch, k=10)[0])
        assert overlaps[1] >= overlaps[0]
        assert overlaps[1] == pytest.approx(1.0)

    def test_guaranteed_top_is_a_subset_of_top(self):
        spec = skewed_stream(n=4000, distinct_keys=1500, seed=4)
        sketch = SpaceSaving(capacity=16)
        for item in spec.events:
            sketch.add(item)
        assert {key for key, _, _ in sketch.guaranteed_top(10)} <= {
            key for key, _, _ in sketch.top(10)
        }

    def test_a_sketch_survives_a_round_trip(self):
        sketch = SpaceSaving(capacity=8)
        for key in "abcdefghijabc":
            sketch.observe(key)
        restored = SpaceSaving.restore(sketch.state())
        assert restored.counts == sketch.counts
        assert restored.top(5) == sketch.top(5)
