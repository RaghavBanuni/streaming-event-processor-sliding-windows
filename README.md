# Event-Time Stream Processing From Scratch

Watermarks, tumbling / sliding / session windows with merging, pane decomposition, allowed lateness with
revisions and a side output, Space-Saving heavy hitters with their error bound, and checkpoint-restore.
**Pure Python, standard library only.**

## The one question

A stream has no end, so any aggregate over it requires deciding *when a result may be published* — and that
decision is unanswerable in principle, because more data may always arrive. Every mechanism here is a way of
being explicit about it.

It starts with two timestamps. An event has an **event time** (when the thing happened, on a phone that may
have been in a tunnel) and an **ingest time** (when the pipeline saw it, after a retry and a queue backlog).
Group by ingest time and the 09:00–09:05 revenue figure changes when a consumer lags, and never converges,
because it was never a question about the world. `python -m streamproc.cli eventtime` prints both against the
truth on the same stream:

```
window        truth   event-time   ingest-time    ingest error
[   0,  60)     397          397           383        -3.5%
[  60, 120)     412          412           419        +1.7%
[ 120, 180)     388          388           401        +3.4%
```

Event time reconstructs what happened. Ingest time answers a question about plumbing, and would answer it
differently tomorrow from identical input. Both conserve the total, which is why the error is easy to miss:
nothing is lost, everything is **filed under the wrong window**.

> Numbers depend on the seed. The tests assert the structure: with a watermark lag covering the delays,
> event-time windows reproduce the truth *exactly*, and ingest-time windows on the same data do not.

```bash
python -m streamproc.cli eventtime     # the two clocks, against the truth
python -m streamproc.cli watermark     # the lag trade, swept
python -m streamproc.cli lateness      # revisions and the side output
python -m streamproc.cli panes         # same answers, a sixth of the work
python -m streamproc.cli sessions      # data-driven windows, and merges
python -m streamproc.cli heavyhitters  # Space-Saving vs exact, with error bounds
python -m streamproc.cli restart       # a checkpoint reproduces an uninterrupted run
```

## Watermarks: the trade, not the solution

A watermark is a claim: *no event below time W will arrive from now on*.

```
W = max event time seen - lag
```

`lag` buys completeness with latency and there is no setting that gives both:

```
lag    dropped   dropped %   mean firing delay
  0       1483      37.08%                 0.0
 10        108       2.70%                10.0
 20         79       1.98%                20.0
200          2       0.05%               200.0
```

The parameter is a bet on the tail of a delay distribution that has no maximum — only a quantile someone chose
to believe in. `PercentileWatermark` estimates the lag from observed delays, so a lagging consumer widens the
window automatically; it is still a quantile, and at 0.99 it expects to be wrong 1% of the time. That is what
choosing 0.99 *means*, and it is the honest way to read every watermark configuration in production.

Watermarks are **monotonic by construction**. A window that has fired cannot be un-fired, because downstream
has already seen the result. Which is exactly why late data needs its own path.

## Four explicit decisions about late data

1. **Deduplicate** on event id within a TTL. Queues guarantee at-least-once, so counting without dedup counts
   retries. Dedup state cannot grow forever, so what this implements is *exactly-once within a bounded window
   of time* — and `Stats.dedup_expired` counts the ids forgotten, each of which will be double-counted if a
   retry arrives later. Every production "exactly once" claim carries this asterisk; here it is a number.
2. **Fire** when the watermark passes the window end.
3. **Retain** for `allowed_lateness` and emit a **revision** if a late event lands in the grace period. A
   revision is only worth emitting if the consumer can accept a correction — appended blindly it double-counts,
   which makes a revision-aware pipeline report *worse* numbers than one that ignores late data entirely.
   `totals_by_window` keeps only the latest revision per key for that reason.
4. **Divert** anything later to a counted side output. Not dropped silently: a pipeline that discards data
   without saying so is one whose numbers can never be reconciled against a batch job, and that reconciliation
   is the only real test a streaming aggregate ever gets.

Lateness costs memory — state for every open window is held for the whole grace period — and the tests assert
both halves: more lateness, fewer drops, higher peak state.

## Combinability, and why one property decides everything

```
merge(agg(A), agg(B)) == agg(A + B)
```

Count, sum, min, max and mean-as-(sum, count) satisfy it. An exact median does not, and no engineering makes
it so. Three things follow from the property, all load-bearing:

- **Pane decomposition.** A sliding window of 60 stepping by 10 puts every event in six windows. Aggregate
  once into panes of `gcd(size, step)` and compose windows from panes instead: one update per event, exactly
  `size/step` less work, identical answers (asserted, not assumed). Unavailable for percentiles — which is
  why they are expensive in every stream processor and why sketches exist.
- **Session merging.** A late event landing between two sessions **merges them**, and their aggregates merge
  without revisiting the original events, which the pipeline no longer has. It also means a session's identity
  is provisional: anything downstream that keyed on a session id before the idle gap elapsed keyed on a guess.
- **Checkpointing.** State that merges is state that serialises, splits and reassigns.

## Heavy hitters in bounded memory

Exact top-k needs a counter per distinct key, which on a Zipf key space means state proportional to a tail of
keys seen once each. Space-Saving (Metwally, Agrawal & El Abbadi, 2005) keeps `capacity` counters and takes
over the smallest one when a new key arrives, inheriting its count:

```
capacity   eps      top-10 overlap   worst count error   state ratio
      16   0.0625            70%              38.0%           0.3%
      64   0.0156           100%               6.1%           1.3%
     256   0.0039           100%               1.4%           5.1%
```

The takeover makes every count an **upper bound**, so a genuinely frequent key can never be missed while an
infrequent one may be reported inflated. With `capacity = 1/eps`, any key above `eps*N` is guaranteed to be
monitored — so `guaranteed_top()` returns only entries whose lower bound clears that threshold. The difference
between a top-k list and one you can defend.

## Checkpointing

`checkpoint()` snapshots window state, fired markers, the dedup table, the watermark and session assignments.
The test that matters asserts that processing half a stream, snapshotting, restoring into a fresh processor and
continuing produces output **identical** to an uninterrupted run. That equivalence is the definition; anything
weaker is a backup. The dedup table has to be in the snapshot, or every event replayed after a restart is
counted twice.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Combinability as an algebraic property over every aggregator, pane-vs-naive equality, checkpoint equivalence,
the sketch's never-underestimate direction, hand-built late-data streams where firing order is unambiguous,
half-open window boundaries, watermark monotonicity under an extremely late event, and bounded state growth.

## Limits

- **Single process, single partition.** No shuffle, no key groups, no rescaling, no distributed snapshot
  barrier. Chandy-Lamport is what makes Flink's checkpoints consistent *across operators*, and it is absent
  here — this is the semantics of one operator, not a cluster.
- **No two-phase-commit sink.** Exactly-once end-to-end requires transactional output; dedup at the input is
  only half of it.
- **Pure Python.** Hundreds of thousands of events per minute, not millions per second. State is in memory:
  RocksDB-style spilling and incremental snapshots are what a real deployment adds.
- **No sketches beyond Space-Saving.** No HyperLogLog, no Count-Min, no t-digest; `DistinctExact` is included
  specifically to show the unbounded cost they exist to avoid.
- **No joins.** Stream-stream interval joins and stream-table temporal joins are the other half of practical
  stream processing and are not attempted.
- **Synthetic data.** Delay distributions are exponential with a planted late tail; real ones are worse and
  bimodal (mobile clients reconnecting in batches).

## References

- Akidau et al. (2015), *The Dataflow Model* — event time, windowing, triggers, the completeness trade.
- Akidau (2015), *Streaming 101 / 102* — the clearest statement of why event time is not optional.
- Carbone et al. (2015), *Apache Flink: stream and batch processing in a single engine*.
- Carbone et al. (2015), *Lightweight asynchronous snapshots for distributed dataflows* — Flink checkpointing.
- Li et al. (2005), *No pane, no gain: efficient evaluation of sliding-window aggregates*.
- Metwally, Agrawal & El Abbadi (2005), *Efficient computation of frequent and top-k elements in data streams*.
- Arasu & Widom (2004), *Resource sharing in continuous sliding-window aggregates*.

MIT licensed.
