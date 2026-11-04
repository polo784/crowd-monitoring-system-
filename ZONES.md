# Named image regions

The dashboard now estimates crowd in **up to six named rectangles**. Use this to
inspect an entrance or staircase instead of only the full-frame total.

1. Choose a video, Play briefly, then Pause.
2. Select **Draw zone** and drag a rectangle over the image.
3. Enter a name (up to 40 characters), then **Save zone**.
4. A paused snapshot gives the first region estimate. Press Play for updates.
5. Use **Clear zones** to remove the layout. Seek/restart retain zone definitions;
   selecting a different video resets them. Layouts are currently in memory, not
   saved camera profiles. Export JSON to retain their definitions for review.

Saving/clearing zones starts a new segment and clears history. **Export the old
segment first.** An inference already running under the old generation cannot
repopulate the new layout. Drawing is cancelled on backend failure. Stale/current
zone metrics clear along with the full-frame count; outlines remain as definitions,
not evidence of a fresh count. A paused snapshot expires after the existing 5-second
wall-clock freshness rule.

## Count mathematics and limitations

Rectangles use normalized playback-frame edge coordinates:
`[left, top, right, bottom]` in `[0,1]`. This follows the full-frame resize into
the model, including any explicit playback-copy padding. No inference crop or
heatmap-normalization value enters this count.

Each output density cell is treated as uniformly distributed signed mass. The
region count is a float64 sum of raw density multiplied by the fraction of each
cell covered by the rectangle. This avoids losing/double-counting entire boundary
cells when a rectangle crosses the coarse output grid. A full-frame region equals
the raw total; adjacent non-overlapping partitions add to the full count within
floating-point tolerance. Negative contributions remain negative.

**These are approximate regional model estimates, not exact head detections.**
Density smoothing spreads a head's mass across boundaries; the low-resolution grid
and model errors affect localized counts. Whole-frame count accuracy does not prove
ROI accuracy. Overlapping rectangles count shared mass in both zones, so do not add
their counts as if they were disjoint. No physical area, people/m², crowd danger,
localized safety level, flow or movement is inferred. Existing prototype attention
still applies to the whole-frame count only, not to arbitrary region thresholds.

## Provenance and API

`POST /api/control` accepts `action: "zones"` and a `zones` array of
`{"id":"gate","name":"Gate","rect":[0,0,0.5,1]}` objects, with existing session,
generation, timestamp and playing fields. Validation is atomic: an invalid layout
does not advance the generation or destroy history. IDs must be unique; coordinates
finite, ordered and bounded; rectangles nonempty. Requests remain local-only.

Inference/history rows add `zone_counts` keyed by ID. JSON exports include the
rectangle definitions and integration policy. Browser CSV exports also add
`zone_<id>_count` columns. The same raw precision is retained. Exports work from
retained browser history during disconnection; current stale values remain blank.

Backend API version is now `0.2.0`; the launcher will not silently reuse an older
process merely because it loaded the same checkpoint. Stop/relaunch an older
dashboard or use an alternate port. No model/checkpoint change is required.
