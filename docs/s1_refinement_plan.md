# Sentinel-1 SAR Refinement — Implementation Plan

Status: **planned, not implemented.** Resume from this document when ready to code.

## Goal

Add Sentinel-1 SAR backscatter change detection as a second refinement pass on top of the existing Sentinel-2 per-scene refinement in `dbflow_app/pipeline/time_detect.py`. Use the intersection of the S2 and S1 narrow windows to tighten the event window further, then use PRISM daily precipitation to pick a specific event day inside that intersection.

Scope is the **Sentinel-2 backend only**. `time_detect_landsat.py` is untouched because S1 coverage starts Oct 2014 and the Landsat backend targets fires before mid-2015 (i.e., before S1 exists).

The feature is gated behind an opt-in flag and defaults to **off** during development. When the flag is off, the pipeline behaves exactly as it does today.

---

## Context (what exists today)

The Sentinel-2 backend already does two-pass event detection:

1. **Coarse pass.** 30-day median composites of NBR / NDVI / RED over each polygon. A jump in the composite score larger than a local-baseline-derived threshold marks a "coarse event" bounded by `[interval_start_i, interval_end_{i+1}]` (roughly 60 days wide).
2. **Per-scene refinement.** `refine_event_window` pulls every clear Sentinel-2 acquisition in the coarse window ± `refine_margin_days` (15), scores each using the same weights, and returns the adjacent pair with the largest jump — shrinking the window to the Sentinel-2 revisit cadence (~5–15 days). This is gated by `cfg["refine_events"]`.

The CHIRPS precipitation filter was removed in an earlier cleanup; there is currently no precipitation data in the pipeline.

---

## Design decisions (locked in)

| # | Decision | Value |
|---|---|---|
| 1 | Scope | Sentinel-2 backend only; `time_detect.py` only |
| 2 | S1 orbit handling | Ascending and descending as separate time series, results combined per-orbit |
| 3 | "Agreement" definition | Window intersection (Option A) |
| 4 | Precip dataset | PRISM daily (`OREGONSTATE/PRISM/AN81d` in GEE) |
| 5 | Precip role | Strict day-picker inside the agreement window |
| 6 | S2 ∩ S1 disagreement → | Fall back to coarse window |
| 7 | S2 fails but S1 succeeds → | Use S1 window alone |
| 8 | Opt-in flag | `cfg["use_s1_refinement"] = False` default |
| 9 | Minimum polygon area gate for S1 | **Not added here.** User will add a higher-level input-time area gate covering both S2 and Landsat backends. S1 refinement in this pipeline should not duplicate that check. |
| 10 | Zero-precip fallback | Fall back to coarse window (not to midpoint) |
| 11 | VV / VH weighting | VV=0.6, VH=0.4 |
| 12 | Flag surface | Both `DEFAULTS` and `params` override; no UI exposure yet |
| 13 | Orbit strategy config | Keep `s1_orbit_strategy` config key for escape hatches during tuning |
| 14 | Refinement margin | Reuse existing `refine_margin_days` (15) for both S2 and S1 |

---

## Polygon size reference (Pine Gulch 2020, 411 polygons)

Collected from `Hazard Mapping/PINEGULCH2020/pinegulchshp/polygons.shp`:

| Statistic | m² |
|---|---|
| min | 8 |
| 5th percentile | 97 |
| 10th percentile | 228 |
| 25th percentile | 1,128 |
| median | 3,226 |
| 75th percentile | 9,025 |
| 90th percentile | 20,443 |
| 95th percentile | 30,601 |
| max | 89,966 |
| mean | 7,754 |

**Speckle context.** S1 GRD IW is ~10 m pixel, ~4.4 equivalent looks per pixel:
- 1,500 m² ≈ 15 pixels ≈ 66 effective looks → speckle std ≈ 12% of mean (usable)
- 1,000 m² ≈ 10 pixels ≈ 44 looks → speckle std ≈ 15% of mean (marginal)
- 400 m² ≈ 4 pixels ≈ 18 looks → speckle std ≈ 24% of mean (too noisy)

This analysis informed the original "1500 m²" gate proposal, which the user declined in favor of a higher-level input gate. **The implementation does not enforce this inside `time_detect.py`.** Keep the numbers here in case future debugging requires them.

---

## Configuration additions

Add to `DEFAULTS` in `dbflow_app/pipeline/time_detect.py`:

```python
# Sentinel-1 SAR refinement (opt-in, experimental)
"use_s1_refinement": False,
"s1_orbit_strategy": "both",       # "both" | "ascending" | "descending"
"s1_weight_vv": 0.6,
"s1_weight_vh": 0.4,

# PRISM day-picker
"prism_dataset": "OREGONSTATE/PRISM/AN81d",
```

No new `s1_refine_margin_days` — reuse existing `refine_margin_days` so S2 and S1 search the same extended window.

---

## New functions (all in `time_detect.py`)

### `_fetch_s1_acquisitions(ee_polygon, start_str, end_str, orbit_pass, cfg)`

Pulls per-scene polygon-mean VV and VH sigma0 (dB) from `COPERNICUS/S1_GRD`.

Filters applied:
- `instrumentMode == "IW"`
- `orbitProperties_pass == orbit_pass` (literal `"ASCENDING"` or `"DESCENDING"`)
- Date range `[start_str, end_str)` intersecting `ee_polygon`
- `transmitterReceiverPolarisation contains VV` **and** `contains VH` (dual-pol only)

For each image, `reduceRegion(ee.Reducer.mean(), geometry=ee_polygon, scale=10, maxPixels=1e7)` on the VV and VH bands. Returns a list of dicts:

```python
[{"date": "YYYY-MM-DD", "VV": float_dB, "VH": float_dB}, ...]
```

Handles empty collection and per-image nulls gracefully (returns empty list on failure, mirrors `_fetch_individual_acquisitions`).

### `_refine_s1_single_orbit(geom, coarse_iv_start, coarse_iv_end, orbit_pass, cfg)`

Per-orbit S1 refinement. Mirrors `refine_event_window` structure:

1. Build polygon, compute `start_str`, `end_str` using `cfg["refine_margin_days"]`.
2. Call `_fetch_s1_acquisitions` for the specified orbit.
3. Require ≥2 valid scenes; otherwise return `None`.
4. Compute a per-polygon normalized score for each scene:
   - Raw score = `cfg["s1_weight_vv"] * VV + cfg["s1_weight_vh"] * VH` (both in dB)
   - Subtract the median raw score across the scenes in this window (removes site-specific baseline — each polygon is its own reference)
5. Sort by date, find adjacent pair with largest `|Δ|` in normalized score.
6. Return `(pre_dt, post_dt, event_dt)` where `event_dt` is the midpoint, or `None`.

Wrapped in try/except that returns `None` on any failure (matches existing style).

### `refine_event_window_s1(geom, coarse_iv_start, coarse_iv_end, cfg)`

Top-level S1 refinement that respects `cfg["s1_orbit_strategy"]`:

```
if strategy == "ascending":
    return _refine_s1_single_orbit(..., "ASCENDING", ...)
if strategy == "descending":
    return _refine_s1_single_orbit(..., "DESCENDING", ...)
# strategy == "both"
asc = _refine_s1_single_orbit(..., "ASCENDING", ...)
desc = _refine_s1_single_orbit(..., "DESCENDING", ...)
if asc is None and desc is None: return None
if asc is None: return desc
if desc is None: return asc
# both valid → intersect
intersect = _intersect_windows(asc_window, desc_window)
if intersect is None: return None   # orbits disagree → S1 has no opinion
return intersect   # with a recomputed midpoint
```

When asc and desc both exist and overlap, the returned window is their intersection and `event_dt` is the midpoint of the intersection.

### `_intersect_windows(w1, w2)`

Tiny helper. Each window is `(pre_dt, post_dt, event_dt)`. Returns:

```python
overlap_pre  = max(w1.pre,  w2.pre)
overlap_post = min(w1.post, w2.post)
if overlap_post <= overlap_pre:
    return None
return (overlap_pre, overlap_post, overlap_pre + (overlap_post - overlap_pre)/2)
```

Used by both the asc/desc intersection in `refine_event_window_s1` and the S2/S1 intersection in `detect_change_event`.

### `_pick_event_day_prism(geom, pre_dt, post_dt, cfg)`

Uses PRISM daily precipitation to pick a specific event day inside `[pre_dt, post_dt]`.

1. Use `geom.centroid` as the query point (PRISM is 4 km — polygon geometry is moot at that scale).
2. Pull `ImageCollection(cfg["prism_dataset"])` filtered to `[pre_dt, post_dt + 1 day)`, select `"ppt"` band.
3. For each image, `reduceRegion(mean, point, scale=4000)` → `(date, ppt_mm)` list.
4. If window length ≤ 1 day → return that day (even if ppt is zero).
5. If all daily ppt values are zero or the collection is empty → return `None`. The caller interprets this as "zero-precip fallback → use coarse window."
6. Otherwise return the date with the maximum daily precipitation.

Wrapped in try/except returning `None` on failure.

---

## Modified functions

### `detect_change_event(ts, cfg, ref_std=None, geom=None)`

Replace the existing S2-refinement-only block inside the loop:

```python
# Existing:
if geom is not None and cfg.get("refine_events", True):
    r = refine_event_window(geom, coarse_iv_start, coarse_iv_end, cfg)
    if r is not None:
        pre_dt, post_dt, event_dt = r
        candidate_start = pre_dt
        candidate_end = post_dt
        candidate_date = event_dt
        refined = True
```

With this augmented block:

```python
refinement_mode = "coarse"   # "coarse" | "s2" | "s2+s1"

if geom is not None and cfg.get("refine_events", True):
    s2_win = refine_event_window(geom, coarse_iv_start, coarse_iv_end, cfg)
    s1_win = None
    if cfg.get("use_s1_refinement", False):
        s1_win = refine_event_window_s1(geom, coarse_iv_start, coarse_iv_end, cfg)

    # Decision tree
    chosen = None
    if s2_win is not None and s1_win is not None:
        agreement = _intersect_windows(s2_win, s1_win)
        if agreement is not None:
            chosen = agreement
            refinement_mode = "s2+s1"
        # else: disagree → coarse fallback (user decision #6)
    elif s2_win is not None and s1_win is None:
        # Either flag is off, or S1 had no data — keep today's behavior
        chosen = s2_win
        refinement_mode = "s2"
    elif s2_win is None and s1_win is not None:
        # Unusual: S2 failed but S1 succeeded — user decision #7 says accept S1
        chosen = s1_win
        refinement_mode = "s1"
    # else: both None → coarse fallback

    if chosen is not None:
        pre_dt, post_dt, event_dt = chosen
        candidate_start = pre_dt
        candidate_end = post_dt
        candidate_date = event_dt
        refined = True

        # Precip day-picker only applies when S1 agreement is involved
        # (a true agreement window is what justifies pinning to a specific day)
        if refinement_mode == "s2+s1":
            picked_day = _pick_event_day_prism(geom, pre_dt, post_dt, cfg)
            if picked_day is not None:
                candidate_date = picked_day
            else:
                # Zero-precip fallback (user decision #10): collapse to coarse
                candidate_start = coarse_iv_start
                candidate_end = coarse_iv_end
                candidate_date = scores[i + 1]["end_dt"]
                refined = False
                refinement_mode = "coarse"
```

`refinement_mode` is added to the event tuple so `run()` can log it. The tuple grows from:

```python
(candidate_date, candidate_start, candidate_end, change_score, confidence, refined)
```

to:

```python
(candidate_date, candidate_start, candidate_end, change_score, confidence, refined, refinement_mode)
```

**Care point:** update the unpacking in `run()` to match, and update any place that constructs the tuple.

### `run(...)`

Two changes:

1. **Tuple unpacking.** Update the `event_date, start_date, end_date, change_score, confidence, refined = event` line to include `refinement_mode`.

2. **Log line.** Replace the `tag = "refined" if refined else "coarse"` logic with:
   ```python
   tag_map = {"coarse": "coarse", "s2": "refined", "s1": "refined-s1", "s2+s1": "refined+s1"}
   tag = tag_map[refinement_mode]
   ```
   So the log now shows:
   - `[coarse, +/-60d]` — no refinement
   - `[refined, +/-8d]` — S2 only (today's behavior when S1 flag off)
   - `[refined-s1, +/-5d]` — S2 failed, S1 succeeded alone
   - `[refined+s1, +/-3d]` — S2 and S1 agreed; PRISM picked a day

Output shapefile schema is **unchanged**. `EVENT_DATE` takes the PRISM-picked day when `refinement_mode == "s2+s1"`, otherwise the existing midpoint logic. `DATE_START` / `DATE_END` carry whatever window `chosen` produced.

---

## Files touched

| File | Change |
|---|---|
| `dbflow_app/pipeline/time_detect.py` | All new functions + modifications to `detect_change_event` and `run` |
| `dbflow_app/pipeline/time_detect_landsat.py` | **No changes** |
| `dbflow_app/pipeline/merger.py` | **No changes** (schema unchanged) |
| `dbflow_app/app.py` | **No changes** (no UI exposure) |
| `dbflow_app/docs/methodology.md` | Add a new subsection 3.5 "Sentinel-1 SAR refinement (experimental)" describing the logic. Mark as opt-in. Defer this to after implementation. |

---

## Runtime impact

Per polygon, when `use_s1_refinement = True`:
- Existing S2 coarse pass: unchanged
- Existing S2 per-scene refinement: unchanged (1 GEE query)
- **New:** 2 S1 per-scene queries (ascending + descending)
- **New:** 1 PRISM query (cheap — 1 cell, small date range)

Estimated total pipeline runtime increase: **~60–80%** over current. Since the flag defaults to off, this has zero cost on production runs until explicitly enabled.

---

## Testing plan

Before declaring this done:

1. **Smoke test with flag off.** Run pipeline on the Pine Gulch shapefile. Verify `EVENT_DATE` / `DATE_START` / `DATE_END` / `CHG_SCORE` / `CONFIDENCE` match the current baseline exactly for at least 10 polygons. This proves the gating is correct and the S2-only path is untouched.

2. **Smoke test with flag on.** Same shapefile, `use_s1_refinement=True`. Log lines should show a mix of `refined+s1`, `refined`, and `coarse` tags. Sanity-check that `DATE_END - DATE_START` for `refined+s1` rows is smaller than for `refined` rows on average.

3. **Ground-truth spot check.** For 3–5 polygons with known event dates (Grizzly Creek / Cameron Peak from the USGS data release), compare S2-only vs S2+S1 refinement against the truth date. This is validation, not regression — differences are expected and informative.

4. **Edge cases to exercise:**
   - Polygon with only ascending S1 coverage in window (should use asc alone)
   - Polygon where asc and desc windows don't overlap (should fall back to coarse)
   - Polygon where S2 and S1 windows don't overlap (should fall back to coarse)
   - Polygon where PRISM returns zero precip across agreement window (should fall back to coarse)
   - Polygon tiny enough that S1 speckle dominates (check if the higher-level area gate the user plans to add catches this; otherwise watch for spurious agreement)

---

## Open items when resuming

- [ ] Confirm the higher-level input polygon area gate exists before relying on it. If it doesn't, revisit whether to add a local S1 area gate here.
- [ ] Decide whether to also add a minimum-observation-count floor for S1 (e.g., require ≥3 scenes in the extended window before trusting the largest-jump pair). Not strictly necessary for V1.
- [ ] Consider exposing the flag in `app.py` sidebar once the feature is validated on real data. Explicitly out of scope for this plan.
- [ ] After implementation, add a subsection to `docs/methodology.md` documenting the SAR refinement logic and the VV/VH interpretation.

---

## Risks called out

1. **Speckle floor.** Polygons below ~1500 m² will produce noisy S1 signals. User is addressing this at the input layer; verify that gate is live before enabling this feature on production data.
2. **Agreement window can be degenerate.** If S2 picks scenes at days 20 and 25 and S1 picks days 22 and 24, the intersection is [22, 24] — a 2-day window. PRISM then picks one of those 2 days. This is the success case. But if S2 picks [20, 25] and S1 picks [18, 22], the intersection is [20, 22] which is fine; if S1 picks [10, 15] the intersection is empty → coarse fallback. Verify this is what we want empirically.
3. **Directionally ambiguous SAR changes.** Debris flows can increase or decrease backscatter. The `|Δ|` metric handles this, but also means random speckle spikes can masquerade as events. The per-polygon window median normalization partially mitigates this, but this is why we require agreement with S2 rather than trusting S1 alone except in the specific "S2 failed" case.
4. **PRISM boundary.** PRISM daily is CONUS-only. Fires outside CONUS (none expected in CDOT scope) would need a different precip dataset. Document the assumption.
