# Changelog

## Unreleased

## 0.3 (2026081701) — 2026-08-17

- DeepSeek V4 Flash decodes about twice as fast on a Mac: expert reads for
  all three projections start the moment the router names them, the expert
  gather is one dispatch per token instead of one per tile, projections are
  no longer forced one matrix at a time, the FP8 activation rounding uses
  half the operations, and graph-bounding evaluations no longer block. Logits
  are byte-identical to 0.2 at every budget.
- Verification evidence carries forward across published revisions: when a
  model repository gains a commit that only touches its documentation, a
  verified copy stays verified after re-reading only the files that changed,
  and the app says how many it read and how many it carried.
- The memory-budget presets keep their names on one line.
- Diagnostics: an opt-in census of host synchronisations per decode pass
  (`MINIRUN_V4_EVAL_CENSUS=1`) and per-phase attribution in the run record.

## 0.2 (2026081601) — 2026-08-16

- First Developer ID release: signed, notarized DMG at
  downloads.minirun.dev, with Sparkle self-updates.
- DeepSeek V4 Flash: memory dial with device-relative presets, resident layer
  cache and stated-scale execution; per-token phase attribution in Instruments;
  tile digests trusted under held verification authority.
- Kimi K3: sub-layer residency and layer-boundary reclaim, lowering the peak
  on iPhone.
- Verification continues in the background on iOS and finishes with a
  notification; interrupted verifications resume from a checkpoint.
- The memory-budget presets keep their names on one line.

## 0.1 (2026081402) — 2026-08-14

- Initial public developer-preview source distribution.
