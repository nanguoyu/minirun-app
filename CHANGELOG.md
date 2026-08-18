# Changelog

## 0.4 (2026081801) — 2026-08-18

- DeepSeek V4 Flash decodes about a third faster on Mac; the sparse-attention
  heads are computed in one batch. Summing 64 heads together rounds
  differently from summing them one at a time, so wording can differ very
  slightly from 0.3 — the answer is still byte-identical at every memory
  budget, which is the guarantee the memory dial rests on.
- iPhone: *Verify all files* no longer ends the app on iOS 27, where the
  system rejected the background-continuation handler the app registered by
  wildcard; each continuation is now registered by name, and a refusal is
  shown instead of raised.
- The streaming caret follows the text instead of sitting at the end of the
  first line.
- The iOS app targets iPhone only, which is what the product claims; the
  first TestFlight builds come from this release.

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
