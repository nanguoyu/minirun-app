# Changelog

## Unreleased

- The reference run finished. DeepSeek's own inference code, run over the same
  517 GB container on a data-centre GPU, answers the same question with the same
  eighteen tokens Minirun produces on a Mac and on an iPhone. The 0.5 notes said
  that run had not finished; it has, and it agrees.

## 0.5 (2026091101) — 2026-09-11

- DeepSeek V4.1 Flash chats on Mac and iPhone. The 517 GB container answers on
  both devices now. On a MacBook Pro (M1 Pro, 32 GB) over a USB4 enclosure it
  runs at about 5 seconds a token at the 14.9 GB Balanced budget, where it holds
  all forty blocks and the output head and a token reads only the 4.5 GB of
  experts it routes to; the first reply starts after about half a minute. On an
  iPhone 16 Pro over a powered dock it runs at about 21 seconds a token at the
  1.9 GB floor, where it can hold nothing and re-reads 11.7 GB a token, and the
  first reply takes about a minute. The same question produced the same eighteen
  tokens at every budget it has been asked at and on both devices, which is the
  property the dial is built on: a budget changes what a run costs, not what it
  says. Whether those are the tokens the published checkpoint computes is a
  different question, and the reference run that answers it has not finished.
- The memory dial reaches DeepSeek V4.1 Flash. Its three presets were one
  number — Floor, Balanced and Generous all read 3.40 GB — because the dial had
  never read what the model is made of, so dragging the slider bought nothing
  and every chat streamed all forty blocks from the drive on every token. It now
  reads the container's own manifests, and the presets are positions on a real
  ladder: Floor still runs in 3.40 GB and holds nothing, and Balanced holds all
  forty blocks and the output head in 14.9 GB, which drops what a token reads
  from 11.7 GB to 4.5 GB and roughly halves the time a word takes. On a Mac with
  the room for it, a new chat starts at Balanced; on a phone it starts at Floor,
  and the dial says by how much Balanced is out of reach rather than pretending
  the phone can hold it.
- A finished download is verified. Minirun already checks every file against the
  digest the repository publishes as it lands, and checks the whole folder again
  before it calls the transfer done — and then threw that away, so a 517 GB
  download ended with the copy reading *Not verified* and the model page asking
  you to read the whole drive a third time. It now records what the transfer
  proved, file by file, and the copy reads *fully verified* the moment the
  download finishes. Nothing is assumed: if the transfer cannot account for
  every published file — because it adopted files that were already on the drive,
  say — the copy stays unverified, the row says why, and *Verify all files* is
  still there.
- One copy is one row. A download used to register the folder it created as a
  storage location of its own, inside the drive you had already added, so the
  same model appeared twice under *Copies on this device* and *Verify all files*
  started two passes over the same drive at once. A folder inside a folder
  Minirun already watches is not a second place; where two of your folders do
  cover the same model, it is listed once.
- The model page puts the copies on your drive above the facts about the model.
  *Readiness* is no longer a column of sentences taking a third of the width: it
  is two lines at the top of *About* — *Chat · supported here*, *Files · fully
  verified* — and where a copy is, whether it has been checked, and the controls
  that check or remove it come straight after the transfer.
- Settings, Storage and About have been redesigned to match the model page.
  Sections are headings and hairlines instead of bordered cards under
  ALL-CAPS labels; a preference is its name, the sentence that says what it
  does, and its control on the same line. Storage now reads a folder the way
  the rest of the app reads a model: a drive glyph, the folder's name, its
  path, and one coloured dot with one sentence — *Mounted · assessed 3 days
  ago*, or *Not connected. Plug the drive in, or remove the folder.* What a
  drive measured is a list with the figures and their units lined up, and
  *Assess again* and *Forget this measurement* sit under it as links rather
  than as buttons crammed into a heading. Adding a folder is the one filled
  button on the page, and the first-run panel no longer wears the amber
  border the app uses for real problems. About states its version and its
  build as two figures, its three links as links, and credits the publisher
  artwork it ships.
- The models list, Downloads and Download details now read like the model page.
  A model is a row: the publisher's mark, its name, one line saying what it is,
  and on the right a coloured dot with one sentence — *Ready on K3NVME*,
  *Downloading · 47.6 GB of 517 GB*, *ARCHIVE is not connected*, *Not on this
  device* — over its size. The rounded badges are gone; so is the guesswork
  they invited, because a row now says the same thing the model page says, in
  fewer words. Downloads gives each transfer the same block the model page
  does — how much has landed out of how much, the rate, the file, where it is
  going — and a stopped one can be forgotten from there too, which until now
  only the model page and Download details offered. Download details states
  whether the index and the repository tree agree as a sentence with a green
  dot, and explains the digest trap in plain type instead of in code type.
- The model page has been redesigned. It reads like a product page now: the
  publisher's mark, the model's name, one line saying what it is, and one
  sentence with a coloured dot saying exactly where it stands — downloading to
  a named drive with the time left, ready with every file matching its
  published digest, or waiting for a drive you have unplugged. A running
  transfer is the one thing with weight on the screen: how much has landed out
  of how much, the rate, the file, the time left, and where it is going. The
  rest are facts in a quiet list. Warnings appear only when there is something
  to do about them, and say what to do. A model whose copy is on a drive in a
  drawer no longer offers to download it again.
- A transfer that stopped can be forgotten. A cancelled or failed attempt used
  to stay in the transfer list forever, even after you deleted what it left
  yourself — now *Forget* removes the record, from the row in *Download
  details* or from the card on the model page. It deletes nothing: the
  confirmation names the destination and says the files on the drive are not
  touched. A transfer that is still running, paused or verifying cannot be
  forgotten, and does not offer to be.
- Download details has a back button on the Mac; the transfer badge says
  "downloading" like the transfer list does; the per-file line says how far
  the current file has come instead of "resuming at offset"; and the "start
  another copy" note no longer appears under a running transfer.
- A download no longer empties a repository into the folder you picked. The
  folder you choose is the parent, and Minirun makes one folder inside it named
  after the model — `Kimi-K3-minirun`, `DeepSeek-V4.1-Flash-minirun` — so
  choosing a whole drive puts the model in its own folder instead of scattering
  `index.json`, `LICENSE` and `layer00/` across the drive's root. The sheet
  shows the resulting path before you start, an existing folder from the same
  model is picked up and continued, and a folder that holds anything else is
  refused by name rather than written into.
- A cancelled or interrupted transfer says what is on the drive now, not what
  it remembered: the files it left are counted again when the model screen
  opens, so deleting them yourself turns *Continue with kept files…* into
  *Download again…* instead of an offer to resume from bytes that are gone. A
  drive that is unplugged says so rather than reading as empty, and a stopped
  transfer no longer shows a speed or an ETA.
- A transfer's own details are reached from the transfer itself instead of from
  a lone button floating beneath it.
- DeepSeek V4.1 Flash is in the model list: 517 GB, downloadable and
  verifiable, with no chat support yet — the app says so rather than
  offering a run it cannot do.

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
