# Contributing to Minirun

Thank you for helping improve Minirun. This repository accepts contributions
under the Apache License 2.0 and uses the Developer Certificate of Origin 1.1
(DCO) instead of a Contributor License Agreement.

## Sign off every commit

Every contribution must carry a `Signed-off-by` line certifying the DCO in
[DCO](DCO). Git can add it for you:

```sh
git commit -s
```

Use a name and email address that identify you as the contributor. The sign-off
is a legal certification, not an authorship decoration; do not add another
person's sign-off without their authorization. Contributors retain copyright
in their contributions and license them to the project under Apache-2.0.

## Keep changes reviewable

- Open an issue before a large architecture, model-format, or product change.
- Add correctness tests before or with optimized code.
- Add a reproducible benchmark and measurement metadata for changes to I/O,
  memory management, scheduling, or kernels.
- Keep model-specific logic out of the generic storage layer.
- Fail unsupported cases explicitly; never add a fallback that loads a complete
  model into memory.
- Do not commit model weights, credentials, captured conversations, private
  paths, signing identities, provisioning profiles, or restricted assets.

## Build and test

Minirun requires Apple silicon and Xcode. The primary local gates are:

```sh
xcodebuild test -scheme minirun-Package -destination 'platform=macOS' \
  -only-testing:MinirunKitTests

xcodebuild test -project Apps/Minirun/Minirun.xcodeproj \
  -scheme Minirun-macOS -destination 'platform=macOS' \
  CODE_SIGNING_ALLOWED=NO

xcodebuild build -project Apps/Minirun/Minirun.xcodeproj \
  -scheme Minirun-iOS -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO
```

The project does not require paid GitHub-hosted macOS CI. Maintainers record
the applicable local gates before merging or publishing a release.

## Public discussions

Issues, pull requests, commits, and DCO sign-offs are public and retained in
the project history. Report security vulnerabilities privately as described in
[SECURITY.md](SECURITY.md).
