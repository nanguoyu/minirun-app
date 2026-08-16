# Third-party notices

This file records software and artwork distributed with, or resolved as a
dependency of, the Minirun source distribution. Model weights are not included
in this repository and carry their own licenses when obtained separately.

## mlx-swift

- Source: <https://github.com/ml-explore/mlx-swift>
- Pinned version: 0.31.4
- License: MIT

> Copyright © 2023 ml-explore
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the “Software”), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Sparkle

- Source: <https://github.com/sparkle-project/Sparkle>
- Pinned version: 2.9.5
- License: MIT

Sparkle is linked by the macOS app only, and is redistributed inside the macOS
product as `Sparkle.framework` together with the helper tools and XPC services
it bundles. The iOS app does not link it.

> Copyright (c) 2006-2013 Andy Matuschak.
> Copyright (c) 2009-2013 Elgato Systems GmbH.
> Copyright (c) 2011-2014 Kornel Lesiński.
> Copyright (c) 2015-2017 Mayur Pawashe.
> Copyright (c) 2014 C.W. Betts.
> Copyright (c) 2014 Petroules Corporation.
> Copyright (c) 2014 Big Nerd Ranch.
> All rights reserved.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Sparkle also embeds Google Toolbox for Mac and bspatch/bsdiff, whose notices
travel inside the distributed framework's own `LICENSE` resources.

## Swift Numerics

- Source: <https://github.com/apple/swift-numerics>
- Resolved version: 1.1.1
- License: Apache License 2.0 with Runtime Library Exception

Copyright © 2019 Apple Inc. and the Swift project authors.

The full Apache License 2.0 appears in [LICENSE](LICENSE). Swift Numerics adds
this exception:

> As an exception, if you use this Software to compile your source code and
> portions of this Software are embedded into the binary product as a result,
> you may redistribute such product without providing attribution as would
> otherwise be required by Sections 4(a), 4(b) and 4(d) of the License.

## Lobe Icons

- Source: <https://github.com/lobehub/lobe-icons>
- Package: `@lobehub/icons-static-svg` 1.94.0
- License: MIT

> MIT License
>
> Copyright (c) 2023 LobeHub
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The bundled publisher SVGs are supplied by Lobe Icons. Publisher names and
logos remain trademarks of their respective owners and are used only for
identification; see [TRADEMARKS.md](TRADEMARKS.md).

## Apple platform frameworks

Minirun links system frameworks supplied by Apple, including SwiftUI,
Foundation, CryptoKit, AppKit or UIKit, and Darwin interfaces. Those frameworks
are not redistributed in this source repository and remain subject to Apple's
terms.
