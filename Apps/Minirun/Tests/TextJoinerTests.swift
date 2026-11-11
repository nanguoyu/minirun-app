import XCTest

/// No line of the product joins two pieces of text with a middle dot.
///
/// A dot between two facts leaves the reader to work out how they relate, and
/// the product had more than a hundred of them: status lines, figure rows,
/// precision strings, comments, the tests that pinned them. The rule is now a
/// sentence, a comma, or a separate line or element, and this scan is what
/// stops the dot coming back one string at a time. It reads the App's sources
/// and tests, its design record, the package modules the App's surfaces are
/// built from, and the public README templates, and it names every offending
/// file and line.
///
/// The look-alikes count, because they draw the same mark: the bullet, the
/// bullet operator, the dot operator and the katakana middle dot. The one use
/// of a bullet that is not a joiner is a list marker at the start of a line of
/// text — the Markdown renderer draws one before each item — so a bullet that
/// opens a string literal, or follows a written `\n`, is a marker and passes.
final class TextJoinerTests: XCTestCase {

    /// Spelled as scalars so that this file does not contain what it bans.
    static let banned: Set<UInt32> = [0x00B7, 0x2022, 0x2219, 0x22C5, 0x30FB]
    static let bullet: UInt32 = 0x2022

    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // Minirun
            .deletingLastPathComponent()  // Apps
            .deletingLastPathComponent()
            // A checkout under a symlinked directory (`/tmp` is one) is
            // enumerated by its real path, so both sides are standardized.
            .standardizedFileURL
    }

    /// Directories scanned whole, relative to the repository root, and the
    /// file-name prefixes each one leaves to its owner.
    static let scanned: [(directory: String, extensions: Set<String>, only: [String]?)] = [
        ("Apps/Minirun/Sources", ["swift"], nil),
        ("Apps/Minirun/Tests", ["swift"], nil),
        ("Sources/MinirunKit", ["swift"], nil),
        ("Sources/StorageCore", ["swift"], nil),
        ("Sources/iOSBench", ["swift"], nil),
        ("Sources/BenchScenarios", ["swift"], nil),
        ("Sources/MLXBridge", ["swift"], nil),
        ("Sources/ModelAdapters", ["swift"], nil),
        ("Sources/MinirunRunners", ["swift"], nil),
        ("Tests/BenchScenariosTests", ["swift"], nil),
        ("Tests/MLXBridgeTests", ["swift"], nil),
        ("Tests/MinirunKitTests", ["swift"], nil),
        ("Tests/StorageCoreTests", ["swift"], nil),
        ("Tests/iOSBenchTests", ["swift"], nil),
        ("Tests/ModelAdaptersTests", ["swift"], nil),
        ("Tests/MinirunRunnersTests", ["swift"], nil),
        ("Tools/public_release/templates", ["md", "yml"], nil),
    ]
    static let singleFiles = ["Apps/Minirun/DESIGN.md", "README.md"]
    /// The public repository's ruleset requires a status check by this job's
    /// exact name, so the name changes together with that ruleset or not at all.
    static let requiredCheckName = "Tools/public_release/templates/.github/workflows/ci.yml"

    static func files() -> [URL] {
        let root = repositoryRoot
        var found = singleFiles.map { root.appendingPathComponent($0) }
        for entry in scanned {
            let directory = root.appendingPathComponent(entry.directory)
            let enumerator = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: nil)
            while let url = enumerator?.nextObject() as? URL {
                let name = url.lastPathComponent
                guard entry.extensions.contains(url.pathExtension) else { continue }
                if let only = entry.only, !only.contains(where: { name.hasPrefix($0) }) {
                    continue
                }
                if url.path.hasSuffix(requiredCheckName) { continue }
                found.append(url)
            }
        }
        return found.sorted { $0.path < $1.path }
    }

    /// Every banned mark on the line that is not a list marker, as columns.
    static func offences(in line: Substring) -> [Int] {
        let scalars = Array(line.unicodeScalars)
        var columns: [Int] = []
        for (index, scalar) in scalars.enumerated() where banned.contains(scalar.value) {
            if scalar.value == bullet {
                let opensLiteral = index >= 1 && scalars[index - 1] == "\""
                let followsNewline =
                    index >= 2 && scalars[index - 2] == "\\" && scalars[index - 1] == "n"
                if opensLiteral || followsNewline { continue }
            }
            columns.append(index + 1)
        }
        // The same mark spelled for a URL or for HTML, as a badge or a page
        // would carry it. Assembled so that this line does not match itself.
        let text = String(line)
        for spelling in ["%C2" + "%B7", "&mid" + "dot;", "&#18" + "3;"] {
            if let range = text.range(of: spelling, options: .caseInsensitive) {
                columns.append(text.distance(from: text.startIndex, to: range.lowerBound) + 1)
            }
        }
        return columns
    }

    func testNoLineJoinsTextWithAMiddleDot() throws {
        let files = Self.files()
        let root = Self.repositoryRoot.path + "/"
        // The scan has to be reading the product, or a moved directory would
        // make it pass by reading nothing.
        for expected in [
            "Apps/Minirun/Sources/DesignSystem/Components/Instruments.swift",
            "Apps/Minirun/Sources/Screens/ModelDetailView.swift",
            "Sources/MinirunKit/Runner/RunnerFacade.swift",
            "Sources/MinirunRunners/K3DecodeRunner.swift",
            "Tools/public_release/templates/README.md",
            "Apps/Minirun/DESIGN.md",
        ] {
            XCTAssertTrue(
                files.contains { $0.standardizedFileURL.path == root + expected }, "the scan missed \(expected)")
        }
        XCTAssertGreaterThan(files.count, 150)

        var found: [String] = []
        for url in files {
            let text = try String(contentsOf: url, encoding: .utf8)
            for (number, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
                .enumerated()
            {
                for column in Self.offences(in: line) {
                    let path = url.path.replacingOccurrences(of: root, with: "")
                    found.append("\(path):\(number + 1):\(column)")
                }
            }
        }
        XCTAssertTrue(
            found.isEmpty,
            "say it as a sentence, use a comma, or give each fact its own line or element:\n"
                + found.joined(separator: "\n"))
    }

    func testTheScanFindsAJoinerAndPassesAListMarker() {
        func line(_ scalars: [UInt32]) -> Substring {
            Substring(String(String.UnicodeScalarView(scalars.compactMap(Unicode.Scalar.init))))
        }
        let space: UInt32 = 0x20
        let a: UInt32 = 0x61
        let b: UInt32 = 0x62
        let quote: UInt32 = 0x22
        let backslash: UInt32 = 0x5C
        let n: UInt32 = 0x6E
        for mark in Self.banned {
            XCTAssertEqual(
                Self.offences(in: line([a, space, mark, space, b])), [3],
                String(format: "U+%04X between two words is a joiner", mark))
        }
        XCTAssertEqual(Self.offences(in: line([quote, Self.bullet, space, a])), [])
        XCTAssertEqual(
            Self.offences(in: line([a, backslash, n, Self.bullet, space, b])), [])
        XCTAssertEqual(
            Self.offences(in: line([quote, 0x00B7, space, a])), [2],
            "only the bullet is ever a list marker")
        XCTAssertEqual(
            Self.offences(in: Substring("badge/platform-macOS%2015%2B%20%c2" + "%b7%20iOS")), [32])
    }
}
