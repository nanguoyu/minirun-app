import XCTest

@testable import MinirunApp

/// Every control on a product screen has a name a screen reader can say.
///
/// ## Why this reads the declarations instead of walking the live tree
///
/// The obvious test is to host each screen and walk its accessibility tree.
/// SwiftUI does not build one until an assistive client attaches: a hosted
/// `NSHostingView` answers `accessibilityChildren()` with zero children, and
/// after a self-directed `AXUIElementCreateApplication(getpid())` query it
/// answers with container elements only — scroll areas, the outline, the
/// composer's text field — never the buttons inside them. Self-inspection
/// through `AXUIElement` is degenerate in a hosted test process: every window
/// it returns answers `AXRole` with `AXApplication`, and walking it reaches the
/// menu bar and stops. A real tree needs a UI test driving a launched app,
/// which this suite is not.
///
/// So the audit reads what the screens declare, which is the thing a developer
/// actually gets wrong: an icon-only button with no `accessibilityLabel`. It
/// resolves names through the app's own components too, because a row that
/// names itself names the button wrapped around it.
final class AccessibilityNamesTests: XCTestCase {

    // MARK: The audit over the real screens

    func testEveryControlOnTheProductScreensNamesItself() throws {
        let audit = try ControlNameAudit.overProductSources()
        let unnamed = audit.unnamed
        XCTAssertTrue(
            unnamed.isEmpty,
            "controls with no name a screen reader can say:\n"
                + unnamed.map(\.description).joined(separator: "\n"))
    }

    /// The audit is worth nothing if it inspects three files and a stub. These
    /// are the screens the report named, and each has to be carrying controls.
    func testTheAuditCoversTheRootConversationModelSettingsAndDownloadScreens() throws {
        let audit = try ControlNameAudit.overProductSources()
        for screen in [
            "RootView.swift", "ConversationView.swift", "ConversationListView.swift",
            "ConversationSettingsView.swift", "ModelCatalogView.swift", "ModelDetailView.swift",
            "SettingsView.swift", "StorageSettingsView.swift", "DownloadsView.swift",
            "DownloadDetailView.swift", "DestinationPicker.swift", "InstrumentPanelView.swift",
            "MemoryDialView.swift", "AboutView.swift",
        ] {
            XCTAssertGreaterThan(
                audit.controlCount(inFileNamed: screen), 0,
                "\(screen) was not audited, so its controls are unguarded")
        }
        XCTAssertGreaterThan(
            audit.controls.count, 120,
            "the audit should be finding the app's whole control surface")
    }

    // MARK: The audit's own guard

    /// An icon-only button with no label is exactly the defect this exists to
    /// catch, and a test that cannot fail is not a test.
    func testTheAuditFailsAnIconOnlyButtonWithNoLabel() {
        let source = """
            struct Sidebar: View {
                var body: some View {
                    Button {
                        newChat()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                    .buttonStyle(.plain)
                    .help("New chat")
                }
            }
            """

        let unnamed = ControlNameAudit.unnamedControls(in: source, fileName: "Sidebar.swift")
        XCTAssertEqual(unnamed.count, 1)
        XCTAssertEqual(unnamed.first?.line, 3)
        XCTAssertTrue(unnamed.first?.description.contains("Sidebar.swift:3") ?? false)
    }

    /// `help` is a pointer tooltip, not a name: the phone has no pointer, and
    /// VoiceOver does not read it in place of a missing label.
    func testAHelpTooltipDoesNotCountAsAName() {
        let source = """
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
            }
            .help("Send")
            """

        XCTAssertEqual(
            ControlNameAudit.unnamedControls(in: source, fileName: "Composer.swift").count, 1)
    }

    func testAnIconOnlyButtonWithALabelPasses() {
        let source = """
            Button {
                newChat()
            } label: {
                Image(systemName: "square.and.pencil")
            }
            .buttonStyle(.plain)
            .accessibilityLabel("new chat")
            """

        XCTAssertTrue(
            ControlNameAudit.unnamedControls(in: source, fileName: "Sidebar.swift").isEmpty)
    }

    /// A button whose title is its first argument already exposes it, whether
    /// the title is a literal or a value.
    func testATitledButtonNeedsNoSecondName() {
        let source = """
            Button("New chat") { newChat() }
            Button(actionTitle, action: action)
            Button(ForgetTransferConfirmation.actionTitle, role: .destructive) { forget() }
            Picker("Sort", selection: $sort) { Text("Name").tag(0) }.labelsHidden()
            Toggle("Check automatically", isOn: $automatic).labelsHidden()
            """

        XCTAssertTrue(
            ControlNameAudit.unnamedControls(in: source, fileName: "Screens.swift").isEmpty)
    }

    /// A control with no title argument and no text in its label is named only
    /// if something in that label names itself. A component that does is a
    /// name; one that does not is the same nameless button one level down.
    func testAComposedLabelIsNamedOnlyWhenItsComponentNamesItself() {
        let source = """
            Button { open(chat) } label: {
                ConversationRow(conversation: chat)
            }
            .buttonStyle(.plain)
            Button { select(volume) } label: {
                SilentGlyph(volume: volume)
            }
            """

        let unnamed = ControlNameAudit.unnamedControls(
            in: source, fileName: "Lists.swift",
            componentsThatNameThemselves: ["ConversationRow"])
        XCTAssertEqual(unnamed.count, 1)
        XCTAssertTrue(unnamed.first?.declaration.contains("SilentGlyph") ?? false)
    }

    /// A decorative control that is deliberately outside the accessibility tree
    /// is not an unnamed one.
    func testAnExplicitlyHiddenControlIsNotAnOffender() {
        let source = """
            Button(action: nudge) { Image(systemName: "chevron.right") }
                .accessibilityHidden(true)
            """

        XCTAssertTrue(
            ControlNameAudit.unnamedControls(in: source, fileName: "Decorative.swift").isEmpty)
    }

    // MARK: The components the audit resolves names through

    /// The index is what keeps the audit from demanding a second label on a
    /// button whose row already speaks. It has to actually find the app's
    /// components, and it has to answer honestly for one that says nothing.
    func testTheComponentIndexReadsWhetherAComponentNamesItself() {
        let source = """
            struct ConversationRow: View {
                var body: some View {
                    Text(conversation.title)
                }
            }

            struct SilentGlyph: View {
                var body: some View {
                    Circle().fill(MRColor.hairline)
                }
            }
            """

        let index = ControlNameAudit.componentIndex(in: [source])
        XCTAssertEqual(index["ConversationRow"], true)
        XCTAssertEqual(index["SilentGlyph"], false)
    }

    func testTheProductComponentIndexKnowsTheRowsButtonsWrapAround() throws {
        let index = try ControlNameAudit.productComponentIndex()
        XCTAssertEqual(index["ConversationRow"], true)
        XCTAssertEqual(index["VolumeRow"], true)
    }
}

/// A small Swift-source reader: it finds control declarations, the label they
/// draw and the modifiers applied to them, and answers whether a name is
/// reachable from any of the three.
enum ControlNameAudit {

    struct Control: CustomStringConvertible {
        let fileName: String
        let line: Int
        let declaration: String
        let isNamed: Bool

        var description: String {
            "\(fileName):\(line)  \(declaration)"
        }
    }

    struct Result {
        let controls: [Control]

        var unnamed: [Control] { controls.filter { !$0.isNamed } }

        func controlCount(inFileNamed name: String) -> Int {
            controls.filter { $0.fileName == name }.count
        }
    }

    /// Control types whose first argument, when it is not one of the labelled
    /// forms below, is the title they display.
    private static let controlKinds = ["Button", "Menu", "Picker", "Toggle"]
    /// First-argument labels that mean "no title was given here".
    private static let untitledFirstArguments = ["action:", "isOn:", "selection:", "value:"]

    static func productSourceDirectory() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources")
    }

    static func productSources() throws -> [(name: String, text: String)] {
        let root = productSourceDirectory()
        let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: nil)
        var sources: [(String, String)] = []
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            sources.append((url.lastPathComponent, try String(contentsOf: url, encoding: .utf8)))
        }
        return sources.sorted { $0.0 < $1.0 }
    }

    static func productComponentIndex() throws -> [String: Bool] {
        componentIndex(in: try productSources().map(\.text))
    }

    static func overProductSources() throws -> Result {
        let sources = try productSources()
        let index = componentIndex(in: sources.map(\.text))
        let naming = Set(index.filter(\.value).map(\.key))
        let found = sources.flatMap { source in
            controls(
                in: source.text, fileName: source.name, componentsThatNameThemselves: naming)
        }
        return Result(controls: found)
    }

    static func unnamedControls(
        in source: String, fileName: String,
        componentsThatNameThemselves: Set<String> = []
    ) -> [Control] {
        controls(
            in: source, fileName: fileName,
            componentsThatNameThemselves: componentsThatNameThemselves
        ).filter { !$0.isNamed }
    }

    // MARK: Components

    /// Which `struct …: View` declarations put a name into the accessibility
    /// tree — by drawing text, or by declaring a label of their own, or by
    /// composing a component that does. The last clause is why this is a
    /// fixed point rather than a single pass.
    static func componentIndex(in sources: [String]) -> [String: Bool] {
        var bodies: [String: String] = [:]
        for source in sources {
            let characters = Array(blanked(source))
            var index = 0
            while let found = next("struct", from: index, in: characters) {
                index = found + "struct".count
                guard let name = identifier(after: index, in: characters) else { continue }
                guard let brace = characters[index...].firstIndex(of: "{"),
                    let colon = characters[index..<brace].firstIndex(of: ":"),
                    String(characters[colon..<brace]).contains("View")
                else { continue }
                let end = matchBracket(characters, at: brace)
                bodies[name.text] = String(characters[brace..<end])
            }
        }

        var named: [String: Bool] = bodies.mapValues { body in
            body.contains("Text(") || body.contains("Label(")
                || body.contains(".accessibilityLabel(")
        }
        for _ in 0..<4 {
            var changed = false
            for (name, body) in bodies where named[name] == false {
                for (other, isNamed) in named
                where isNamed && other != name && body.contains("\(other)(") {
                    named[name] = true
                    changed = true
                    break
                }
            }
            if !changed { break }
        }
        return named
    }

    // MARK: Controls

    static func controls(
        in source: String, fileName: String,
        componentsThatNameThemselves: Set<String>
    ) -> [Control] {
        let blankedSource = blanked(source)
        let characters = Array(blankedSource)
        var found: [Control] = []
        for kind in controlKinds {
            var index = 0
            while let start = next(kind, from: index, in: characters) {
                index = start + kind.count
                guard isTokenBoundary(before: start, in: characters),
                    isTokenBoundary(after: index, in: characters)
                else { continue }
                let (labelEnd, chainEnd) = span(from: index, in: characters)
                let label = String(characters[start..<labelEnd])
                let chain = String(characters[labelEnd..<chainEnd])
                found.append(
                    Control(
                        fileName: fileName,
                        line: blankedSource.prefix(start).filter { $0 == "\n" }.count + 1,
                        declaration: condensed(label),
                        isNamed: isNamed(
                            kind: kind, label: label, chain: chain,
                            components: componentsThatNameThemselves)))
            }
        }
        return found.sorted { ($0.line, $0.declaration) < ($1.line, $1.declaration) }
    }

    private static func isNamed(
        kind: String, label: String, chain: String, components: Set<String>
    ) -> Bool {
        if chain.contains(".accessibilityLabel(") || chain.contains(".accessibilityHidden(true)") {
            return true
        }
        if label.contains("Text(") || label.contains("Label(") || label.contains("systemImage:") {
            return true
        }
        if hasTitleArgument(kind: kind, label: label) { return true }
        for component in components where label.contains("\(component)(") { return true }
        return false
    }

    /// `Button("New chat") { … }`, `Picker("Sort", selection: …)` and
    /// `Toggle(title, isOn: …)` all display their first argument. The labelled
    /// forms — `Button(action:)`, `Toggle(isOn:)` — do not.
    private static func hasTitleArgument(kind: String, label: String) -> Bool {
        guard let open = label.firstIndex(of: "(") else { return false }
        let arguments = label[label.index(after: open)...]
            .drop(while: { $0 == " " || $0 == "\n" })
        guard !arguments.hasPrefix(")") else { return false }
        for labelled in untitledFirstArguments where arguments.hasPrefix(labelled) {
            return false
        }
        // A trailing-closure-only form reaches the `{` before any `(`.
        if let brace = label.firstIndex(of: "{"), brace < open { return false }
        return true
    }

    // MARK: Source scanning

    /// The declaration itself, the label it draws, and the modifiers applied to
    /// it: from the control's name to the end of its trailing closures, then on
    /// through every `.modifier(…)` chained after them.
    private static func span(from start: Int, in characters: [Character]) -> (Int, Int) {
        var index = skipWhitespace(from: start, in: characters)
        if index < characters.count, characters[index] == "(" {
            index = matchBracket(characters, at: index)
        }
        while true {
            let next = skipWhitespace(from: index, in: characters)
            guard next < characters.count else { break }
            if characters[next] == "{" {
                index = matchBracket(characters, at: next)
                continue
            }
            if let argument = identifier(after: next, in: characters),
                ["label", "content", "actions", "title"].contains(argument.text)
            {
                let afterName = skipWhitespace(from: argument.end, in: characters)
                guard afterName < characters.count, characters[afterName] == ":" else { break }
                let afterColon = skipWhitespace(from: afterName + 1, in: characters)
                guard afterColon < characters.count, characters[afterColon] == "{" else { break }
                index = matchBracket(characters, at: afterColon)
                continue
            }
            break
        }
        let labelEnd = index
        while true {
            var next = skipWhitespace(from: index, in: characters)
            // A `#if os(iOS)` line inside a chain is not the end of the chain.
            while next < characters.count, characters[next] == "#" {
                next = skipWhitespace(from: endOfLine(from: next, in: characters), in: characters)
            }
            guard next < characters.count, characters[next] == "." ,
                let modifier = identifier(after: next + 1, in: characters)
            else { break }
            index = modifier.end
            while true {
                let argument = skipWhitespace(from: index, in: characters)
                guard argument < characters.count,
                    characters[argument] == "(" || characters[argument] == "{"
                else { break }
                index = matchBracket(characters, at: argument)
            }
        }
        return (labelEnd, index)
    }

    private static func matchBracket(_ characters: [Character], at start: Int) -> Int {
        let closing: [Character: Character] = ["(": ")", "{": "}", "[": "]"]
        var stack: [Character] = [closing[characters[start]] ?? ")"]
        var index = start + 1
        while index < characters.count, !stack.isEmpty {
            let character = characters[index]
            if let close = closing[character] {
                stack.append(close)
            } else if character == stack.last {
                stack.removeLast()
            }
            index += 1
        }
        return index
    }

    private static func skipWhitespace(from start: Int, in characters: [Character]) -> Int {
        var index = start
        while index < characters.count, characters[index].isWhitespace { index += 1 }
        return index
    }

    private static func endOfLine(from start: Int, in characters: [Character]) -> Int {
        var index = start
        while index < characters.count, characters[index] != "\n" { index += 1 }
        return index
    }

    private static func identifier(
        after start: Int, in characters: [Character]
    ) -> (text: String, end: Int)? {
        let begin = skipWhitespace(from: start, in: characters)
        var index = begin
        while index < characters.count,
            characters[index].isLetter || characters[index].isNumber || characters[index] == "_"
        {
            index += 1
        }
        guard index > begin else { return nil }
        return (String(characters[begin..<index]), index)
    }

    private static func next(_ needle: String, from start: Int, in characters: [Character]) -> Int?
    {
        let pattern = Array(needle)
        guard characters.count >= pattern.count else { return nil }
        var index = start
        while index <= characters.count - pattern.count {
            var offset = 0
            while offset < pattern.count, characters[index + offset] == pattern[offset] {
                offset += 1
            }
            if offset == pattern.count { return index }
            index += 1
        }
        return nil
    }

    private static func isTokenBoundary(before index: Int, in characters: [Character]) -> Bool {
        guard index > 0 else { return true }
        let previous = characters[index - 1]
        return !(previous.isLetter || previous.isNumber || previous == "_" || previous == ".")
    }

    private static func isTokenBoundary(after index: Int, in characters: [Character]) -> Bool {
        guard index < characters.count else { return true }
        let next = characters[index]
        return !(next.isLetter || next.isNumber || next == "_")
    }

    /// Comments and the contents of string literals are blanked, keeping every
    /// byte offset and newline, so a `Button` inside a comment or the word
    /// `Link` inside a string is not read as code.
    static func blanked(_ source: String) -> String {
        var output = ""
        let characters = Array(source)
        var index = 0
        func blank(_ count: Int) { output += String(repeating: " ", count: count) }
        while index < characters.count {
            let character = characters[index]
            if character == "/", index + 1 < characters.count, characters[index + 1] == "/" {
                let end = endOfLine(from: index, in: characters)
                blank(end - index)
                index = end
            } else if character == "/", index + 1 < characters.count, characters[index + 1] == "*" {
                var end = index + 2
                while end + 1 < characters.count,
                    !(characters[end] == "*" && characters[end + 1] == "/")
                {
                    end += 1
                }
                end = min(characters.count, end + 2)
                for position in index..<end {
                    output.append(characters[position] == "\n" ? "\n" : " ")
                }
                index = end
            } else if character == "\"", index + 2 < characters.count,
                characters[index + 1] == "\"", characters[index + 2] == "\""
            {
                var end = index + 3
                while end + 2 < characters.count,
                    !(characters[end] == "\"" && characters[end + 1] == "\""
                        && characters[end + 2] == "\"")
                {
                    end += 1
                }
                end = min(characters.count, end + 3)
                for position in index..<end {
                    output.append(characters[position] == "\n" ? "\n" : " ")
                }
                index = end
            } else if character == "\"" {
                output += "\""
                var end = index + 1
                while end < characters.count, characters[end] != "\"", characters[end] != "\n" {
                    if characters[end] == "\\" { end += 1 }
                    end += 1
                }
                blank(max(0, min(end, characters.count) - index - 1))
                if end < characters.count, characters[end] == "\"" {
                    output += "\""
                    index = end + 1
                } else {
                    index = end
                }
            } else {
                output.append(character)
                index += 1
            }
        }
        return output
    }

    private static func condensed(_ declaration: String) -> String {
        let flattened = declaration.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return flattened.count > 96 ? String(flattened.prefix(96)) + "…" : flattened
    }
}
