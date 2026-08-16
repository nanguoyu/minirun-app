import MinirunKit
import MinirunRunners
import SwiftUI

/// Everything that decides how THIS conversation runs.
///
/// The rule that shapes this screen: if a control changes what a run does, it is
/// here and not in Settings. The app's global Settings may hold the value a
/// *new* conversation starts from; this screen says "this chat".
///
/// While a turn is running the whole screen is read-only. That is the same
/// promise the dial makes inside a run (§5.3, stated at start), and a settings
/// panel that could be edited mid-turn would quietly make it a lie.
///
/// **This is a panel, not a manual.** Every explanation that used to be printed
/// here as a paragraph is now an ⓘ; the visible text is a label, a value, or at
/// most one short caption per section. The review that forced the change put it
/// plainly: the operator could not work out how to use their own settings, for
/// the prose covering them.
struct ConversationSettingsView: View {
    let conversationID: UUID

    @Environment(AppModel.self) private var model

    private var conversation: Conversation? { model.conversation(conversationID) }
    private var isRunning: Bool { model.isRunning(conversationID) }

    var body: some View {
        ScrollView {
            if let conversation {
                VStack(alignment: .leading, spacing: MRSpace.s3) {
                    if isRunning { lockedBanner }
                    modelSection(conversation)
                    if let entry = model.entry(conversation.settings.model) {
                        dial(conversation, entry)
                    }
                    generation(conversation)
                }
                .padding(MRSpace.s3)
                .frame(maxWidth: 720, alignment: .leading)
            } else {
                EmptyStateView(
                    headline: "No conversation selected.",
                    message: "Pick a chat to see the settings its turns run under.")
                    .padding(MRSpace.s4)
            }
        }
        // Explicit, because it is not the default it looks like: dropped into
        // an inspector column the panel opened already scrolled past its own
        // Model section, which is the first thing anybody looks for.
        .defaultScrollAnchor(.top)
        .frame(maxWidth: .infinity)
        // No fill of our own. In the inspector column this painted a flat
        // near-white slab beside a sidebar made of system material, and the two
        // never matched in either appearance. The platform's panel background
        // is the one that belongs to a panel.
        .scrollContentBackground(.hidden)
        #if os(iOS)
            // Inline, because this is a sheet over the conversation rather than
            // a page in its own right, and a large title spent the first third
            // of a 6.1-inch screen restating the button that opened it.
            .navigationTitle("Chat settings")
            .navigationBarTitleDisplayMode(.inline)
        #endif
    }

    private var lockedBanner: some View {
        HStack(spacing: MRSpace.s2) {
            Image(systemName: "lock.fill").imageScale(.small)
            Text("Stop this turn to edit its settings.")
                .font(MRType.caption)
                .fixedSize(horizontal: false, vertical: true)
        }
        .foregroundStyle(MRColor.caution)
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard(MRColor.caution.opacity(0.12), stroke: MRColor.caution.opacity(0.4))
        .accessibilityElement(children: .combine)
    }

    // MARK: Model

    private func modelSection(_ conversation: Conversation) -> some View {
        let availability = model.modelAvailability(for: conversation.settings.model)
        let choices = model.modelPickerChoices(current: conversation.settings.model)
        let currentIsAvailable = choices.contains {
            $0.modelID == conversation.settings.model
        }
        return VStack(alignment: .leading, spacing: MRSpace.s2) {
            HStack(spacing: MRSpace.s2) {
                Text("Model").mrLabel(MRColor.secondary)
                Spacer(minLength: MRSpace.s2)
                if currentIsAvailable {
                    Picker(
                        "Model",
                        selection: Binding(
                            get: { conversation.settings.model },
                            set: { newValue in
                                model.selectModel(newValue, in: conversationID)
                            })
                    ) {
                        ForEach(choices) { choice in
                            if let id = choice.modelID {
                                Text(choice.displayName).tag(id)
                            }
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .disabled(isRunning)
                    .frame(maxWidth: 190)
                    .accessibilityLabel("model for this chat")
                } else {
                    Menu(choices.isEmpty ? "No verified models" : "Choose model") {
                        ForEach(choices) { choice in
                            if let id = choice.modelID {
                                Button(choice.displayName) {
                                    model.selectModel(id, in: conversationID)
                                }
                            }
                        }
                    }
                    .disabled(isRunning || choices.isEmpty)
                    .frame(maxWidth: 190)
                    .accessibilityLabel("model for this chat")
                }
            }

            HStack(alignment: .top, spacing: MRSpace.s2) {
                Image(
                    systemName: availability.isAvailable
                        ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                    .foregroundStyle(availability.isAvailable ? MRColor.ok : MRColor.caution)
                    .imageScale(.small)
                VStack(alignment: .leading, spacing: 2) {
                    Text(availability.isAvailable ? "Ready" : "Unavailable")
                        .font(MRType.caption)
                        .foregroundStyle(
                            availability.isAvailable ? MRColor.ok : MRColor.caution)
                    if let detail = availability.productDetail {
                        Text(detail)
                            .font(MRType.smallProse)
                            .foregroundStyle(MRColor.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .accessibilityElement(children: .combine)
        }
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard()
    }

    // MARK: Dial

    @ViewBuilder private func dial(_ conversation: Conversation, _ entry: CatalogEntry)
        -> some View
    {
        if let plan = model.budgetPlan(for: conversation) {
            MemoryDialView(
                plan: plan,
                entry: entry,
                role: .conversation,
                presentation: .compact,
                readOnly: isRunning,
                readAheadDepth: entry.id == .deepseekV4Flash
                    ? nil
                    : Binding(
                        get: { conversation.settings.deterministicReadAheadLayers },
                        set: { newValue in
                            model.update(conversationID) {
                                $0.settings.deterministicReadAheadLayers = newValue
                            }
                        }),
                onChange: { bytes in
                    guard !isRunning else { return }
                    model.setBudget(bytes, in: conversationID)
                })
        }
    }

    // MARK: Generation

    private func generation(_ conversation: Conversation) -> some View {
        let capabilities = model.capabilities(for: conversation.settings.model)
        let ceiling = capabilities?.maximumNewTokens ?? 1
        return VStack(alignment: .leading, spacing: MRSpace.s3) {
            Text("Generation").mrLabel(MRColor.secondary)
            // A response limit is a number somebody types, not a number they
            // click towards sixty-four times. The field takes digits only and
            // commits inside 1...ceiling; the ceiling is the runner's own and
            // is still applied when the model changes.
            HStack(spacing: MRSpace.s2) {
                Text("Response limit (tokens)")
                    .font(MRType.body)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: MRSpace.s2)
                IntegerField(
                    value: Binding(
                        get: { conversation.settings.maximumNewTokens },
                        set: { newValue in
                            model.update(conversationID) {
                                $0.settings.maximumNewTokens = newValue
                            }
                        }),
                    range: 1...max(1, ceiling),
                    accessibilityLabel: "response limit in tokens")
                    .disabled(isRunning || ceiling <= 1)
            }
            Text("1 to \(ceiling) tokens")
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)

            #if os(iOS)
                if conversation.settings.model == .kimiK3,
                    K3ProductMemoryBudget.currentPolicy.isExperimental
                {
                    Text("Experimental iPhone tier · up to \(ceiling) output tokens")
                        .font(MRType.smallProse)
                        .foregroundStyle(MRColor.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            #endif

            if hasAdditionalGenerationControls(capabilities) {
                Divider().overlay(MRColor.hairline)
                generationControls(conversation, capabilities)
            }
        }
        .padding(MRSpace.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .mrCard()
    }

    private func hasAdditionalGenerationControls(_ capabilities: RunnerCapabilities?) -> Bool {
        guard let capabilities else { return false }
        return capabilities.supportedKnobs.contains("expertReadAhead")
            || capabilities.supportedKnobs.contains("expertPoolSlots")
    }

    /// Only the knobs this runner honours are offered. A knob it cannot honour
    /// is refused BY NAME at validation, and offering one here would be building
    /// a control whose only outcome is an error.
    @ViewBuilder private func generationControls(
        _ conversation: Conversation, _ capabilities: RunnerCapabilities?
    ) -> some View {
        VStack(alignment: .leading, spacing: MRSpace.s3) {
            if let capabilities {
                if capabilities.supportedKnobs.contains("expertReadAhead") {
                    optionalStepper(
                        title: "Expert read-ahead",
                        help: "How many expert requests may be prepared ahead of use. Auto uses "
                            + "the model's recommended setting.",
                        range: 0...4,
                        value: Binding(
                            get: { conversation.settings.expertReadAhead },
                            set: { newValue in
                                model.update(conversationID) {
                                    $0.settings.expertReadAhead = newValue
                                }
                            }))
                }
                if capabilities.supportedKnobs.contains("expertPoolSlots") {
                    optionalStepper(
                        title: "Expert pool slots",
                        help: "The maximum number of expert weights kept ready for this chat. "
                            + "Auto uses the model's recommended setting.",
                        range: 1...64,
                        value: Binding(
                            get: { conversation.settings.expertPoolSlots },
                            set: { newValue in
                                model.update(conversationID) {
                                    $0.settings.expertPoolSlots = newValue
                                }
                            }))
                }
            }
        }
        .disabled(isRunning)
    }

    private func optionalStepper(
        title: String, help: String, range: ClosedRange<Int>, value: Binding<Int?>
    ) -> some View {
        HStack(spacing: MRSpace.s2) {
            Text(title)
                .font(MRType.body)
                .fixedSize(horizontal: false, vertical: true)
            HelpTip(help)
            Spacer(minLength: MRSpace.s2)
            Text(value.wrappedValue.map(String.init) ?? "Auto")
                .font(MRType.metric)
            Stepper(
                "",
                value: Binding(
                    get: { value.wrappedValue ?? range.lowerBound },
                    set: { value.wrappedValue = $0 }),
                in: range
            )
            .labelsHidden()
            if value.wrappedValue != nil {
                Button {
                    value.wrappedValue = nil
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                }
                .buttonStyle(.plain)
                .help("Use automatic value")
                .accessibilityLabel("use automatic value")
            }
        }
    }

}
