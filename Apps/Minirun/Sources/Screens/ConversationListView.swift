import SwiftUI
import MinirunKit

/// One conversation, as a row: what it is about, what answers it, and when it
/// last moved.
///
/// The model badge is not decoration — a conversation states its model, and two
/// chats in the same list can be running different ones at different budgets.
struct ConversationRow: View {
    let conversation: Conversation
    let modelName: String
    var isRunning = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(conversation.title)
                .font(MRType.body)
                .foregroundStyle(MRColor.primary)
                .lineLimit(1)
            HStack(spacing: MRSpace.s2) {
                Text(modelName)
                    .font(Self.subtitleFont)
                    .foregroundStyle(MRColor.secondary)
                    .lineLimit(1)
                Text(MRFormat.relative(conversation.updatedAt))
                    .font(Self.subtitleFont)
                    .foregroundStyle(MRColor.tertiary)
                if isRunning {
                    Circle()
                        .fill(MRColor.streamDet)
                        .frame(width: 6, height: 6)
                        .accessibilityHidden(true)
                }
            }
        }
        .padding(.vertical, 2)
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(conversation.title)
        .accessibilityValue(
            "\(modelName), updated \(MRFormat.relative(conversation.updatedAt))"
                + (isRunning ? ", a turn is running" : ""))
    }

    /// A model name and "just now" are prose, not telemetry. The Mac sidebar
    /// keeps its dense mono line; on the phone the same line in mono read as a
    /// code listing under every chat title.
    private static var subtitleFont: Font {
        #if os(iOS)
            MRType.caption
        #else
            MRType.micro
        #endif
    }
}

/// Rename, as a sheet with one field. Small on purpose: a title is the only
/// thing here a person edits by hand.
struct RenameConversationSheet: View {
    let conversationID: UUID
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var title = ""

    var body: some View {
        NavigationStack {
            Form {
                TextField("Title", text: $title)
                    .accessibilityLabel("conversation title")
                Text(
                    "A renamed conversation keeps its name: the title stops being derived from "
                        + "the first message."
                )
                .font(MRType.micro)
                .foregroundStyle(MRColor.tertiary)
            }
            .formStyle(.grouped)
            .navigationTitle("Rename")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Rename") {
                        model.rename(conversationID, to: title)
                        dismiss()
                    }
                    .disabled(title.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
        }
        #if os(macOS)
            .frame(minWidth: 360, minHeight: 200)
        #else
            .presentationDetents([.medium])
            .presentationDragIndicator(.visible)
        #endif
        .onAppear { title = model.conversation(conversationID)?.title ?? "" }
    }
}

/// Immutable presentation state for a destructive conversation action. Taking
/// the title and run state at the moment the action is requested keeps the
/// confirmation truthful even if the list reorders while the dialog is open.
struct ConversationDeletionRequest: Identifiable, Equatable {
    let id: UUID
    let title: String
    let stopsActiveResponse: Bool

    init(_ conversation: Conversation, isRunning: Bool) {
        id = conversation.id
        title = conversation.title
        stopsActiveResponse = isRunning
    }

    var message: String {
        var text = "Conversation: \u{201c}\(title)\u{201d}\n\n"
            + "This permanently deletes the conversation and cannot be undone."
        if stopsActiveResponse {
            text += " The current response will be stopped safely before deletion."
        }
        return text + " Model files and other conversations are not affected."
    }
}

enum EmptyChatsAction: Equatable, Sendable {
    case newChat
    case retryCatalog
    case storage
    case models
}

struct EmptyChatsButton: Equatable, Sendable {
    let title: String
    let action: EmptyChatsAction
}

/// Product hierarchy for the empty Chats destination. This is derived from
/// the same readiness authority that gates creation, so the prominent action
/// can never invite a state transition the model will refuse.
struct EmptyChatsPresentation: Equatable, Sendable {
    let systemImage: String
    let title: String
    let detail: String
    let showsProgress: Bool
    let primary: EmptyChatsButton?
    let secondary: EmptyChatsButton?

    init(readiness: ChatReadiness, modelName: (ModelID) -> String) {
        switch readiness {
        case .ready(let model):
            systemImage = "bubble.left.and.bubble.right"
            title = "Start a conversation"
            detail = "\(modelName(model)) is verified and ready."
            showsProgress = false
            primary = EmptyChatsButton(title: "New Chat", action: .newChat)
            secondary = nil

        case .loadingCatalog:
            systemImage = "arrow.triangle.2.circlepath"
            title = "Loading models"
            detail = "Checking the current model catalog before starting a chat."
            showsProgress = true
            primary = nil
            secondary = nil

        case .catalogUnavailable:
            systemImage = "wifi.exclamationmark"
            title = "Models could not be loaded"
            detail = "Check the connection and try again. Your local files are unchanged."
            showsProgress = false
            primary = EmptyChatsButton(title: "Try Again", action: .retryCatalog)
            secondary = EmptyChatsButton(title: "Open Storage", action: .storage)

        case .needsStorage:
            systemImage = "folder.badge.plus"
            title = "Choose a model folder"
            detail = "Give Minirun access to a folder where it can find or keep local models."
            showsProgress = false
            primary = EmptyChatsButton(title: "Choose Storage", action: .storage)
            secondary = EmptyChatsButton(title: "Find Models", action: .models)

        case .scanningStorage:
            systemImage = "externaldrive"
            title = "Scanning your folders"
            detail = "Looking for complete Minirun models in the folders you chose."
            showsProgress = true
            primary = EmptyChatsButton(title: "Open Storage", action: .storage)
            secondary = nil

        case .needsVerification(let model):
            systemImage = "checkmark.shield"
            title = "Verify \(modelName(model))"
            detail = "A local copy is present. Verify all published files before chatting."
            showsProgress = false
            primary = EmptyChatsButton(title: "Open Models", action: .models)
            secondary = EmptyChatsButton(title: "Storage", action: .storage)

        case .needsModel:
            systemImage = "square.stack.3d.up"
            title = "Find a model"
            detail = "Browse Minirun models, then choose where to keep a local copy."
            showsProgress = false
            primary = EmptyChatsButton(title: "Find Models", action: .models)
            secondary = EmptyChatsButton(title: "Storage", action: .storage)
        }
    }
}

#if os(iOS)
    /// The phone's Chats tab: the same rows as the Mac sidebar, in the idiom the
    /// platform expects — swipe to delete, tap to open.
    struct ConversationListView: View {
        let openSettings: (AppModel.SettingsSection) -> Void
        @Environment(AppModel.self) private var model
        @State private var renaming: UUID?
        @State private var pendingDeletion: ConversationDeletionRequest?

        var body: some View {
            Group {
                if model.conversations.isEmpty {
                    emptyState
                } else {
                    conversationList
                }
            }
            .mrWorkspaceSurface()
            .navigationTitle("Chats")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        _ = model.newReadyConversation()
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                    .disabled(!model.canCreateReadyConversation)
                    .accessibilityLabel("new chat")
                }
            }
            .sheet(item: $renaming) { id in
                RenameConversationSheet(conversationID: id)
            }
            .alert(
                "Delete this conversation?",
                isPresented: deletionConfirmationPresented
            ) {
                if let request = pendingDeletion {
                    Button("Delete conversation", role: .destructive) {
                        pendingDeletion = nil
                        model.delete(request.id)
                    }
                }
                Button("Cancel", role: .cancel) { pendingDeletion = nil }
            } message: {
                if let request = pendingDeletion { Text(request.message) }
            }
        }

        private var conversationList: some View {
            List {
                ForEach(model.conversations) { conversation in
                    Button {
                        model.openConversation(conversation.id)
                    } label: {
                        HStack(spacing: MRSpace.s2) {
                            ConversationRow(
                                conversation: conversation,
                                modelName: modelName(conversation),
                                isRunning: model.isRunning(conversation.id))
                            Image(systemName: "chevron.right")
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(MRColor.tertiary)
                                .accessibilityHidden(true)
                        }
                        // Without this the row only reacted where the glyphs
                        // were: a plain button's hit area is its content, and
                        // a stack of `Text` has no fill to hit. A tap on the
                        // empty trailing half of the row did nothing.
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    // The list and the page are one surface. The default row
                    // background is pure black in dark mode and read as a band
                    // laid over the app's navy.
                    .listRowBackground(MRColor.abyss)
                    .listRowSeparatorTint(MRColor.hairline)
                    .swipeActions(edge: .trailing) {
                        // Both tints are explicit. The app tints its whole
                        // scene, and the destructive role alone was not enough
                        // to keep Delete from drawing in that same blue beside
                        // Rename — two actions, one colour, no hierarchy.
                        Button("Delete", role: .destructive) {
                            requestDeletion(conversation)
                        }
                        .tint(MRColor.refuse)
                        Button("Rename") { renaming = conversation.id }
                            .tint(MRColor.tierPinned)
                    }
                    .contextMenu {
                        Button("Rename") { renaming = conversation.id }
                        Button("Delete\u{2026}", role: .destructive) {
                            requestDeletion(conversation)
                        }
                    }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
        }

        private var emptyState: some View {
            let presentation = EmptyChatsPresentation(
                readiness: model.chatReadiness,
                modelName: { id in
                    model.entry(id)?.descriptor.displayName ?? id.rawValue
                })
            return ContentUnavailableView {
                VStack(spacing: MRSpace.s3) {
                    if presentation.showsProgress {
                        ProgressView()
                            .controlSize(.large)
                            .accessibilityLabel(presentation.title)
                    } else {
                        Image(systemName: presentation.systemImage)
                            .font(.system(size: 34, weight: .light))
                            .foregroundStyle(MRColor.secondary)
                            .accessibilityHidden(true)
                    }
                    Text(presentation.title)
                        .font(MRType.title)
                        .foregroundStyle(MRColor.primary)
                }
            } description: {
                Text(presentation.detail)
                    .font(MRType.body)
                    .foregroundStyle(MRColor.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            } actions: {
                VStack(spacing: MRSpace.s2) {
                    if let primary = presentation.primary {
                        emptyActionButton(primary, prominent: true)
                    }
                    if let secondary = presentation.secondary {
                        emptyActionButton(secondary, prominent: false)
                    }
                }
                .frame(maxWidth: 280)
            }
        }

        @ViewBuilder private func emptyActionButton(
            _ item: EmptyChatsButton,
            prominent: Bool
        ) -> some View {
            if prominent {
                Button {
                    perform(item.action)
                } label: {
                    Text(item.title)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .frame(minHeight: MRAccessibility.minimumIOSTouchTarget)
            } else {
                Button {
                    perform(item.action)
                } label: {
                    Text(item.title)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .frame(minHeight: MRAccessibility.minimumIOSTouchTarget)
            }
        }

        private func perform(_ action: EmptyChatsAction) {
            switch action {
            case .newChat:
                _ = model.newReadyConversation()
            case .retryCatalog:
                Task { await model.refreshPublishedModels() }
            case .storage:
                openSettings(.storage)
            case .models:
                openSettings(.models)
            }
        }

        private var deletionConfirmationPresented: Binding<Bool> {
            Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } })
        }

        private func requestDeletion(_ conversation: Conversation) {
            pendingDeletion = ConversationDeletionRequest(
                conversation, isRunning: model.isRunning(conversation.id))
        }

        private func modelName(_ conversation: Conversation) -> String {
            model.entry(conversation.settings.model)?.descriptor.displayName
                ?? conversation.settings.model.rawValue
        }
    }
#endif

/// `UUID` as a sheet item, so `sheet(item:)` can carry which conversation is
/// being renamed without a second piece of state to keep in step.
extension UUID: @retroactive Identifiable {
    public var id: UUID { self }
}
