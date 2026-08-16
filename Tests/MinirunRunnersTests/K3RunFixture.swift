import Darwin
import Foundation
import MinirunKit
import MinirunRunners
import XCTest

@testable import ModelAdapters

/// The artifact these tests decode, and the request that decodes it.
///
/// One ``SyntheticArtifact`` — a complete K3 artifact at fixture size, with the
/// dense prefix, KDA layers, an MLA layer, routed experts and the three global
/// tables — written once for the whole class. Nothing here is gated on the
/// 1.56 TB drive, which is the point: a lifecycle claim that could only be
/// checked with the flagship attached is a claim nobody checks.
enum K3RunFixture {

    /// Comfortably above anything this process will reach, so `budgetRespected`
    /// is a statement about the run rather than about the test host, and above
    /// the runner's 500 MB working-set reserve, so the read-ahead schedule has
    /// a window to fit pairs into.
    static let declaredBudgetBytes: UInt64 = 8 << 30
    /// The floor and ceiling a fixture has to state for itself: it has no run
    /// record, so it cannot borrow the flagship's 5.8 GB and one-token numbers.
    static let minimumBudgetBytes: UInt64 = 1 << 30
    static let maximumNewTokens = 64

    static func runner(maximumNewTokens: Int = maximumNewTokens) -> K3DecodeRunner {
        K3DecodeRunner(
            scale: .stated(
                minimumBudgetBytes: minimumBudgetBytes, maximumNewTokens: maximumNewTokens))
    }

    static func request(
        root: URL, workingDirectory: URL, newTokens: Int,
        budgetBytes: UInt64 = declaredBudgetBytes, prompt: Prompt = .tokenIDs([3, 17, 200]),
        knobs: RunKnobs = RunKnobs(), scope: StorageScope? = nil,
        runtimeAuthority: ArtifactRuntimeAuthority? = nil, pinPlan: PinPlan? = nil
    ) -> RunRequest {
        RunRequest(
            model: .kimiK3,
            artifact: ArtifactReference(
                root: root, scope: scope, runtimeAuthority: runtimeAuthority),
            prompt: prompt,
            memoryBudgetBytes: budgetBytes,
            pinPlan: pinPlan,
            maximumNewTokens: newTokens,
            knobs: knobs,
            workingDirectory: workingDirectory)
    }

    /// Read-ahead on, reserve off. Depth 1 is what makes a cancellation test
    /// interesting: staged bytes are bytes something still owns, and a run that
    /// was stopped mid-pass has to have given them back.
    static func stagedKnobs() -> RunKnobs {
        var knobs = RunKnobs()
        knobs.deterministicReadAheadLayers = 1
        knobs.readAheadReserveBytes = 0
        return knobs
    }

    /// Descriptors this process holds open.
    ///
    /// The independent half of "teardown freed everything": the run's own
    /// teardown record says what it released, and this says whether the process
    /// believes it. One container reader is one descriptor, so a leak at
    /// flagship scale is 372 of them.
    static func openDescriptors() -> Int {
        var limit = rlimit()
        guard getrlimit(RLIMIT_NOFILE, &limit) == 0 else { return -1 }
        let ceiling = min(Int(limit.rlim_cur), 4096)
        var count = 0
        for descriptor in 0..<ceiling where fcntl(Int32(descriptor), F_GETFD) != -1 {
            count += 1
        }
        return count
    }
}

/// The terminal event of a run, and everything it carried.
struct DrainedRun {
    var events: [RunEvent] = []
    var tokens: [TokenEvent] = []
    var telemetry: [RunTelemetry] = []
    var phases: [RunPhase] = []
    var phaseSummaries: [RunPhaseSummary] = []
    var logs: [String] = []
    var acceptance: RunAcceptance?
    var finished: RunSummary?
    var cancelled: RunSummary?
    var failure: Error?

    var terminal: RunSummary? { finished ?? cancelled }

    /// The run's own result JSON, as a dictionary. Parsed rather than decoded
    /// through a mirror type: a mirror would have to be kept in step with the
    /// payload, and a test that silently stopped checking a field it no longer
    /// knew about would be worse than no test.
    func payload() throws -> [String: Any] {
        let data = try XCTUnwrap(terminal?.resultJSON)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// Consume a session to its end. `onToken` runs inside the loop, which is
    /// where a Stop button lives too.
    static func drain(
        _ session: RunSession, onToken: @escaping (TokenEvent) -> Void = { _ in }
    ) async -> DrainedRun {
        var run = DrainedRun()
        do {
            for try await event in session.events {
                run.events.append(event)
                switch event {
                case .accepted(let acceptance): run.acceptance = acceptance
                case .phase(let phase): run.phases.append(phase)
                case .token(let token):
                    run.tokens.append(token)
                    onToken(token)
                case .telemetry(let telemetry): run.telemetry.append(telemetry)
                case .phaseSummary(let summary): run.phaseSummaries.append(summary)
                case .log(let line): run.logs.append(line)
                case .finished(let summary): run.finished = summary
                case .cancelled(let summary): run.cancelled = summary
                }
            }
        } catch {
            run.failure = error
        }
        return run
    }
}
