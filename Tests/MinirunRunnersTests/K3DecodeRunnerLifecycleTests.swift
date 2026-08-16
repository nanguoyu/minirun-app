import Foundation
import StorageCore
import XCTest

@testable import MinirunKit
@testable import ModelAdapters
@testable import MinirunRunners

/// Does a run start, say what it is doing, stop when it is told, and give back
/// everything it took?
///
/// Five claims, all on a fixture-sized artifact so they are checked on this Mac
/// rather than promised for a device:
///
/// 1. A decode reaches a token, and every telemetry sample it emits accounts for
///    every byte it read.
/// 2. A Stop pressed mid-decode ends the run as a **named** ending carrying what
///    it had measured, not as a silently short result.
/// 3. Teardown happens before the terminal event, and the process's descriptor
///    count says so independently of the run's own record.
/// 4. Cancelling the Task that drains the stream is the same door as
///    `RunSession.cancel`.
/// 5. A memory-dial plan reaches the real source, serves a second pass from its
///    pinned tier, and reports that every resident byte was freed before the
///    terminal event.
final class K3DecodeRunnerLifecycleTests: XCTestCase {

    // ~30 MB of expert containers, so it is written once for the class.
    private static var shared: SyntheticArtifact.Bundle?
    private static var sharedRoot: URL?

    private var bundle: SyntheticArtifact.Bundle!
    private var working: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        if Self.shared == nil {
            let root = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent(
                    "minirun-runner-\(ProcessInfo.processInfo.processIdentifier)")
            try? FileManager.default.removeItem(at: root)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            Self.shared = try SyntheticArtifact.write(root: root)
            Self.sharedRoot = root
        }
        bundle = Self.shared
        working = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-runner-work-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: working, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let working { try? FileManager.default.removeItem(at: working) }
        try super.tearDownWithError()
    }

    override class func tearDown() {
        if let root = sharedRoot { try? FileManager.default.removeItem(at: root) }
        shared = nil
        sharedRoot = nil
        super.tearDown()
    }

    // MARK: - A run that finishes

    func testProductCensusComesFromTheOpenedManifestTable() throws {
        let census = try K3DecodeRunner.inspectArtifactCensus(
            ArtifactReference(root: bundle.root), workingDirectory: working)
        let metadata = try ArtifactWeightSource.inspectMetadataCensus(
            translated: bundle.translated.path, config: bundle.config)

        XCTAssertEqual(census.layerStoredBytes, metadata.layerStoredBytes)
        XCTAssertEqual(census.layerResidentBytes, metadata.layerResidentBytes)
        XCTAssertEqual(census.expertStrideBytes, metadata.expertStrideBytes)
        XCTAssertEqual(census.expertsPerLayer, bundle.shape.experts)
        XCTAssertEqual(census.routedExpertsPerToken, bundle.shape.expertsPerToken)
        XCTAssertEqual(census.moeLayerCount, bundle.shape.layers - 1)
        XCTAssertEqual(
            census.layerTableSHA256,
            ArtifactCensus.layerTableSHA256(
                storedBytes: metadata.layerStoredBytes,
                residentBytes: metadata.layerResidentBytes))
        XCTAssertTrue(
            zip(census.layerStoredBytes, census.layerResidentBytes).contains {
                $0.1 != $0.0 * 2
            },
            "the fixture's F32 vectors must prove residency is not a blanket 2x guess")
    }

    func testFlagshipResponseFramingStopsBeforeAnotherExpensivePass() {
        let base = 163_584
        XCTAssertTrue(
            K3GenerationBoundary.endsResponse(
                tokenID: base + K3SpecialToken.endOfSequence.rawValue,
                vocabularySize: 163_840))
        XCTAssertTrue(
            K3GenerationBoundary.endsResponse(
                tokenID: base + K3SpecialToken.endOfMessage.rawValue,
                vocabularySize: 163_840))
        XCTAssertTrue(
            K3GenerationBoundary.endsResponse(
                tokenID: base + K3SpecialToken.close.rawValue,
                vocabularySize: 163_840))
        XCTAssertFalse(
            K3GenerationBoundary.endsResponse(
                tokenID: base + K3SpecialToken.open.rawValue,
                vocabularySize: 163_840))
        XCTAssertFalse(K3GenerationBoundary.endsResponse(tokenID: 4, vocabularySize: 256))
    }

    /// Two passes, two tokens, and an accounting identity that holds in every
    /// sample rather than only at the end.
    func testATinyDecodeReachesItsTokensAndAccountsForEveryByte() async throws {
        let runner = K3RunFixture.runner()
        let session = try runner.start(
            K3RunFixture.request(root: bundle.root, workingDirectory: working, newTokens: 2))
        let run = await DrainedRun.drain(session)

        XCTAssertNil(run.failure, "\(run.failure.map { "\($0)" } ?? "")")
        let summary = try XCTUnwrap(run.finished, "the run did not finish")
        XCTAssertNil(run.cancelled)

        // The stream's shape: an acceptance first, a token per pass, a terminal
        // summary last and nothing after it.
        if case .accepted = run.events.first {} else {
            XCTFail("the first event was not .accepted")
        }
        if case .finished = run.events.last {} else {
            XCTFail("the last event was not .finished")
        }
        XCTAssertEqual(run.tokens.count, 2)
        XCTAssertEqual(run.tokens.map(\.index), [0, 1])
        XCTAssertEqual(run.tokens.map(\.isPrefillToken), [true, false])
        XCTAssertEqual(summary.tokenIDs.count, 2)
        XCTAssertEqual(run.tokens.map(\.tokenID), summary.tokenIDs)
        XCTAssertTrue(
            summary.tokenIDs.allSatisfy { (0..<bundle.config.vocabSize).contains($0) },
            "a decoded id is outside the vocabulary: \(summary.tokenIDs)")
        // No vocabulary travels with this runner, and it says so rather than
        // inventing text from a tokenizer it does not have.
        XCTAssertNil(summary.text)
        XCTAssertTrue(run.tokens.allSatisfy { $0.text == nil })
        XCTAssertTrue(run.tokens.allSatisfy { $0.secondsSincePreviousToken > 0 })

        // One telemetry sample per layer per pass, and every one of them
        // balanced: read bytes are deterministic bytes plus expert bytes, with
        // nothing unattributed.
        XCTAssertEqual(run.telemetry.count, bundle.config.numHiddenLayers * 2)
        for sample in run.telemetry {
            XCTAssertTrue(sample.bytes.isBalanced, "unbalanced sample: \(sample.bytes)")
            XCTAssertEqual(sample.bytes.unattributedBytes, 0)
            XCTAssertEqual(sample.declaredBudgetBytes, K3RunFixture.declaredBudgetBytes)
            let instruments = try XCTUnwrap(sample.instrumentation)
            let expert = try XCTUnwrap(instruments.expert)
            XCTAssertGreaterThanOrEqual(expert.phaseSeconds, expert.ioWaitSeconds)
            XCTAssertGreaterThanOrEqual(expert.cacheHits, 0)
            XCTAssertGreaterThanOrEqual(expert.cacheMisses, 0)
            XCTAssertNotNil(instruments.readAhead)
        }
        // Monotonic, because a byte counter that went backwards would mean two
        // counters were being read as one.
        let totals = run.telemetry.map(\.bytes.totalBytesRead)
        XCTAssertEqual(totals, totals.sorted())

        // Prefill is not a decode token. During token 1's in-flight pass there
        // is no completed decode denominator yet, even though progressive
        // bytes keep moving.
        let layers = bundle.config.numHiddenLayers
        let secondPass = Array(run.telemetry[layers..<(layers * 2)])
        let firstBoundary = try XCTUnwrap(secondPass.first?.completedTokenBytes)
        XCTAssertGreaterThan(firstBoundary.totalBytesRead, 0)
        XCTAssertTrue(secondPass.allSatisfy { $0.completedTokenBytes == firstBoundary })
        XCTAssertTrue(secondPass.allSatisfy { $0.bytesPerToken == nil })
        XCTAssertTrue(secondPass.allSatisfy { $0.tokensPerSecond == nil })
        XCTAssertTrue(secondPass.allSatisfy { $0.generationStage == .decode })
        XCTAssertTrue(
            secondPass.contains { $0.bytes.totalBytesRead > firstBoundary.totalBytesRead },
            "the fixture did not exercise progressive bytes in the second pass")

        let final = summary.finalTelemetry
        XCTAssertTrue(final.bytes.isBalanced, "\(final.bytes)")
        let deterministic = try XCTUnwrap(final.bytes.deterministicBytesRead)
        let expert = try XCTUnwrap(final.bytes.expertBytesRead)
        XCTAssertGreaterThan(deterministic, 0, "no deterministic bytes were read")
        XCTAssertGreaterThan(expert, 0, "no expert bytes were read")
        XCTAssertEqual(final.bytes.totalBytesRead, deterministic + expert)
        XCTAssertEqual(final.completedTokenBytes, final.bytes)
        XCTAssertEqual(final.prefillBytes, firstBoundary)
        let decodeBytes = try XCTUnwrap(final.bytes.subtracting(firstBoundary))
        XCTAssertEqual(final.bytesPerToken, decodeBytes.totalBytesRead)
        XCTAssertEqual(final.decodeTokensCompleted, 1)
        XCTAssertGreaterThan(try XCTUnwrap(final.prefillSeconds), 0)
        XCTAssertGreaterThan(try XCTUnwrap(final.decodeSeconds), 0)
        XCTAssertEqual(final.generationStage, .terminal)
        XCTAssertEqual(final.instrumentation?.readAhead?.nonFreeSlots, 0)
        XCTAssertEqual(final.instrumentation?.readAhead?.outstandingStagedBytes, 0)
        XCTAssertGreaterThan(try XCTUnwrap(final.instrumentation?.expert?.cacheMisses), 0)

        // The budget was stated before the run, and the peak is quoted against
        // it rather than against physical memory.
        XCTAssertEqual(summary.finalTelemetry.declaredBudgetBytes, K3RunFixture.declaredBudgetBytes)
        XCTAssertTrue(
            summary.budgetRespected,
            "peak \(summary.peakFootprintBytes) B against \(K3RunFixture.declaredBudgetBytes) B")
        XCTAssertEqual(summary.budgetRespected, summary.finalTelemetry.budgetRespected)
        XCTAssertGreaterThan(summary.peakFootprintBytes, 0)
        XCTAssertGreaterThan(try XCTUnwrap(summary.tokensPerSecond), 0)
        XCTAssertGreaterThan(summary.wallSeconds, 0)
        XCTAssertFalse(summary.gitRevision.isEmpty)
        XCTAssertFalse(summary.thermalTrail.isEmpty)

        // The acceptance states the knobs the run used, not the ones the caller
        // set: every one of them is filled in.
        let acceptance = try XCTUnwrap(run.acceptance)
        XCTAssertEqual(acceptance.handle, session.handle)
        XCTAssertEqual(acceptance.declaredBudgetBytes, K3RunFixture.declaredBudgetBytes)
        XCTAssertNotNil(acceptance.effectiveKnobs.expertReadAhead)
        XCTAssertNotNil(acceptance.effectiveKnobs.queueDepth)
        XCTAssertNotNil(acceptance.effectiveKnobs.logitChunkRows)
        XCTAssertNotNil(acceptance.effectiveKnobs.granularity)

        let payload = try run.payload()
        XCTAssertEqual(payload["outcome"] as? String, "finished")
        XCTAssertEqual(payload["bytesBalanced"] as? Bool, true)
        XCTAssertEqual(payload["nonFiniteLogits"] as? Int, 0)
        XCTAssertEqual((payload["generatedTokenIds"] as? [Int])?.count, 2)
        XCTAssertEqual((payload["passSeconds"] as? [Double])?.count, 2)
        XCTAssertGreaterThan(try XCTUnwrap(payload["prefillSeconds"] as? Double), 0)
        XCTAssertGreaterThan(try XCTUnwrap(payload["decodeSeconds"] as? Double), 0)
        XCTAssertEqual(
            (payload["layerSeconds"] as? [Double])?.count, bundle.config.numHiddenLayers)
        XCTAssertEqual(
            (payload["logitsSHA256"] as? String)?.count, 64, "the run recorded no logits digest")
        XCTAssertEqual(summary.logitsDigest?.algorithm, .sha256)
        XCTAssertEqual(summary.logitsDigest?.tokenIndex, 1)
        XCTAssertEqual(summary.logitsDigest?.hex, payload["logitsSHA256"] as? String)
        XCTAssertEqual(summary.resultJSONFilename, "k3-decode-run.json")
        let configuration = try XCTUnwrap(payload["configuration"] as? [String: Any])
        XCTAssertEqual(
            configuration["blobReference"] as? String, "absolutePath",
            "an unrooted fixture must describe the absolute references it actually used")
        try assertTornDown(payload)

        print(
            "[runner] \(summary.headline); "
                + "\(deterministic) B deterministic + \(expert) B expert")
    }

    /// Where a K3 pass went, published at the same boundary as its token.
    ///
    /// The terms are differences of two run-scoped readings plus the model's
    /// own per-pass arrays, so the two claims worth checking on a real decode
    /// are that each pass gets exactly one summary and that the terms fit
    /// inside the wall the pass was measured against. A pass whose subtraction
    /// failed closed is absent rather than present as zeros, so this asserts
    /// the fixture produced the attributed case.
    func testEachPassPublishesWhereItsWallTimeWent() async throws {
        let runner = K3RunFixture.runner()
        let run = await DrainedRun.drain(
            try runner.start(
                K3RunFixture.request(
                    root: bundle.root, workingDirectory: working, newTokens: 2,
                    knobs: K3RunFixture.stagedKnobs())))

        XCTAssertNil(run.failure, "\(run.failure.map { "\($0)" } ?? "")")
        XCTAssertEqual(run.phaseSummaries.count, 2, "one summary per pass")
        XCTAssertEqual(run.phaseSummaries.map(\.passKind), [.prefill, .decode])
        XCTAssertEqual(run.phaseSummaries.map(\.passIndex), [0, 1])
        XCTAssertEqual(
            run.phaseSummaries.count, run.logs.filter { $0.hasPrefix("[k3 phase]") }.count,
            "the event and the log line are one measurement, not two")

        for (index, summary) in run.phaseSummaries.enumerated() {
            let label = "pass \(index)"
            XCTAssertGreaterThan(summary.passSeconds, 0, label)
            XCTAssertTrue(summary.isBalanced, "\(label): the terms do not fit inside the pass")
            XCTAssertLessThanOrEqual(
                summary.attributedSeconds, summary.passSeconds + 1e-9, label)
            XCTAssertGreaterThanOrEqual(summary.unattributedSeconds, 0, label)
            XCTAssertTrue(summary.terms.allSatisfy { $0.seconds >= 0 }, label)
            // The two the panel leans on, and the two that are beside the pass
            // rather than inside it.
            XCTAssertNotNil(summary.seconds(of: RunPhaseTermName.expertIOWait), label)
            XCTAssertNotNil(summary.seconds(of: RunPhaseTermName.expertGatherCompute), label)
            XCTAssertEqual(
                Set(summary.besideTerms.map(\.name)),
                [RunPhaseTermName.stagedRead, RunPhaseTermName.layerTotal], label)
            XCTAssertEqual(
                summary.terms.first { $0.name == RunPhaseTermName.layerTotal }?.count,
                bundle.config.numHiddenLayers, label)
            XCTAssertEqual(
                summary.terms.first { $0.name == RunPhaseTermName.attentionBarrier }?.count,
                bundle.config.numHiddenLayers, label)
        }

        // The expert phase is entered on every pass of a MoE artifact, so the
        // split is evidence rather than a set of structural zeros.
        XCTAssertGreaterThan(
            run.phaseSummaries.reduce(0) {
                $0 + ($1.seconds(of: RunPhaseTermName.expertGatherCompute) ?? 0)
            }, 0)

        // The record carries the same summaries the stream did.
        let payload = try run.payload()
        let recorded = try XCTUnwrap(payload["phaseSummaries"] as? [[String: Any]])
        XCTAssertEqual(recorded.count, 2)
        XCTAssertEqual(recorded.first?["passKind"] as? String, "prefill")
        XCTAssertEqual(
            recorded.compactMap { $0["passSeconds"] as? Double },
            run.phaseSummaries.map(\.passSeconds))
    }

    func testSyntheticRuntimeCompletesTheProductTokenCeiling() async throws {
        let requested = K3RunFixture.maximumNewTokens
        let run = await DrainedRun.drain(
            try K3RunFixture.runner().start(
                K3RunFixture.request(
                    root: bundle.root, workingDirectory: working,
                    newTokens: requested)))

        XCTAssertNil(run.failure, "\(run.failure.map { "\($0)" } ?? "")")
        XCTAssertEqual(run.tokens.count, requested)
        XCTAssertEqual(run.finished?.tokenIDs.count, requested)
        XCTAssertTrue(run.finished?.finalTelemetry.bytes.isBalanced == true)
        XCTAssertTrue(run.finished?.budgetRespected == true)
        XCTAssertLessThan(
            try XCTUnwrap(run.finished?.peakFootprintBytes),
            256 << 20,
            "64 recurrent product passes grew beyond the tiny fixture's bounded working set")
    }

    func testProductRunConsumesFullyVerifiedRootedDescriptors() async throws {
        let published = working.appendingPathComponent("published", isDirectory: true)
        try SyntheticArtifact.writePublished(from: bundle, at: published)
        let rooted = try makeRuntimeAuthority(for: published, beneath: working)

        // Warm MLX before the descriptor comparison. Device-global descriptors
        // belong to the process, not to one rooted artifact run.
        let warm = await DrainedRun.drain(
            try K3RunFixture.runner().start(
                K3RunFixture.request(
                    root: bundle.root, workingDirectory: working, newTokens: 1)))
        let expected = try XCTUnwrap(warm.finished?.tokenIDs.first)
        let before = K3RunFixture.openDescriptors()

        let run = await DrainedRun.drain(
            try K3RunFixture.runner().start(
                K3RunFixture.request(
                    root: published, workingDirectory: working, newTokens: 1,
                    runtimeAuthority: rooted)))

        XCTAssertNil(run.failure, "\(run.failure.map { "\($0)" } ?? "")")
        XCTAssertEqual(run.finished?.tokenIDs, [expected])
        XCTAssertEqual(K3RunFixture.openDescriptors(), before)
        XCTAssertTrue(
            run.logs.contains(where: { $0.contains("translated:") }),
            "the rooted published layout did not pass through production translation")
        let payload = try run.payload()
        let configuration = try XCTUnwrap(payload["configuration"] as? [String: Any])
        XCTAssertEqual(
            configuration["blobReference"] as? String, "artifactReference",
            "the product result must describe its rooted descriptor-backed references")
    }

    /// The memory dial is not a decorative App calculation: the exact plan is
    /// accepted by the facade, configures `ArtifactWeightSource`, serves the
    /// second token from resident stored bytes, and is present beside the
    /// post-teardown metrics in the run's reproducibility payload.
    func testAPinPlanIsHonouredAndReleasedByTheRealRuntime() async throws {
        let probe = try ArtifactWeightSource(
            translated: bundle.translated.path, globals: bundle.globals.path,
            config: bundle.config)
        let stored = probe.layerBytes.map(\.storedBytes)
        let resident = probe.layerBytes.map(\.residentBytes)
        let poolProbe = try FlagshipRoutedExpertBackend(layers: probe.streams)
        let expertPoolBytes = UInt64(poolProbe.budgetBytes)
        poolProbe.shutdown()
        probe.shutdown()

        let deterministic = stored.reduce(UInt64(0), +)
        let census = try ArtifactCensus(
            layerStoredBytes: stored,
            layerResidentBytes: resident,
            // Keep the synthetic globals outside this fixture's budget. Their
            // real per-token reads are deliberately not part of this narrow
            // pinned-layer census.
            globalsStoredBytes: K3RunFixture.declaredBudgetBytes * 2,
            globalsBytesReadPerToken: 0,
            expertStrideBytes: 0,
            expertsPerLayer: 0,
            routedExpertsPerToken: 0,
            moeLayerCount: 0,
            measuredDeterministicBytesPerToken: deterministic)
        let floor = WorkingSetFloor(
            widestResidentLayerBytes: resident.max() ?? 0,
            expertPoolBytes: expertPoolBytes,
            workingReserveBytes: 500_000_000)
        let plan = try MemoryDialPlanner.plan(
            budgetBytes: K3RunFixture.declaredBudgetBytes,
            census: census,
            floor: floor,
            maximumNewTokens: 2)
        XCTAssertFalse(plan.pinnedLayers.isEmpty)
        XCTAssertNil(plan.pinnedGlobals)
        XCTAssertNil(plan.expertHotSet)

        let session = try K3RunFixture.runner().start(
            K3RunFixture.request(
                root: bundle.root, workingDirectory: working, newTokens: 2,
                pinPlan: plan))
        let run = await DrainedRun.drain(session)

        XCTAssertNil(run.failure, "\(run.failure.map { "\($0)" } ?? "")")
        XCTAssertEqual(run.acceptance?.pinPlan, plan)
        let summary = try XCTUnwrap(run.finished)
        XCTAssertTrue(summary.finalTelemetry.bytes.isBalanced)

        let payload = try run.payload()
        let recordedPlan = try XCTUnwrap(payload["pinPlan"] as? [String: Any])
        XCTAssertEqual(
            recordedPlan["budgetBytes"] as? Int,
            Int(K3RunFixture.declaredBudgetBytes))
        let pins = try XCTUnwrap(payload["pinnedTier"] as? [String: Any])
        XCTAssertEqual(pins["pinnedLayerCount"] as? Int, plan.pinnedLayers.count)
        XCTAssertGreaterThan(pins["bytesLoaded"] as? Int ?? 0, 0)
        XCTAssertGreaterThan(pins["bytesServed"] as? Int ?? 0, 0)
        XCTAssertGreaterThan(pins["servedLayerCount"] as? Int ?? 0, 0)
        XCTAssertEqual(pins["residentBytes"] as? Int, 0)
        XCTAssertEqual(pins["freedBytes"] as? Int, pins["bytesLoaded"] as? Int)
        XCTAssertEqual(pins["isAccountingBalanced"] as? Bool, true)
        try assertTornDown(payload)
    }

    private func makeRuntimeAuthority(
        for artifact: URL, beneath registeredLocation: URL
    ) throws -> ArtifactRuntimeAuthority {
        let root = try ArtifactVerificationRoot.open(
            relativeComponents: [artifact.lastPathComponent], beneath: registeredLocation)
        let keys: Set<URLResourceKey> = [.isRegularFileKey]
        let enumerator = try XCTUnwrap(
            FileManager.default.enumerator(
                at: artifact, includingPropertiesForKeys: Array(keys),
                options: [.skipsHiddenFiles]))
        let canonicalRoot = artifact.resolvingSymlinksInPath().standardizedFileURL.path
        var files: [ArtifactVerifiedFile] = []
        for case let url as URL in enumerator {
            guard try url.resourceValues(forKeys: keys).isRegularFile == true else { continue }
            let canonicalFile = url.resolvingSymlinksInPath().standardizedFileURL.path
            guard canonicalFile.hasPrefix(canonicalRoot + "/") else {
                throw POSIXError(.EPERM)
            }
            let relative = String(canonicalFile.dropFirst(canonicalRoot.count + 1))
            let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
            guard descriptor >= 0 else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            let identity: ArtifactFilesystemIdentity
            do {
                identity = try ArtifactFilesystemEvidence.identity(
                    of: descriptor, path: relative, expectedDirectory: false)
            } catch {
                _ = Darwin.close(descriptor)
                throw error
            }
            _ = Darwin.close(descriptor)
            let payload = relative.hasSuffix(".bin") || relative.hasSuffix(".mxfp4tile")
            files.append(
                ArtifactVerifiedFile(
                    path: relative, expectedSizeBytes: identity.sizeBytes,
                    digest: .sha256(
                        hex: try FileDigestComputer.sha256(ofFileAt: url)),
                    isPayload: payload, filesystem: identity))
        }
        files.sort { $0.path < $1.path }
        let repoFiles = files.map {
            RepoFile(
                path: $0.path, sizeBytes: $0.expectedSizeBytes, digest: $0.digest,
                isPayload: $0.isPayload)
        }
        let total = try checkedByteTotal(files)
        let payloads = files.filter(\.isPayload)
        let metadata = files.filter { !$0.isPayload }
        let payloadBytes = try checkedByteTotal(payloads)
        let metadataBytes = try checkedByteTotal(metadata)
        let planIdentity = ArtifactDigestPlanIdentity.compute(repoFiles)
        let index = ArtifactIndexIdentity.parse(
            try Data(contentsOf: artifact.appendingPathComponent("index.json")))
        let evidence = ArtifactVerificationEvidence(
            model: .kimiK3,
            repository: HuggingFaceRepoRef(
                repoID: "nanguoyu/Kimi-K3-minirun",
                revision: String(repeating: "e", count: 40)),
            treeIdentity: planIdentity,
            completenessAuthorityIdentity: planIdentity,
            selectedPlanIdentity: planIdentity,
            treeFileCount: files.count,
            treePayloadFileCount: payloads.count,
            treeMetadataFileCount: metadata.count,
            treeBytes: total,
            treePayloadBytes: payloadBytes,
            treeMetadataBytes: metadataBytes,
            selectedBytes: total,
            index: index,
            root: root.artifactIdentity,
            files: files)
        return try ArtifactRuntimeAuthority(root: root, evidence: evidence)
    }

    private func checkedByteTotal(_ files: [ArtifactVerifiedFile]) throws -> UInt64 {
        try files.reduce(0) { total, file in
            let (next, overflow) = total.addingReportingOverflow(file.expectedSizeBytes)
            if overflow { throw POSIXError(.EOVERFLOW) }
            return next
        }
    }

    func testAnArtifactMismatchFailsBeforeThePlanIsAccepted() async throws {
        let probe = try ArtifactWeightSource(
            translated: bundle.translated.path, globals: bundle.globals.path,
            config: bundle.config)
        var stored = probe.layerBytes.map(\.storedBytes)
        let resident = probe.layerBytes.map(\.residentBytes)
        let poolProbe = try FlagshipRoutedExpertBackend(layers: probe.streams)
        let expertPoolBytes = UInt64(poolProbe.budgetBytes)
        poolProbe.shutdown()
        probe.shutdown()
        stored[0] += 1
        let census = try ArtifactCensus(
            layerStoredBytes: stored, layerResidentBytes: resident,
            globalsStoredBytes: K3RunFixture.declaredBudgetBytes * 2,
            globalsBytesReadPerToken: 0, expertStrideBytes: 0,
            expertsPerLayer: 0, routedExpertsPerToken: 0, moeLayerCount: 0,
            measuredDeterministicBytesPerToken: stored.reduce(0, +))
        let plan = try MemoryDialPlanner.plan(
            budgetBytes: K3RunFixture.declaredBudgetBytes,
            census: census,
            floor: WorkingSetFloor(
                widestResidentLayerBytes: resident.max() ?? 0,
                expertPoolBytes: expertPoolBytes,
                workingReserveBytes: 500_000_000))

        let run = await DrainedRun.drain(
            try K3RunFixture.runner().start(
                K3RunFixture.request(
                    root: bundle.root, workingDirectory: working, newTokens: 1,
                    pinPlan: plan)))
        XCTAssertNotNil(run.failure)
        XCTAssertNil(run.acceptance, "a plan that cannot bind to the artifact was accepted")
        XCTAssertTrue(run.events.isEmpty, "preflight events escaped before acceptance")
        XCTAssertTrue("\(run.failure!)".contains("census does not describe this artifact"))
    }

    /// Two runs of the same request produce the same token ids. The facade adds
    /// a thread, a stream and a cancel latch to the decode path; none of them
    /// may change what it computes.
    func testTheFacadeDoesNotChangeWhatIsDecoded() async throws {
        let runner = K3RunFixture.runner()
        var digests: [String] = []
        var ids: [[Int]] = []
        for _ in 0..<2 {
            let session = try runner.start(
                K3RunFixture.request(root: bundle.root, workingDirectory: working, newTokens: 2))
            let run = await DrainedRun.drain(session)
            let summary = try XCTUnwrap(run.finished)
            ids.append(summary.tokenIDs)
            digests.append(try XCTUnwrap(run.payload()["logitsSHA256"] as? String))
        }
        XCTAssertEqual(ids[0], ids[1])
        XCTAssertEqual(digests[0], digests[1], "the same request decoded to different logits")

        // And the same ids the model produces when it is driven directly, with
        // no runner in the picture at all.
        let source = try ArtifactWeightSource(
            translated: bundle.translated.path, globals: bundle.globals.path,
            config: bundle.config)
        let backend = try FlagshipRoutedExpertBackend(layers: source.streams)
        defer {
            backend.shutdown()
            source.shutdown()
        }
        let model = K3Model(config: bundle.config, source: source, expertBackend: backend)
        var state = TinyK3State()
        var direct: [Int] = []
        var capture = try model.forward(tokenIds: [3, 17, 200], state: &state)
        direct.append(capture.greedyTokenId)
        capture = try model.forward(tokenIds: [capture.greedyTokenId], state: &state)
        direct.append(capture.greedyTokenId)
        XCTAssertEqual(ids[0], direct, "the facade decoded something the model does not")
    }

    // MARK: - A run that is stopped

    /// Stop, pressed as the first token arrives, with the deterministic
    /// read-ahead staging a layer ahead — so there are bytes in flight to give
    /// back at the moment the run is interrupted.
    func testCancellingMidDecodeEndsTheRunAndFreesEverything() async throws {
        // Warm: the first MLX run in a process opens the Metal device and its
        // descriptors, and those are the process's, not the run's.
        _ = await DrainedRun.drain(
            try K3RunFixture.runner().start(
                K3RunFixture.request(root: bundle.root, workingDirectory: working, newTokens: 1)))
        let before = K3RunFixture.openDescriptors()

        let runner = K3RunFixture.runner()
        let requested = 48
        let session = try runner.start(
            K3RunFixture.request(
                root: bundle.root, workingDirectory: working, newTokens: requested,
                knobs: K3RunFixture.stagedKnobs()))
        let cancel = session.cancel
        let run = await DrainedRun.drain(session) { token in
            if token.index == 0 { cancel() }
        }

        XCTAssertNil(run.failure)
        let summary = try XCTUnwrap(run.cancelled, "the run did not end as .cancelled")
        XCTAssertNil(run.finished)
        if case .cancelled = run.events.last {} else {
            XCTFail("the last event was not .cancelled")
        }
        // A named ending, not a silently short result: it stopped early *and*
        // it says what it had measured when it did.
        XCTAssertLessThan(
            summary.tokenIDs.count, requested,
            "the run reached every token, so nothing was cancelled mid-decode")
        XCTAssertGreaterThanOrEqual(summary.tokenIDs.count, 1)
        XCTAssertTrue(summary.headline.hasPrefix("cancelled after"), summary.headline)
        XCTAssertTrue(summary.finalTelemetry.bytes.isBalanced, "\(summary.finalTelemetry.bytes)")
        XCTAssertGreaterThan(summary.finalTelemetry.bytes.totalBytesRead, 0)

        let payload = try run.payload()
        XCTAssertEqual(payload["outcome"] as? String, "cancelled")
        try assertTornDown(payload)
        // The staging arm ran, and gave back what it had staged.
        let readAhead = try XCTUnwrap(payload["deterministicReadAhead"] as? [String: Any])
        XCTAssertEqual(readAhead["depth"] as? Int, 1)
        XCTAssertGreaterThan(readAhead["stagedBytes"] as? Int ?? 0, 0)
        XCTAssertEqual(readAhead["outstandingStagedBytes"] as? Int, 0)

        let after = K3RunFixture.openDescriptors()
        XCTAssertLessThanOrEqual(
            after, before,
            "a cancelled run left \(after - before) descriptors open")
    }

    /// The second door. Cancelling the Task that drains the stream tears the
    /// run down exactly as `RunSession.cancel` does — and because that consumer
    /// is gone, the proof cannot be an event: it is the descriptors coming back.
    func testCancellingTheConsumingTaskIsTheSameDoor() async throws {
        _ = await DrainedRun.drain(
            try K3RunFixture.runner().start(
                K3RunFixture.request(root: bundle.root, workingDirectory: working, newTokens: 1)))
        let before = K3RunFixture.openDescriptors()

        let session = try K3RunFixture.runner().start(
            K3RunFixture.request(
                root: bundle.root, workingDirectory: working, newTokens: 48,
                knobs: K3RunFixture.stagedKnobs()))
        let firstToken = expectation(description: "the run produced a token")
        firstToken.assertForOverFulfill = false
        let consumer = Task {
            for try await event in session.events {
                if case .token = event { firstToken.fulfill() }
            }
        }
        await fulfillment(of: [firstToken], timeout: 60)
        consumer.cancel()

        // The run holds descriptors until its teardown finishes, and its
        // teardown is on its own thread; polling is what a consumer that has
        // walked away can honestly observe.
        var after = K3RunFixture.openDescriptors()
        let deadline = Date().addingTimeInterval(60)
        while after > before, Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
            after = K3RunFixture.openDescriptors()
        }
        XCTAssertLessThanOrEqual(
            after, before,
            "cancelling the consuming Task left \(after - before) descriptors open")
    }

    /// A run stopped before it opens anything still ends as a cancellation, and
    /// still tears down: the latch is checked before the first read, not only
    /// between layers.
    func testCancellingBeforeTheFirstReadStillEndsAsACancellation() async throws {
        let session = try K3RunFixture.runner().start(
            K3RunFixture.request(root: bundle.root, workingDirectory: working, newTokens: 4))
        session.cancel()
        let run = await DrainedRun.drain(session)
        let summary = try XCTUnwrap(run.cancelled)
        XCTAssertTrue(summary.tokenIDs.isEmpty)
        XCTAssertTrue(summary.finalTelemetry.bytes.isBalanced)
        XCTAssertNil(summary.tokensPerSecond, "a run with no token has no rate to quote")
    }

    /// The scope a run reads through is the operator's grant, and the runner's
    /// own bracket drops it — on the cancelled path too.
    func testTheArtifactScopeIsReleasedByTheRunner() async throws {
        let scope = StorageScope(url: bundle.root)
        let session = try K3RunFixture.runner().start(
            K3RunFixture.request(
                root: bundle.root, workingDirectory: working, newTokens: 1, scope: scope))
        _ = await DrainedRun.drain(session)
        XCTAssertTrue(scope.isReleased, "the run did not release the scope it was handed")

        let cancelled = StorageScope(url: bundle.root)
        let stopped = try K3RunFixture.runner().start(
            K3RunFixture.request(
                root: bundle.root, workingDirectory: working, newTokens: 8, scope: cancelled))
        stopped.cancel()
        _ = await DrainedRun.drain(stopped)
        XCTAssertTrue(cancelled.isReleased, "a cancelled run kept the scope")
    }

    /// The runner stream carries StorageCore's concrete error, not a sentence
    /// wrapped in `RunError.underlying`. The App needs these fields to render a
    /// short transfer without parsing prose or inventing an offset.
    func testAStorageFailureKeepsItsConcreteTypeAndFields() async throws {
        let broken = working.appendingPathComponent("broken-translated", isDirectory: true)
        try FileManager.default.copyItem(at: bundle.translated, to: broken)
        let missing = try XCTUnwrap(
            FileManager.default.contentsOfDirectory(
                at: broken, includingPropertiesForKeys: nil
            ).first { $0.pathExtension == "mxfp4tile" })
        // Leave the manifest intact and remove the file it authorises. The
        // source therefore reaches StorageCore's real `open(2)` error boundary
        // instead of failing earlier on an incomplete synthetic manifest set.
        try FileManager.default.removeItem(at: missing)

        let request = RunRequest(
            model: .kimiK3,
            artifact: ArtifactReference(
                root: bundle.root, auxiliaryRoots: ["translated": broken]),
            prompt: .tokenIDs([3, 17, 200]),
            memoryBudgetBytes: K3RunFixture.declaredBudgetBytes,
            maximumNewTokens: 1,
            workingDirectory: working)
        let session = try K3RunFixture.runner().start(request)
        let run = await DrainedRun.drain(session)

        guard case StorageCoreError.posix(let operation, let path, let code)? =
            run.failure as? StorageCoreError
        else {
            return XCTFail("expected concrete StorageCoreError.posix, got \(String(describing: run.failure))")
        }
        XCTAssertEqual(operation, "open")
        XCTAssertEqual(URL(fileURLWithPath: path).lastPathComponent, missing.lastPathComponent)
        XCTAssertTrue(
            path.hasSuffix("/broken-translated/\(missing.lastPathComponent)"), path)
        XCTAssertNotEqual(code, 0)
        XCTAssertFalse(run.failure is RunError, "the adapter erased the storage fields")
    }

    // MARK: -

    /// The teardown record the run wrote itself, as the terminal event carried
    /// it. Every terminal event has one, cancellation included.
    private func assertTornDown(
        _ payload: [String: Any], file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let teardown = try XCTUnwrap(payload["teardown"] as? [String: Any], file: file, line: line)
        XCTAssertEqual(teardown["sourceShutDown"] as? Bool, true, file: file, line: line)
        XCTAssertEqual(
            teardown["expertBackendShutDown"] as? Bool, true, file: file, line: line)
        XCTAssertEqual(
            teardown["deterministicWorkingSetReleased"] as? Bool, true,
            "the terminal event retained the deterministic source", file: file, line: line)
        XCTAssertEqual(
            teardown["expertWorkingSetReleased"] as? Bool, true,
            "the terminal event retained the expert buffer pool", file: file, line: line)
        XCTAssertEqual(
            teardown["outstandingStagedBytes"] as? Int, 0, "staged bytes were not freed",
            file: file, line: line)
        XCTAssertEqual(
            teardown["readAheadAccountingBalanced"] as? Bool, true, file: file, line: line)
    }
}
