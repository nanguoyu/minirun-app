import XCTest

@testable import MinirunKit

/// The product-facing decomposition: its arithmetic, its aggregate, and the two
/// things it promises about records — that an older one still decodes, and that
/// a term this build has never heard of survives a round trip and still reaches
/// a screen.
final class RunPhaseSummaryTests: XCTestCase {

    private func summary(
        kind: RunPassKind = .decode, index: Int? = 1, seconds: Double = 10,
        terms: [RunPhaseTerm]
    ) -> RunPhaseSummary {
        RunPhaseSummary(
            passKind: kind, passIndex: index, passSeconds: seconds, terms: terms)
    }

    // MARK: The identity

    func testTheResidualIsThePassMinusTheTermsInsideIt() {
        let split = summary(
            terms: [
                RunPhaseTerm(name: RunPhaseTermName.expertIOWait, seconds: 6, count: 48),
                RunPhaseTerm(name: RunPhaseTermName.expertGatherCompute, seconds: 2),
            ])

        XCTAssertEqual(split.attributedSeconds, 8, accuracy: 1e-9)
        XCTAssertEqual(split.unattributedSeconds, 2, accuracy: 1e-9)
        XCTAssertTrue(split.isBalanced)
        XCTAssertEqual(split.fraction(of: 6) ?? 0, 0.6, accuracy: 1e-9)
        XCTAssertEqual(split.seconds(of: RunPhaseTermName.expertIOWait), 6)
        XCTAssertNil(split.seconds(of: "neverMeasured"))
    }

    /// A term measured beside the pass overlaps it by construction — a
    /// background stager reads while the loop computes — so summing it into the
    /// identity would make the parts exceed the whole.
    func testTermsBesideThePassAreReportedButNeverSummedIntoIt() {
        let split = summary(
            terms: [
                RunPhaseTerm(name: RunPhaseTermName.deterministicRead, seconds: 1),
                RunPhaseTerm(
                    name: RunPhaseTermName.stagedRead, seconds: 40, isInsidePass: false),
                RunPhaseTerm(
                    name: RunPhaseTermName.layerTotal, seconds: 9, count: 93,
                    isInsidePass: false),
            ])

        XCTAssertEqual(split.attributedSeconds, 1, accuracy: 1e-9)
        XCTAssertEqual(split.unattributedSeconds, 9, accuracy: 1e-9)
        XCTAssertTrue(split.isBalanced, "40 s of overlapped staging is not 40 s of the pass")
        XCTAssertEqual(split.insideTerms.map(\.name), [RunPhaseTermName.deterministicRead])
        XCTAssertEqual(
            split.besideTerms.map(\.name),
            [RunPhaseTermName.stagedRead, RunPhaseTermName.layerTotal])
    }

    func testTermsThatOverrunThePassAreNotBalanced() {
        XCTAssertFalse(
            summary(seconds: 4, terms: [RunPhaseTerm(name: "a", seconds: 5)]).isBalanced)
        XCTAssertFalse(
            summary(terms: [RunPhaseTerm(name: "a", seconds: -1)]).isBalanced)
        XCTAssertFalse(
            summary(seconds: .nan, terms: [RunPhaseTerm(name: "a", seconds: 1)]).isBalanced)
    }

    /// What a bar draws: the named terms largest first, then the residual, so
    /// the slices always sum to the pass rather than to the part that was named.
    func testTheDrawnOrderIsLargestFirstWithTheResidualLast() {
        let split = summary(
            terms: [
                RunPhaseTerm(name: "small", seconds: 1),
                RunPhaseTerm(name: "large", seconds: 5),
                RunPhaseTerm(name: "beside", seconds: 3, isInsidePass: false),
                RunPhaseTerm(name: "medium", seconds: 2),
            ])

        XCTAssertEqual(
            split.orderedTermsWithResidual.map(\.name),
            ["large", "medium", "small", RunPhaseTermName.unattributed])
        XCTAssertEqual(
            split.orderedTermsWithResidual.reduce(0) { $0 + $1.seconds }, 10, accuracy: 1e-9)
    }

    func testAFullyAttributedPassDrawsNoResidualSlice() {
        let split = summary(terms: [RunPhaseTerm(name: "all", seconds: 10)])
        XCTAssertEqual(split.orderedTermsWithResidual.map(\.name), ["all"])
    }

    // MARK: The aggregate

    func testTheDecodeAggregateSumsSecondsAndCountsAndStatesItsPassCount() throws {
        let aggregate = try XCTUnwrap(
            RunPhaseSummary.aggregate([
                summary(
                    index: 1, seconds: 10,
                    terms: [
                        RunPhaseTerm(name: "io", seconds: 6, count: 4),
                        RunPhaseTerm(name: "gather", seconds: 1),
                    ]),
                summary(
                    index: 2, seconds: 20,
                    terms: [
                        RunPhaseTerm(name: "io", seconds: 10, count: 6),
                        RunPhaseTerm(name: "gather", seconds: 3),
                    ]),
            ]))

        XCTAssertEqual(aggregate.passKind, .decode)
        XCTAssertNil(aggregate.passIndex, "an aggregate is not one pass")
        XCTAssertEqual(aggregate.passCount, 2)
        XCTAssertEqual(aggregate.passSeconds, 30, accuracy: 1e-9)
        XCTAssertEqual(aggregate.secondsPerPass ?? 0, 15, accuracy: 1e-9)
        XCTAssertEqual(aggregate.seconds(of: "io"), 16)
        XCTAssertEqual(aggregate.terms.first { $0.name == "io" }?.count, 10)
        XCTAssertEqual(aggregate.unattributedSeconds, 10, accuracy: 1e-9)
    }

    /// A partial sum of counts is not a count: if one pass counted a bracket
    /// and another did not, the aggregate reports no count rather than the
    /// count of the passes that happened to have one.
    func testACountIsAggregatedOnlyWhenEveryPassCountedIt() {
        let aggregate = RunPhaseSummary.aggregate([
            summary(terms: [RunPhaseTerm(name: "io", seconds: 1, count: 4)]),
            summary(terms: [RunPhaseTerm(name: "io", seconds: 1)]),
        ])
        XCTAssertNil(aggregate?.terms.first?.count)

        let partial = RunPhaseSummary.aggregate([
            summary(terms: [RunPhaseTerm(name: "io", seconds: 1, count: 4)]),
            summary(terms: [RunPhaseTerm(name: "other", seconds: 1, count: 2)]),
        ])
        XCTAssertNil(
            partial?.terms.first { $0.name == "io" }?.count,
            "a term missing from a pass cannot carry that pass's share of the count")
    }

    func testMixedPassKindsDoNotAggregate() {
        XCTAssertNil(
            RunPhaseSummary.aggregate([
                summary(kind: .prefill, index: 0, terms: []),
                summary(kind: .decode, index: 1, terms: []),
            ]),
            "a decode aggregate that absorbed the prefill would misattribute the turn")
        XCTAssertNil(RunPhaseSummary.aggregate([]))
    }

    // MARK: Records

    func testARoundTripKeepsEveryTermIncludingOnesThisBuildDoesNotKnow() throws {
        let original = summary(
            kind: .prefill, index: 0, seconds: 12,
            terms: [
                RunPhaseTerm(name: RunPhaseTermName.expertIOWait, seconds: 5, count: 48),
                RunPhaseTerm(name: "kvCacheRebuild", seconds: 2, count: 93),
                RunPhaseTerm(name: "someFutureStager", seconds: 30, isInsidePass: false),
            ])
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(RunPhaseSummary.self, from: data)

        XCTAssertEqual(decoded, original)
        XCTAssertEqual(decoded.seconds(of: "kvCacheRebuild"), 2)
        XCTAssertFalse(
            try XCTUnwrap(decoded.terms.first { $0.name == "someFutureStager" }).isInsidePass)
        XCTAssertEqual(decoded.attributedSeconds, 7, accuracy: 1e-9)

        // The derived terms are written for a reader of the record and never
        // read back, so the two cannot disagree.
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["attributedSeconds"] as? Double, 7)
        XCTAssertEqual(object["unattributedSeconds"] as? Double, 5)
    }

    /// A screen must not silently drop a term it does not recognise: a dropped
    /// slice under-reports a pass that still sums to its wall time.
    func testAnUnknownTermStillGetsAReadableLabel() {
        XCTAssertEqual(
            RunPhaseTermName.displayName(for: RunPhaseTermName.expertIOWait),
            "Waiting for expert weights")
        XCTAssertEqual(RunPhaseTermName.displayName(for: "kvCacheRebuild"), "Kv cache rebuild")
        XCTAssertEqual(RunPhaseTermName.displayName(for: "reads"), "Reads")
        XCTAssertEqual(RunPhaseTermName.displayName(for: ""), "")
    }

    /// The record written before terms could sit beside a pass, and before an
    /// aggregate could state a pass count.
    func testAnOlderRecordDecodesWithTheMeaningItWasWrittenUnder() throws {
        let json = """
            {
              "passKind": "decode",
              "passIndex": 3,
              "passSeconds": 8,
              "terms": [
                { "name": "expertIOWait", "seconds": 5 },
                { "name": "reclaim", "seconds": 1, "count": 93 }
              ]
            }
            """
        let decoded = try JSONDecoder().decode(
            RunPhaseSummary.self, from: Data(json.utf8))

        XCTAssertEqual(decoded.passCount, 1)
        XCTAssertEqual(decoded.passIndex, 3)
        XCTAssertTrue(decoded.terms.allSatisfy(\.isInsidePass))
        XCTAssertEqual(decoded.attributedSeconds, 6, accuracy: 1e-9)
        XCTAssertEqual(decoded.unattributedSeconds, 2, accuracy: 1e-9)
    }
}
