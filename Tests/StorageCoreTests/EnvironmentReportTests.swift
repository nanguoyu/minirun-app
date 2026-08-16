import Foundation
import XCTest

@testable import StorageCore

final class EnvironmentReportTests: XCTestCase {
    func testDeviceInfoReportsPlausibleValues() {
        let device = DeviceInfo.current()
        XCTAssertGreaterThan(device.physicalMemoryBytes, 0)
        XCTAssertGreaterThan(device.pageSizeBytes, 0)
        XCTAssertGreaterThanOrEqual(device.activeProcessorCount, 1)
        XCTAssertNotNil(device.modelIdentifier)
        if let logical = device.cpuCoreCount {
            XCTAssertGreaterThanOrEqual(logical, device.activeProcessorCount)
        }
        // Where the CPU is asymmetric, the per-level counts must add up to the total.
        if let performance = device.performanceCoreCount,
            let efficiency = device.efficiencyCoreCount,
            let logical = device.cpuCoreCount
        {
            XCTAssertEqual(performance + efficiency, logical)
        }
    }

    func testOSInfoReportsThePlatformThatProducedTheRecord() {
        let os = OSInfo.current()
        #if os(iOS)
        XCTAssertEqual(os.name, "iOS")
        #elseif os(macOS)
        XCTAssertEqual(os.name, "macOS")
        #else
        XCTFail("StorageCore has no recorded OS name for this platform")
        #endif
        XCTAssertFalse(os.version.isEmpty)
        XCTAssertEqual(os.kernelType, "Darwin")
    }

    func testSysctlReturnsNilForUnknownKey() {
        XCTAssertNil(Sysctl.string("minirun.definitely.not.a.key"))
        XCTAssertNil(Sysctl.integer("minirun.definitely.not.a.key"))
    }

    func testStorageInfoDescribesExistingDirectory() {
        let info = StorageInfo.describing(path: NSTemporaryDirectory())
        XCTAssertTrue(info.pathExists)
        XCTAssertTrue(info.pathIsDirectory)
        XCTAssertTrue(info.pathIsReadable)
        XCTAssertNotNil(info.volumeName)
        XCTAssertNotNil(info.filesystemType)
        XCTAssertNotNil(info.volumeMountPath)
        XCTAssertGreaterThan(info.volumeTotalBytes ?? 0, 0)
    }

    /// `minirun env --path <file that does not exist yet>` has to work: the
    /// point is often to check the target volume *before* generating a file
    /// onto it.
    func testStorageInfoFallsBackToNearestExistingAncestor() {
        let missing =
            URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-does-not-exist-\(UUID().uuidString)")
            .appendingPathComponent("nor-does-this")
        let info = StorageInfo.describing(path: missing.path)

        XCTAssertFalse(info.pathExists)
        XCTAssertFalse(info.pathIsDirectory)
        XCTAssertNil(info.fileSizeBytes)
        XCTAssertNotNil(info.volumeName, "volume must still resolve via the nearest ancestor")
        XCTAssertNotNil(info.filesystemType)
    }

    func testStorageInfoReportsFileSize() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-size-\(UUID().uuidString).bin")
        try Data(repeating: 0xAB, count: 4096).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let info = StorageInfo.describing(path: url.path)
        XCTAssertTrue(info.pathExists)
        XCTAssertFalse(info.pathIsDirectory)
        XCTAssertEqual(info.fileSizeBytes, 4096)
    }

    func testMemoryUsageIsAvailable() throws {
        let usage = try XCTUnwrap(MemoryUsage.current())
        XCTAssertGreaterThan(usage.residentBytes, 0)
        XCTAssertGreaterThanOrEqual(usage.peakResidentBytes, usage.residentBytes)
    }

    func testReportRoundTripsThroughJSON() throws {
        let report = EnvironmentReport.current(path: NSTemporaryDirectory())
        let data = try JSONOutput.encoder().encode(report)
        let decoded = try JSONOutput.decoder().decode(EnvironmentReport.self, from: data)
        XCTAssertEqual(report, decoded)
    }

    func testReportJSONUsesSnakeCaseKeys() throws {
        let report = EnvironmentReport.current(path: NSTemporaryDirectory())
        let data = try JSONOutput.encoder().encode(report)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])

        for key in ["schema_version", "report_type", "timestamp", "git_revision", "device", "os", "storage", "process"] {
            XCTAssertNotNil(object[key], "missing top-level key '\(key)'")
        }
        let device = try XCTUnwrap(object["device"] as? [String: Any])
        XCTAssertNotNil(device["physical_memory_bytes"])
        XCTAssertNotNil(device["page_size_bytes"])
        let storage = try XCTUnwrap(object["storage"] as? [String: Any])
        XCTAssertNotNil(storage["resolved_path"])
        XCTAssertNotNil(storage["filesystem_type"])
    }

    /// Optional environment fields keep their keys, so `device.cpu_brand` is
    /// present-and-null on a machine that does not report it rather than
    /// absent. Forced here by describing a path with no file, which nulls
    /// `file_size_bytes`.
    func testOptionalFieldsAppearAsExplicitNulls() throws {
        let missing = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("minirun-null-\(UUID().uuidString)")
        let report = EnvironmentReport.current(path: missing.path)
        let data = try JSONOutput.encoder().encode(report)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        let storage = try XCTUnwrap(object["storage"] as? [String: Any])
        XCTAssertTrue(storage.keys.contains("file_size_bytes"))
        XCTAssertTrue(storage["file_size_bytes"] is NSNull, "absent value must encode as null")

        for key in [
            "volume_name", "volume_mount_path", "filesystem_type", "volume_total_bytes",
            "volume_available_bytes", "volume_is_internal", "volume_is_removable",
            "volume_is_ejectable", "volume_is_read_only",
        ] {
            XCTAssertTrue(storage.keys.contains(key), "storage.\(key) must always be present")
        }

        let device = try XCTUnwrap(object["device"] as? [String: Any])
        for key in [
            "model_identifier", "cpu_brand", "cpu_core_count", "physical_core_count",
            "performance_core_count", "efficiency_core_count",
        ] {
            XCTAssertTrue(device.keys.contains(key), "device.\(key) must always be present")
        }

        let process = try XCTUnwrap(object["process"] as? [String: Any])
        XCTAssertTrue(process.keys.contains("resident_bytes"))
        XCTAssertTrue(process.keys.contains("peak_resident_bytes"))
    }

    func testHumanSummaryMentionsKeyFacts() {
        let summary = EnvironmentReport.current(path: NSTemporaryDirectory()).humanSummary()
        XCTAssertTrue(summary.contains("device"))
        XCTAssertTrue(summary.contains("storage"))
        #if os(iOS)
        XCTAssertTrue(summary.contains("iOS"))
        #elseif os(macOS)
        XCTAssertTrue(summary.contains("macOS"))
        #endif
    }
}
