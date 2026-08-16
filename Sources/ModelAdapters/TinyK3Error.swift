import Foundation

/// Everything this adapter refuses to do, said out loud.
///
/// Spec §5.8 ("fail explicitly") and §15.6: an unsupported modality, a missing
/// tensor or a geometry the reference never exercised is reported with the
/// names and shapes involved. Nothing here falls back to a plausible guess —
/// a silently wrong architecture is the failure mode M6A's fixtures exist to
/// catch, and a silently *substituted* one would defeat them.
public enum TinyK3Error: Error, CustomStringConvertible, LocalizedError {
    case missingTensor(String)
    case unsupportedDType(tensor: String, dtype: String)
    case malformedSafetensors(String)
    case configuration(String)
    case unsupportedArchitecture(String)
    case shapeMismatch(tensor: String, expected: [Int], actual: [Int])
    case tokenizer(String)

    public var description: String {
        switch self {
        case .missingTensor(let name):
            return "checkpoint has no tensor named '\(name)'"
        case .unsupportedDType(let tensor, let dtype):
            return "tensor '\(tensor)' has dtype \(dtype), which this reader does not decode"
        case .malformedSafetensors(let detail):
            return "safetensors file is malformed: \(detail)"
        case .configuration(let detail):
            return "config.json cannot be used as a tiny-K3 configuration: \(detail)"
        case .unsupportedArchitecture(let detail):
            return "this adapter does not implement: \(detail)"
        case .shapeMismatch(let tensor, let expected, let actual):
            return "tensor '\(tensor)' has shape \(actual), expected \(expected)"
        case .tokenizer(let detail):
            return "tokenizer: \(detail)"
        }
    }

    public var errorDescription: String? { description }
}
