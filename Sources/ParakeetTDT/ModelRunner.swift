import CoreML
import Foundation

/// Typed wrappers around the three converted Parakeet submodules so the
/// decoder loop stays readable. Each wrapper validates its input shape and
/// extracts the one or two outputs it cares about.
///
/// All three use the default ``MLPredictionOptions`` (no ane-only pinning
/// beyond what the ``MLModelConfiguration`` says).
public final class ModelRunner {
    private let encoder: MLModel
    private let decoder: MLModel
    private let joint: MLModel

    public struct EncoderShapes {
        public let batch: Int        // 1
        public let maxTime: Int      // 3000 mel frames -> 375 encoder frames
        public let numMelBins: Int   // 128
    }

    public let encoderShapes: EncoderShapes
    public let decoderHiddenLayers: Int
    public let decoderHiddenSize: Int
    public let blankTokenId: Int
    public let durations: [Int]
    public let vocabSize: Int
    public let maxSymbolsPerStep: Int

    public init(
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        encoderShapes: EncoderShapes,
        decoderHiddenLayers: Int,
        decoderHiddenSize: Int,
        blankTokenId: Int,
        durations: [Int],
        vocabSize: Int,
        maxSymbolsPerStep: Int
    ) {
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.encoderShapes = encoderShapes
        self.decoderHiddenLayers = decoderHiddenLayers
        self.decoderHiddenSize = decoderHiddenSize
        self.blankTokenId = blankTokenId
        self.durations = durations
        self.vocabSize = vocabSize
        self.maxSymbolsPerStep = maxSymbolsPerStep
    }

    // MARK: - Encoder

    /// Run the encoder on a mel feature chunk.
    /// Input: ``features`` is ``[time][mel]``, ``mask`` is ``[time]``.
    /// Both are padded / truncated to ``encoderShapes.maxTime`` before being
    /// handed to the model.
    public func runEncoder(
        features: [[Float]],
        mask: [Int32]
    ) throws -> (hidden: MLMultiArray, mask: MLMultiArray) {
        precondition(features.count == mask.count)
        let t = encoderShapes.maxTime
        let m = encoderShapes.numMelBins

        let feats = try MLMultiArray(
            shape: [NSNumber(value: 1), NSNumber(value: t), NSNumber(value: m)],
            dataType: .float32
        )
        let msk = try MLMultiArray(
            shape: [NSNumber(value: 1), NSNumber(value: t)],
            dataType: .int32
        )

        // Zero-init then copy.
        feats.withUnsafeMutableBytes { raw, _ in
            memset(raw.baseAddress!, 0, raw.count)
        }
        msk.withUnsafeMutableBytes { raw, _ in
            memset(raw.baseAddress!, 0, raw.count)
        }

        let copyT = min(features.count, t)
        let fPtr = UnsafeMutablePointer<Float32>(
            OpaquePointer(feats.dataPointer)
        )
        for ti in 0..<copyT {
            let row = features[ti]
            let dstBase = fPtr.advanced(by: ti * m)
            let copyM = min(row.count, m)
            row.withUnsafeBufferPointer { src in
                memcpy(dstBase, src.baseAddress!, copyM * MemoryLayout<Float32>.size)
            }
        }
        let mPtr = UnsafeMutablePointer<Int32>(
            OpaquePointer(msk.dataPointer)
        )
        for ti in 0..<copyT { mPtr[ti] = mask[ti] }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: feats),
            "attention_mask": MLFeatureValue(multiArray: msk),
        ])
        let out = try encoder.prediction(from: provider)

        guard let hidden = out.featureValue(for: "encoder_hidden")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "encoder_hidden") }
        guard let outMask = out.featureValue(for: "encoder_mask")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "encoder_mask") }
        return (hidden, outMask)
    }

    // MARK: - Decoder (one step)

    public func runDecoderStep(
        inputIds: MLMultiArray,
        hidden: MLMultiArray,
        cell: MLMultiArray
    ) throws -> (decoderHidden: MLMultiArray, nextHidden: MLMultiArray, nextCell: MLMultiArray) {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "hidden": MLFeatureValue(multiArray: hidden),
            "cell": MLFeatureValue(multiArray: cell),
        ])
        let out = try decoder.prediction(from: provider)
        guard let dh = out.featureValue(for: "decoder_hidden")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "decoder_hidden") }
        guard let nh = out.featureValue(for: "next_hidden")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "next_hidden") }
        guard let nc = out.featureValue(for: "next_cell")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "next_cell") }
        return (dh, nh, nc)
    }

    // MARK: - Joint

    public func runJoint(
        encoderFrame: MLMultiArray,
        decoderState: MLMultiArray
    ) throws -> (tokenLogits: MLMultiArray, durationLogits: MLMultiArray) {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_frame": MLFeatureValue(multiArray: encoderFrame),
            "decoder_state": MLFeatureValue(multiArray: decoderState),
        ])
        let out = try joint.prediction(from: provider)
        guard let tl = out.featureValue(for: "token_logits")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "token_logits") }
        guard let dl = out.featureValue(for: "duration_logits")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "duration_logits") }
        return (tl, dl)
    }
}
