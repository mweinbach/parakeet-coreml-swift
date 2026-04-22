import CoreML
import Foundation

/// Typed wrappers around the three converted Parakeet submodules.
///
/// All input `MLMultiArray`s are allocated exactly once and reused across
/// every prediction call. The ``MLFeatureProvider``s handed to Core ML are
/// also cached (see ``FeatureBag``) so each decode step costs one
/// ``MLModel.prediction(from:)`` call and nothing else on the Swift side.
public final class ModelRunner {
    private let encoder: MLModel
    private let decoder: MLModel
    private let joint: MLModel

    public struct EncoderShapes {
        public let batch: Int       // 1
        public let maxTime: Int     // 3000 mel frames (traced)
        public let numMelBins: Int  // 128
    }

    public let encoderShapes: EncoderShapes
    public let decoderHiddenLayers: Int
    public let decoderHiddenSize: Int
    public let blankTokenId: Int
    public let durations: [Int]
    public let vocabSize: Int
    public let maxSymbolsPerStep: Int

    // MARK: - Persistent input buffers

    /// Encoder inputs. Reused across chunks.
    public let encoderFeatures: MLMultiArray
    public let encoderMask: MLMultiArray
    let encoderInputs: FeatureBag

    /// Decoder inputs. Reused across every step of the greedy decode.
    public let decoderInputIds: MLMultiArray
    public let decoderHidden: MLMultiArray
    public let decoderCell: MLMultiArray
    let decoderInputs: FeatureBag

    /// Joint inputs. Reused across every step.
    public let jointEncoderFrame: MLMultiArray
    public let jointDecoderState: MLMultiArray
    let jointInputs: FeatureBag

    /// Prediction options shared by every submodule. Intentionally empty
    /// (default config); kept around so we can flip knobs in one place.
    let predictionOptions = MLPredictionOptions()

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
    ) throws {
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

        // --- Encoder inputs ---
        self.encoderFeatures = try MLMultiArray(
            shape: [
                NSNumber(value: 1),
                NSNumber(value: encoderShapes.maxTime),
                NSNumber(value: encoderShapes.numMelBins),
            ],
            dataType: .float32
        )
        self.encoderMask = try MLMultiArray(
            shape: [NSNumber(value: 1), NSNumber(value: encoderShapes.maxTime)],
            dataType: .int32
        )
        self.encoderInputs = FeatureBag([
            "input_features": MLFeatureValue(multiArray: encoderFeatures),
            "attention_mask": MLFeatureValue(multiArray: encoderMask),
        ])

        // --- Decoder inputs ---
        self.decoderInputIds = try MLMultiArray(shape: [1, 1], dataType: .int32)
        self.decoderHidden = try MLMultiArray(
            shape: [
                NSNumber(value: decoderHiddenLayers), 1,
                NSNumber(value: decoderHiddenSize),
            ],
            dataType: .float32
        )
        self.decoderCell = try MLMultiArray(
            shape: [
                NSNumber(value: decoderHiddenLayers), 1,
                NSNumber(value: decoderHiddenSize),
            ],
            dataType: .float32
        )
        self.decoderInputs = FeatureBag([
            "input_ids": MLFeatureValue(multiArray: decoderInputIds),
            "hidden": MLFeatureValue(multiArray: decoderHidden),
            "cell": MLFeatureValue(multiArray: decoderCell),
        ])

        // --- Joint inputs ---
        self.jointEncoderFrame = try MLMultiArray(
            shape: [1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )
        self.jointDecoderState = try MLMultiArray(
            shape: [1, NSNumber(value: decoderHiddenSize)],
            dataType: .float32
        )
        self.jointInputs = FeatureBag([
            "encoder_frame": MLFeatureValue(multiArray: jointEncoderFrame),
            "decoder_state": MLFeatureValue(multiArray: jointDecoderState),
        ])
    }

    // MARK: - Encoder

    /// Write mel features + mask into the reused input buffers, run the
    /// encoder, return the output `encoder_hidden` / `encoder_mask` arrays.
    public func runEncoder(
        features: [[Float]],
        mask: [Int32]
    ) throws -> (hidden: MLMultiArray, mask: MLMultiArray) {
        let t = encoderShapes.maxTime
        let m = encoderShapes.numMelBins

        let fPtr = encoderFeatures.dataPointer
            .bindMemory(to: Float32.self, capacity: t * m)
        memset(fPtr, 0, t * m * MemoryLayout<Float32>.size)
        let copyT = min(features.count, t)
        for ti in 0..<copyT {
            let row = features[ti]
            let copyM = min(row.count, m)
            row.withUnsafeBufferPointer { src in
                memcpy(
                    fPtr.advanced(by: ti * m),
                    src.baseAddress!,
                    copyM * MemoryLayout<Float32>.size
                )
            }
        }

        let mPtr = encoderMask.dataPointer
            .bindMemory(to: Int32.self, capacity: t)
        memset(mPtr, 0, t * MemoryLayout<Int32>.size)
        for ti in 0..<min(mask.count, t) { mPtr[ti] = mask[ti] }

        let out = try encoder.prediction(
            from: encoderInputs, options: predictionOptions
        )
        guard let hidden = out.featureValue(for: "encoder_hidden")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "encoder_hidden") }
        guard let outMask = out.featureValue(for: "encoder_mask")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "encoder_mask") }
        return (hidden, outMask)
    }

    // MARK: - Decoder (one step)

    /// The caller has already written the desired input_ids / hidden /
    /// cell values into the persistent buffers. We just dispatch the
    /// prediction and unpack the three outputs.
    public func runDecoderStep() throws -> (
        decoderHidden: MLMultiArray,
        nextHidden: MLMultiArray,
        nextCell: MLMultiArray
    ) {
        let out = try decoder.prediction(
            from: decoderInputs, options: predictionOptions
        )
        guard let dh = out.featureValue(for: "decoder_hidden")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "decoder_hidden") }
        guard let nh = out.featureValue(for: "next_hidden")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "next_hidden") }
        guard let nc = out.featureValue(for: "next_cell")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "next_cell") }
        return (dh, nh, nc)
    }

    // MARK: - Joint

    /// As with ``runDecoderStep``, the caller writes the input buffers
    /// directly.
    public func runJoint() throws -> (
        tokenLogits: MLMultiArray,
        durationLogits: MLMultiArray
    ) {
        let out = try joint.prediction(
            from: jointInputs, options: predictionOptions
        )
        guard let tl = out.featureValue(for: "token_logits")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "token_logits") }
        guard let dl = out.featureValue(for: "duration_logits")?.multiArrayValue
        else { throw ParakeetError.missingOutput(name: "duration_logits") }
        return (tl, dl)
    }
}
