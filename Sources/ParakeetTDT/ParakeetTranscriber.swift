import CoreML
import Foundation

/// End-to-end Parakeet TDT transcriber.
///
/// Expects a directory laid out like this (matching the HuggingFace repo):
///
/// ```
/// modelsRoot/
///   encoder.mlpackage/    (or encoder.mlmodelc, if already compiled)
///   decoder.mlpackage/
///   joint.mlpackage/
///   tokenizer.json
/// ```
///
/// The first call compiles each ``.mlpackage`` into a cached ``.mlmodelc``.
/// Subsequent launches find the cached compiled bundle and skip the compile
/// step.
///
/// Usage:
///
/// ```swift
/// let t = try ParakeetTranscriber(modelsRoot: url)
/// let transcription = try t.transcribe(audioURL: audio)
/// print(transcription.text)
/// ```
public final class ParakeetTranscriber {
    public let computeUnits: ParakeetComputeUnits
    public let chunkMelFrames: Int    // must match the encoder's traced shape
    public let sampleRate: Int

    private let runner: ModelRunner
    private let tokenizer: Tokenizer
    private let featureExtractor: MelFeatureExtractor
    private let cacheURL: URL

    /// Load the transcriber. Compiles any `.mlpackage`s that aren't already
    /// in the cache. Use `deleteSourceAfterCompile: true` to drop the raw
    /// `.mlpackage` from disk once compilation succeeds (halves peak disk
    /// usage on space-constrained devices).
    public init(
        modelsRoot: URL,
        computeUnits: ParakeetComputeUnits = .ane,
        chunkMelFrames: Int = 3000,
        sampleRate: Int = 16_000,
        deleteSourceAfterCompile: Bool = false,
        cacheDirectory: URL? = nil
    ) throws {
        self.computeUnits = computeUnits
        self.chunkMelFrames = chunkMelFrames
        self.sampleRate = sampleRate

        let cache = ModelCache(
            cacheDirectory: cacheDirectory,
            deleteSourceAfterCompile: deleteSourceAfterCompile
        )
        self.cacheURL = cache.cacheDirectory

        let encoderURL = try ParakeetTranscriber.resolveModel(
            under: modelsRoot, named: "encoder"
        )
        let decoderURL = try ParakeetTranscriber.resolveModel(
            under: modelsRoot, named: "decoder"
        )
        let jointURL = try ParakeetTranscriber.resolveModel(
            under: modelsRoot, named: "joint"
        )
        let tokenizerURL = modelsRoot.appendingPathComponent("tokenizer.json")

        let encCompiled = try cache.compiledURL(for: encoderURL)
        let decCompiled = try cache.compiledURL(for: decoderURL)
        let joiCompiled = try cache.compiledURL(for: jointURL)

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.mlComputeUnits

        let encoder = try MLModel(contentsOf: encCompiled, configuration: config)
        let decoder = try MLModel(contentsOf: decCompiled, configuration: config)
        let joint = try MLModel(contentsOf: joiCompiled, configuration: config)

        // Decoder stateful sizes are encoded in the spec's hidden / cell
        // input shapes: [num_layers, 1, hidden].
        let (decLayers, decHidden) = ParakeetTranscriber.readDecoderStateShape(
            from: decoder
        )

        self.runner = try ModelRunner(
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            encoderShapes: ModelRunner.EncoderShapes(
                batch: 1, maxTime: chunkMelFrames, numMelBins: 128
            ),
            decoderHiddenLayers: decLayers,
            decoderHiddenSize: decHidden,
            blankTokenId: 8192,
            durations: [0, 1, 2, 3, 4],
            vocabSize: 8193,
            maxSymbolsPerStep: 10
        )
        self.tokenizer = try Tokenizer(tokenizerJSONURL: tokenizerURL)
        self.featureExtractor = try MelFeatureExtractor(
            sampleRate: sampleRate,
            hopLength: 160,
            winLength: 400,
            nFFT: 512,
            numMelFilters: 128,
            preemphasis: 0.97
        )
    }

    // MARK: - High-level transcription

    /// Transcribe a full audio file. Long files are chunked into
    /// non-overlapping ``chunkMelFrames * hopLength / sampleRate``-second
    /// windows (30 s with the default 3000 mel frames). Token streams from
    /// every chunk are concatenated, then detokenized in one pass.
    public func transcribe(audioURL: URL) throws -> Transcription {
        let audio = try AudioLoader.loadMono16k(at: audioURL)
        return try transcribe(samples: audio)
    }

    /// Transcribe an already-loaded mono `Float` buffer at ``sampleRate``.
    public func transcribe(samples: [Float]) throws -> Transcription {
        let audioDuration = Double(samples.count) / Double(sampleRate)
        let chunkSamples = chunkMelFrames * featureExtractor.hopLength

        var tokens = [Int]()
        var frames = [Int]()
        var durations = [Int]()
        var totalFrameOffset = 0

        var melElapsed = 0.0
        var encoderElapsed = 0.0
        var decodeElapsed = 0.0
        let start = Date()

        var cursor = 0
        while cursor < samples.count {
            let end = min(cursor + chunkSamples, samples.count)
            var chunk = Array(samples[cursor..<end])
            if chunk.count < chunkSamples {
                chunk.append(
                    contentsOf: [Float](repeating: 0, count: chunkSamples - chunk.count)
                )
            }

            let tMel = Date()
            let features = featureExtractor.extract(from: chunk)
            melElapsed += Date().timeIntervalSince(tMel)

            let tEnc = Date()
            let (encHidden, encMask) = try runner.runEncoder(
                features: features.mel, mask: features.attentionMask
            )
            encoderElapsed += Date().timeIntervalSince(tEnc)

            let decoded = try GreedyTDTDecoder.decode(
                encoderHidden: encHidden,
                encoderMask: encMask,
                runner: runner
            )
            decodeElapsed += decoded.elapsedSeconds

            tokens.append(contentsOf: decoded.tokenIds)
            frames.append(contentsOf: decoded.frameIndices.map { $0 + totalFrameOffset })
            durations.append(contentsOf: decoded.durations)
            totalFrameOffset += encHidden.shape[1].intValue
            cursor += chunkSamples
        }

        let tDetok = Date()
        let text = tokenizer.decode(tokens, skipSpecial: true)
        let detokElapsed = Date().timeIntervalSince(tDetok)

        let elapsed = Date().timeIntervalSince(start)
        return Transcription(
            text: text,
            tokenIds: tokens,
            frameIndices: frames,
            durations: durations,
            audioDurationSeconds: audioDuration,
            inferenceDurationSeconds: elapsed,
            timing: TranscriptionTiming(
                melExtract: melElapsed,
                encoder: encoderElapsed,
                decoderLoop: decodeElapsed,
                detokenize: detokElapsed
            )
        )
    }

    // MARK: - Helpers

    /// Look for ``<name>.mlmodelc`` (preferred; already compiled) then
    /// ``<name>.mlpackage`` inside ``modelsRoot``.
    private static func resolveModel(
        under root: URL, named: String
    ) throws -> URL {
        let candidates = [
            root.appendingPathComponent("\(named).mlmodelc"),
            root.appendingPathComponent("\(named).mlpackage"),
        ]
        for url in candidates {
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        throw ParakeetError.modelNotFound(url: candidates[1])
    }

    /// Sniff the decoder's ``hidden`` input shape to figure out the LSTM's
    /// (num_layers, hidden_size). The spec records it as the symbolic
    /// shape ``[num_layers, 1, hidden]``.
    private static func readDecoderStateShape(
        from model: MLModel
    ) -> (layers: Int, hidden: Int) {
        if let desc = model.modelDescription.inputDescriptionsByName["hidden"],
           let con = desc.multiArrayConstraint
        {
            let shape = con.shape.map(\.intValue)
            if shape.count == 3 {
                return (shape[0], shape[2])
            }
        }
        return (2, 640)  // Parakeet TDT 0.6B defaults.
    }

    /// Cache directory where compiled `.mlmodelc`s live. Exposed so callers
    /// can clear it if they want to force a recompile or free disk.
    public var compiledCacheDirectory: URL { cacheURL }
}
