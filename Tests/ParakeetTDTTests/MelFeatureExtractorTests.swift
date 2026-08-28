import XCTest
@testable import ParakeetTDT

/// Pins the `validSamples` behaviour of the mel extractor.
///
/// The regression this guards: a short utterance zero-padded into the encoder's
/// fixed 30-second chunk had its per-bin normalisation statistics computed over
/// the padding too. For a quiet real-microphone capture the statistics then
/// described the silence rather than the speech, and the model decoded nothing —
/// dictation silently produced an empty transcript (found 2026-08-28, on audio
/// Apple's recogniser read fine).
final class MelFeatureExtractorTests: XCTestCase {
    /// A deterministic speech-ish test signal: a few tones with an envelope.
    private func signal(seconds: Double, sampleRate: Int = 16_000) -> [Float] {
        let n = Int(seconds * Double(sampleRate))
        return (0..<n).map { i in
            let t = Float(i) / Float(sampleRate)
            let envelope = 0.05 * (1 + sin(2 * .pi * 3 * t)) / 2
            return envelope * (sin(2 * .pi * 220 * t) + 0.5 * sin(2 * .pi * 1300 * t))
        }
    }

    func testPaddingAmountDoesNotChangeValidFrames() throws {
        let extractor = try MelFeatureExtractor()
        let short = signal(seconds: 1.5)
        let pad10 = short + [Float](repeating: 0, count: 10 * 16_000 - short.count)
        let pad30 = short + [Float](repeating: 0, count: 30 * 16_000 - short.count)

        let a = extractor.extract(from: pad10, validSamples: short.count)
        let b = extractor.extract(from: pad30, validSamples: short.count)

        // The frames covering real audio must not depend on how much padding
        // follows them. This is exactly what the original bug violated: the
        // normalisation statistics included the padding, so more padding meant
        // different (and, for quiet audio, useless) features.
        let valid = short.count / extractor.hopLength + 1
        XCTAssertGreaterThan(valid, 0)
        for t in 0..<valid {
            for i in 0..<extractor.numMelFilters {
                XCTAssertEqual(a.mel[t][i], b.mel[t][i],
                               "frame \(t) bin \(i) depends on the padding length")
            }
        }
    }

    func testValidRegionIsNormalised() throws {
        let extractor = try MelFeatureExtractor()
        let short = signal(seconds: 1.5)
        let padded = short + [Float](repeating: 0, count: 30 * 16_000 - short.count)

        let features = extractor.extract(from: padded, validSamples: short.count)
        let valid = short.count / extractor.hopLength + 1

        // Statistics are computed over the valid region, so per bin its mean is
        // ~0 and its standard deviation ~1. With the padding included (the bug)
        // the valid region's own mean is far from zero.
        for i in 0..<extractor.numMelFilters {
            let column = (0..<valid).map { features.mel[$0][i] }
            let mean = column.reduce(0, +) / Float(valid)
            XCTAssertEqual(mean, 0, accuracy: 0.01, "bin \(i) mean")
        }
    }

    func testMaskMarksThePadding() throws {
        let extractor = try MelFeatureExtractor()
        let short = signal(seconds: 1.5)
        let padded = short + [Float](repeating: 0, count: 30 * 16_000 - short.count)

        let features = extractor.extract(from: padded, validSamples: short.count)
        let valid = features.attentionMask.filter { $0 == 1 }.count

        // Same frame arithmetic as the extractor: length / hop + 1.
        XCTAssertEqual(valid, short.count / extractor.hopLength + 1)
        // Valid frames lead, padding trails — no interleaving.
        XCTAssertEqual(Array(features.attentionMask.prefix(valid)),
                       [Int32](repeating: 1, count: valid))
        XCTAssertEqual(Array(features.attentionMask.dropFirst(valid)),
                       [Int32](repeating: 0, count: features.numFrames - valid))
    }

    func testDefaultTreatsWholeWaveformAsValid() throws {
        let extractor = try MelFeatureExtractor()
        let features = extractor.extract(from: signal(seconds: 2))
        XCTAssertTrue(features.attentionMask.allSatisfy { $0 == 1 })
    }
}
