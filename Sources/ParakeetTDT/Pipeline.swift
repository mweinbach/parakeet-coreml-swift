import CoreML
import Dispatch
import Foundation

/// 3-stage pipeline for long-form transcription: mel -> encoder -> decode.
///
/// The three stages hit different hardware (CPU / ANE-or-GPU / CPU), so
/// running them sequentially per chunk is leaving money on the table --
/// the encoder is idle while we extract the next mel, the CPU cores are
/// idle while we wait for the encoder. Pipelining collapses that to
/// ``max(stage_times) * num_chunks`` instead of ``sum(stage_times) *
/// num_chunks``.
///
/// Two bounded queues (``BlockingQueue``) sit between the stages and
/// provide backpressure so we never hold more than a few IOSurfaces
/// in-flight at once -- important because Core ML's ANE output buffers
/// come out of a pool with limited capacity.
enum Pipeline {

    struct Result {
        var tokens: [Int]
        var frames: [Int]
        var durations: [Int]
        var melElapsed: Double
        var encoderElapsed: Double
        var decodeElapsed: Double
    }

    static func run(
        chunks: [[Float]],
        featureExtractor: MelFeatureExtractor,
        runner: ModelRunner
    ) throws -> Result {
        if chunks.isEmpty {
            return Result(
                tokens: [], frames: [], durations: [],
                melElapsed: 0, encoderElapsed: 0, decodeElapsed: 0
            )
        }

        // Stage 1 -> Stage 2: mel features + original chunk index.
        let melQueue = BlockingQueue<MelItem>(capacity: 2)
        // Stage 2 -> Stage 3: encoder outputs.
        let encQueue = BlockingQueue<EncoderItem>(capacity: 2)

        let globalError = ErrorSlot()

        // --- Stage 1: CPU mel extraction ---
        let melQ = DispatchQueue(label: "parakeet.mel", qos: .userInitiated)
        let melTotal = AtomicDouble()
        melQ.async {
            for (i, chunk) in chunks.enumerated() {
                if globalError.hasError { break }
                let t0 = Date()
                let features = featureExtractor.extract(from: chunk)
                melTotal.add(Date().timeIntervalSince(t0))
                melQueue.put(MelItem(index: i, features: features))
            }
            melQueue.close()
        }

        // --- Stage 2: ANE / GPU / CPU encoder ---
        let encQ = DispatchQueue(label: "parakeet.encoder", qos: .userInitiated)
        let encTotal = AtomicDouble()
        encQ.async {
            while let mel = melQueue.take() {
                if globalError.hasError { break }
                do {
                    let t0 = Date()
                    let (hidden, mask) = try runner.runEncoder(
                        features: mel.features.mel,
                        mask: mel.features.attentionMask
                    )
                    encTotal.add(Date().timeIntervalSince(t0))
                    encQueue.put(
                        EncoderItem(index: mel.index, hidden: hidden, mask: mask)
                    )
                } catch {
                    globalError.set(error)
                    break
                }
            }
            encQueue.close()
        }

        // --- Stage 3: CPU decode loop (main thread) ---
        var tokens = [Int]()
        var frames = [Int]()
        var durations = [Int]()
        var totalFrameOffset = 0
        var decodeElapsed = 0.0

        while let item = encQueue.take() {
            if globalError.hasError { break }
            do {
                let decoded = try autoreleasepool {
                    try GreedyTDTDecoder.decode(
                        encoderHidden: item.hidden,
                        encoderMask: item.mask,
                        runner: runner
                    )
                }
                decodeElapsed += decoded.elapsedSeconds
                tokens.append(contentsOf: decoded.tokenIds)
                frames.append(
                    contentsOf: decoded.frameIndices.map { $0 + totalFrameOffset }
                )
                durations.append(contentsOf: decoded.durations)
                totalFrameOffset += item.hidden.shape[1].intValue
            } catch {
                globalError.set(error)
                break
            }
        }

        // Drain / signal upstream stages so they can unblock and exit.
        melQueue.close()
        encQueue.close()

        if let err = globalError.error {
            throw err
        }

        return Result(
            tokens: tokens,
            frames: frames,
            durations: durations,
            melElapsed: melTotal.value,
            encoderElapsed: encTotal.value,
            decodeElapsed: decodeElapsed
        )
    }
}

// MARK: - Helpers

private struct MelItem {
    let index: Int
    let features: MelFeatureExtractor.Features
}

private struct EncoderItem {
    let index: Int
    let hidden: MLMultiArray
    let mask: MLMultiArray
}

/// Capacity-bounded blocking MPSC queue. Thin wrapper around a Swift array
/// with two ``DispatchSemaphore``s: one for "slots free" (producer waits
/// on it when full), one for "items available" (consumer waits on it when
/// empty). A ``NSLock`` serializes the buffer itself.
private final class BlockingQueue<T>: @unchecked Sendable {
    private var buffer: [T] = []
    private var closed = false
    private let lock = NSLock()
    private let freeSlots: DispatchSemaphore
    private let itemsAvailable = DispatchSemaphore(value: 0)

    init(capacity: Int) {
        freeSlots = DispatchSemaphore(value: capacity)
    }

    func put(_ value: T) {
        freeSlots.wait()
        lock.lock()
        if closed {
            lock.unlock()
            freeSlots.signal()
            return
        }
        buffer.append(value)
        lock.unlock()
        itemsAvailable.signal()
    }

    func take() -> T? {
        itemsAvailable.wait()
        lock.lock()
        if buffer.isEmpty {
            // Closed and drained.
            lock.unlock()
            itemsAvailable.signal()  // wake any other waiters so they also exit
            return nil
        }
        let value = buffer.removeFirst()
        lock.unlock()
        freeSlots.signal()
        return value
    }

    func close() {
        lock.lock()
        let wasClosed = closed
        closed = true
        lock.unlock()
        if !wasClosed {
            // Wake all waiters -- takers find buffer empty + closed and return nil.
            itemsAvailable.signal()
            freeSlots.signal()
        }
    }
}

/// Simple error slot + lock for cross-thread error propagation.
private final class ErrorSlot: @unchecked Sendable {
    private var _error: Error?
    private let lock = NSLock()

    var error: Error? {
        lock.lock(); defer { lock.unlock() }
        return _error
    }

    var hasError: Bool {
        lock.lock(); defer { lock.unlock() }
        return _error != nil
    }

    func set(_ err: Error) {
        lock.lock(); defer { lock.unlock() }
        if _error == nil { _error = err }
    }
}

/// Cross-thread float accumulator for timing totals.
private final class AtomicDouble: @unchecked Sendable {
    private var _value: Double = 0
    private let lock = NSLock()
    var value: Double {
        lock.lock(); defer { lock.unlock() }
        return _value
    }
    func add(_ d: Double) {
        lock.lock(); defer { lock.unlock() }
        _value += d
    }
}
