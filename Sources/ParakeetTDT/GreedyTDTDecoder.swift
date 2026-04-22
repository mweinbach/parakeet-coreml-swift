import Accelerate
import CoreML
import Foundation

/// Greedy TDT (Token-and-Duration Transducer) decoder.
///
/// Ports the Python ``greedy_tdt_decode`` loop line-for-line:
///   - Maintain two LSTM state tensors (``hidden``, ``cell``) and the last
///     emitted token.
///   - For each encoder frame ``t``:
///     - Call decoder once (or reuse cached output if we just emitted blank).
///     - Call joint(encoder[t], decoder_state).
///     - Argmax over token logits; argmax over duration logits.
///     - If token is blank: advance t by ``max(duration, 1)``.
///     - Otherwise: emit token, advance t by duration if > 0, else keep t
///       and try again, capped by ``maxSymbolsPerStep`` to avoid infinite
///       loops on "always non-blank, always duration=0" degenerate cases.
public enum GreedyTDTDecoder {

    public struct Output {
        public let tokenIds: [Int]
        public let frameIndices: [Int]
        public let durations: [Int]
    }

    public static func decode(
        encoderHidden: MLMultiArray,
        encoderMask: MLMultiArray,
        runner: ModelRunner
    ) throws -> Output {
        // encoderHidden shape: [1, T, H]. Extract validFrames from the mask.
        guard encoderHidden.shape.count == 3 else {
            throw ParakeetError.unexpectedOutputShape(
                name: "encoder_hidden",
                got: encoderHidden.shape.map(\.intValue),
                expected: "[1, T, hidden]"
            )
        }
        let hiddenSize = encoderHidden.shape[2].intValue
        let tMax = encoderHidden.shape[1].intValue

        let maskPtr = UnsafeMutablePointer<Int32>(
            OpaquePointer(encoderMask.dataPointer)
        )
        var validFrames = 0
        for i in 0..<encoderMask.shape.last!.intValue {
            validFrames += Int(maskPtr[i])
        }
        validFrames = min(validFrames, tMax)

        let hLayers = runner.decoderHiddenLayers
        let hSize = runner.decoderHiddenSize
        let blank = runner.blankTokenId
        let durations = runner.durations
        let maxSym = runner.maxSymbolsPerStep

        // Reusable MLMultiArrays for the decoder loop.
        let hidden = try MLMultiArray(
            shape: [NSNumber(value: hLayers), 1, NSNumber(value: hSize)],
            dataType: .float32
        )
        let cell = try MLMultiArray(
            shape: [NSNumber(value: hLayers), 1, NSNumber(value: hSize)],
            dataType: .float32
        )
        let inputIds = try MLMultiArray(
            shape: [1, 1],
            dataType: .int32
        )
        let encoderFrame = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenSize)],
            dataType: .float32
        )
        let decoderState = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenSize)],
            dataType: .float32
        )
        zero(hidden)
        zero(cell)

        let idsPtr = UnsafeMutablePointer<Int32>(OpaquePointer(inputIds.dataPointer))
        idsPtr[0] = Int32(blank)

        let encFramePtr = UnsafeMutablePointer<Float32>(OpaquePointer(encoderFrame.dataPointer))
        let decStatePtr = UnsafeMutablePointer<Float32>(OpaquePointer(decoderState.dataPointer))
        let encHiddenPtr = UnsafeMutablePointer<Float32>(OpaquePointer(encoderHidden.dataPointer))

        var tokens = [Int]()
        var frameIdx = [Int]()
        var durationOut = [Int]()
        var decoderCache: MLMultiArray? = nil

        var t = 0
        while t < validFrames {
            // Copy encoder_hidden[0, t, :] into the encoder_frame buffer.
            memcpy(
                encFramePtr,
                encHiddenPtr.advanced(by: t * hiddenSize),
                hiddenSize * MemoryLayout<Float32>.size
            )

            var symbols = 0
            var advanced = false
            while symbols < maxSym {
                let cached = decoderCache
                let decHidden: MLMultiArray
                if cached == nil || idsPtr[0] != Int32(blank) {
                    let out = try runner.runDecoderStep(
                        inputIds: inputIds, hidden: hidden, cell: cell
                    )
                    decHidden = out.decoderHidden
                    copyMultiArray(from: out.nextHidden, to: hidden)
                    copyMultiArray(from: out.nextCell, to: cell)
                    decoderCache = decHidden
                } else {
                    decHidden = cached!
                }

                // decoder_hidden shape: [1, U, hidden]. Take the last time step.
                let decShape = decHidden.shape.map(\.intValue)
                let decU = decShape.count >= 3 ? decShape[decShape.count - 2] : 1
                let lastT = decU - 1
                let decHiddenPtr = UnsafeMutablePointer<Float32>(
                    OpaquePointer(decHidden.dataPointer)
                )
                memcpy(
                    decStatePtr,
                    decHiddenPtr.advanced(by: lastT * hiddenSize),
                    hiddenSize * MemoryLayout<Float32>.size
                )

                let jointOut = try runner.runJoint(
                    encoderFrame: encoderFrame,
                    decoderState: decoderState
                )
                let tokLogits = jointOut.tokenLogits
                let durLogits = jointOut.durationLogits
                let tokenId = argmax(tokLogits)
                let durIdx = argmax(durLogits)
                let duration = durations[durIdx]

                if tokenId == blank {
                    t += max(duration, 1)
                    advanced = true
                    break
                }

                tokens.append(tokenId)
                frameIdx.append(t)
                durationOut.append(duration)
                idsPtr[0] = Int32(tokenId)
                symbols += 1
                if duration > 0 {
                    t += duration
                    advanced = true
                    break
                }
            }
            if !advanced { t += 1 }
        }

        return Output(
            tokenIds: tokens,
            frameIndices: frameIdx,
            durations: durationOut
        )
    }

    // MARK: - helpers

    @inline(__always)
    static func zero(_ arr: MLMultiArray) {
        arr.withUnsafeMutableBytes { raw, _ in
            memset(raw.baseAddress!, 0, raw.count)
        }
    }

    @inline(__always)
    static func copyMultiArray(from src: MLMultiArray, to dst: MLMultiArray) {
        src.withUnsafeBytes { sBuf in
            dst.withUnsafeMutableBytes { dBuf, _ in
                let n = min(sBuf.count, dBuf.count)
                memcpy(dBuf.baseAddress!, sBuf.baseAddress!, n)
            }
        }
    }

    @inline(__always)
    static func argmax(_ array: MLMultiArray) -> Int {
        let n = vDSP_Length(array.count)
        let ptr = UnsafePointer<Float32>(OpaquePointer(array.dataPointer))
        var maxVal: Float = 0
        var idx: vDSP_Length = 0
        vDSP_maxvi(ptr, 1, &maxVal, &idx, n)
        return Int(idx)
    }
}
