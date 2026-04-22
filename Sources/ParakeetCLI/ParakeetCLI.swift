import ArgumentParser
import Foundation
import ParakeetTDT

@main
struct Parakeet: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "parakeet",
        abstract: "Transcribe audio with Parakeet TDT on Core ML.",
        subcommands: [Transcribe.self],
        defaultSubcommand: Transcribe.self
    )
}

struct Transcribe: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe an audio file."
    )

    @Argument(
        help: "Path to audio file (wav / flac / mp3 / m4a / ...). Mono-downmixed and resampled to 16 kHz automatically."
    )
    var audio: String

    @Option(
        name: [.short, .customLong("models")],
        help: "Directory containing encoder.mlpackage, decoder.mlpackage, joint.mlpackage, tokenizer.json."
    )
    var models: String

    @Option(
        name: .customLong("compute-units"),
        help: "Core ML scheduler target: ane (Neural Engine + CPU), gpu (GPU + CPU), cpu (CPU only), all. Default: ane."
    )
    var computeUnits: String = "ane"

    @Flag(
        name: .customLong("delete-source-after-compile"),
        help: "Delete the source .mlpackage after successful compile. Halves peak disk usage at the cost of having to re-download if you clear the cache."
    )
    var deleteSourceAfterCompile: Bool = false

    @Flag(
        name: .customLong("show-timing"),
        help: "Print detailed timing breakdown after the transcript."
    )
    var showTiming: Bool = false

    @Option(
        name: .customLong("max-seconds"),
        help: "Cap audio length. Default: full file."
    )
    var maxSeconds: Double? = nil

    func validate() throws {
        guard ["ane", "gpu", "cpu", "all"].contains(computeUnits) else {
            throw ValidationError(
                "compute-units must be one of: ane, gpu, cpu, all (got \(computeUnits))"
            )
        }
    }

    func run() throws {
        let units: ParakeetComputeUnits = {
            switch computeUnits {
            case "gpu": return .gpu
            case "cpu": return .cpu
            case "all": return .all
            default:    return .ane
            }
        }()

        let modelsURL = URL(fileURLWithPath: models, isDirectory: true)
        let audioURL = URL(fileURLWithPath: audio)

        FileHandle.standardError.write(Data(
            "Loading models (compute: \(computeUnits))...\n".utf8
        ))
        let loadStart = Date()
        let transcriber = try ParakeetTranscriber(
            modelsRoot: modelsURL,
            computeUnits: units,
            deleteSourceAfterCompile: deleteSourceAfterCompile
        )
        let loadSecs = Date().timeIntervalSince(loadStart)
        FileHandle.standardError.write(Data(
            "Models ready in \(String(format: "%.2f", loadSecs)) s\n".utf8
        ))

        let samples: [Float]
        if let cap = maxSeconds {
            let raw = try AudioLoader.loadMono16k(at: audioURL)
            let capCount = min(raw.count, Int(cap * Double(transcriber.sampleRate)))
            samples = Array(raw[0..<capCount])
        } else {
            samples = try AudioLoader.loadMono16k(at: audioURL)
        }
        let seconds = Double(samples.count) / Double(transcriber.sampleRate)
        FileHandle.standardError.write(Data(
            "Transcribing \(String(format: "%.2f", seconds)) s of audio...\n".utf8
        ))

        let result = try transcriber.transcribe(samples: samples)

        print(result.text)

        if showTiming {
            var err = StderrStream()
            print(
                """

                ----- timing -----
                audio:     \(String(format: "%.3f", result.audioDurationSeconds)) s
                inference: \(String(format: "%.3f", result.inferenceDurationSeconds)) s
                RTFx:      \(String(format: "%.2fx", result.rtfx))
                tokens:    \(result.tokenIds.count)
                """,
                to: &err
            )
        }
    }
}

/// Lightweight stderr writer so we can redirect timing output away from
/// stdout (which carries the transcript).
private struct StderrStream: TextOutputStream {
    mutating func write(_ string: String) {
        FileHandle.standardError.write(Data(string.utf8))
    }
}
