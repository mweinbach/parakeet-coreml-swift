# parakeet-coreml-swift

Swift package for on-device speech-to-text using NVIDIA's Parakeet TDT
0.6B v3 compiled to Core ML. Runs on macOS 14+ and iOS 17+, with a single
entry point:

```swift
import ParakeetTDT

let transcriber = try ParakeetTranscriber(
    modelsRoot: URL(fileURLWithPath: "/path/to/parakeet-coreml"),
    computeUnits: .ane     // or .gpu, .cpu, .all
)
let result = try transcriber.transcribe(audioURL: audio)
print(result.text)
print("\(result.rtfx) x realtime")
```

The first `transcribe` call compiles each `.mlpackage` into a cached
`.mlmodelc` bundle under `~/Library/Caches/com.parakeet-tdt/mlmodelc`;
subsequent launches skip compilation.

## Models

Pre-converted Core ML packages are published at:

- [`mweinbach1/parakeet-tdt-0.6b-v3-coreml`](https://huggingface.co/mweinbach1/parakeet-tdt-0.6b-v3-coreml)
  (encoder 4-bit per-grouped-channel palettized, decoder / joint fp16)

Download them however you like (`hf download ...`, `git lfs clone ...`,
fetch-on-first-launch, etc.) and hand the resulting directory to
`ParakeetTranscriber(modelsRoot:)`.

Expected layout:

```
modelsRoot/
├── encoder.mlpackage/     (or encoder.mlmodelc, pre-compiled)
├── decoder.mlpackage/
├── joint.mlpackage/
└── tokenizer.json
```

## Compute units

| Option | Maps to | Notes |
|---|---|---|
| `.ane` (default) | `cpuAndNeuralEngine` | Best power/latency balance on iOS. |
| `.gpu` | `cpuAndGPU` | Fastest *per-op* on M-class Macs; see note below. |
| `.cpu` | `cpuOnly` | Portable fallback. |
| `.all` | `all` | Let Core ML pick. |

The scheduler decides per-op what runs where. You can flip between these
at load time without rebuilding the `.mlpackage`.

**Note on GPU vs ANE from Swift:** the 4-bit palettized encoder's *per-op*
GPU throughput is ~3x ANE on Apple silicon, but the small-batch decoder
loop (~3000 tiny joint calls per 17.5 min of audio) amortizes Core ML's
per-prediction dispatch overhead differently on each target. From Python's
`coremltools.MLModel.predict`, GPU wins overall (~414× RTFx vs ~245× ANE).
From Swift's `MLModel.prediction(from:)` the same `.mlpackage`s come in at
~115x (GPU) vs ~155x (ANE) RTFx because dispatch overhead on small
predictions dominates. If you're doing long-form batch transcription and
can afford it, compile + host the models in a long-lived process so the
warm-up cost amortizes away, and experiment with both targets.

## Benchmarks on test_audio.mp3 (17.5 min, 16 kHz mono, M4 Max)

Same `.mlpackage` artifacts, just different `--compute-units`:

| Target | Inference | RTFx | WER vs fp16 ref |
|---|---:|---:|---:|
| ANE | 6.78 s | 154.6x | 3.53% |
| GPU | 9.11 s | 115.1x | (comparable) |

Python numbers for the exact same `.mlpackage`s, for reference:

| Target | Inference | RTFx |
|---|---:|---:|
| ANE | 4.27 s | 245.5x |
| GPU | 2.53 s | 414.2x |

Swift is currently ~35% slower than Python's `coremltools` on this
workload, mostly because we allocate fresh `MLMultiArray`s per decoder
step. PRs welcome.

## CLI

```bash
swift run parakeet transcribe path/to/audio.wav \
    --models /path/to/parakeet-coreml \
    --compute-units ane \
    --show-timing
```

## How this package is organised

- `ParakeetTranscriber` -- top-level API. Owns model loading + chunked
  greedy decode.
- `ModelCache` -- `.mlpackage` -> `.mlmodelc` compilation + caching. Has
  an opt-in `deleteSourceAfterCompile` flag for disk-constrained setups.
- `AudioLoader` -- AVFoundation-based any-format -> 16 kHz mono `[Float]`.
- `MelFeatureExtractor` -- matches HuggingFace's
  `ParakeetFeatureExtractor` numerically. Implements the full pipeline in
  `Accelerate`/`vDSP`: preemphasis -> frame + Hann -> STFT -> |X|^2 ->
  Slaney mel (128 filters) -> log -> per-feature normalization.
- `Tokenizer` -- minimal decode-only SentencePiece BPE tokenizer.
- `ModelRunner` + `GreedyTDTDecoder` -- encoder / LSTM decoder / joint
  wrappers, plus the port of the Python greedy TDT loop.

## One-time compile, delete source pattern

If your deployment target is disk-constrained and you'd like to ship the
`.mlpackage` bundle in the app's resources but only hold on to the
compiled form:

```swift
let transcriber = try ParakeetTranscriber(
    modelsRoot: packagedURL,
    computeUnits: .ane,
    deleteSourceAfterCompile: true    // removes `.mlpackage`s after compile
)
```

The cache lives at `compiledCacheDirectory`; delete it to force a rebuild
on next launch.

## License

Apache 2.0 for this Swift package. The underlying Parakeet TDT model is
CC-BY-4.0 from NVIDIA; see the HuggingFace model card for full terms.
