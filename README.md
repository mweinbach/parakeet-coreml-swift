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
| `.gpu` | `cpuAndGPU` | Fastest on M-class Macs for this model. |
| `.cpu` | `cpuOnly` | Portable fallback. |
| `.all` | `all` | Let Core ML pick. |

The scheduler decides per-op what runs where. You can flip between these
at load time without rebuilding the `.mlpackage`.

## Benchmarks on `test_audio.mp3` (17.5 min, 16 kHz mono, M5 Max)

Same `.mlpackage` artifacts, just different `--compute-units`:

| Target | Inference | RTFx | Encoder | Decode loop |
|---|---:|---:|---:|---:|
| CPU | 7.72 s | 135.8× | 5.96 s | 1.45 s |
| **ANE** | **4.27 s** | **245.5×** | 2.53 s | 1.44 s |
| GPU | 2.81 s | 373.2× | 1.03 s | 1.46 s |

Reference Python `coremltools` numbers on the same model on the same
machine: CPU 138.2×, ANE 245.5×, GPU 414.2×. Swift is within ~1% of
Python on ANE and CPU; GPU has ~10% overhead in how we feed the encoder
input array, left as a future optimization.

## How the fast path works

The decoder loop makes ~3000 tiny `MLModel.prediction(from:)` calls on a
17.5-min clip (one per emitted symbol). We keep this cheap by:

- **Pre-allocating every input `MLMultiArray` once** (`input_ids`,
  `hidden`, `cell`, `encoder_frame`, `decoder_state`) and reusing them
  across every step. The `MLFeatureProvider` handed to Core ML is a
  custom `FeatureBag` that wraps those arrays and just answers
  `featureValue(for:)` lookups.
- **Scoping prediction outputs in `autoreleasepool`**, so the
  IOSurface-backed output buffers from each prediction are returned to
  the pool before the next iteration. Without this you exhaust the
  IOSurface pool in a few seconds.
- **Argmax via `vDSP_maxvi`** on the raw `MLMultiArray` data pointer.
- **All per-symbol state as raw `UnsafeMutablePointer`s**; no
  `[Float]` allocations inside the loop.

The end-to-end result is a Swift implementation that runs at Core ML's
native speed on this model -- identical to what you'd get from
`coremltools` in Python, without any `numpy`, `torch`, or Python
in the hot path.

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
