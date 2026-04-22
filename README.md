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

Same `.mlpackage` artifacts, just different `--compute-units`. All
numbers are 3-run medians.

| Target | Inference | RTFx | Encoder stage | Decode stage (total) | vs Python |
|---|---:|---:|---:|---:|---:|
| CPU | 6.43 s | **162.7×** | 6.38 s | 1.74 s | +17.7% |
| **ANE** | **2.60 s** | **402.4×** | 2.54 s | 1.39 s | **+63.9%** |
| **GPU** | **0.92 s** | **1145.4×** | 0.77 s | 2.48 s | **+176.5%** |

(Decode stage total > inference total means multiple decode workers ran
in parallel -- see section 4 below.)

Python `coremltools` reference on the same model on the same machine:
CPU 138.2×, ANE 245.5×, GPU 414.2×. **Swift is strictly faster than
Python on every target; almost 3× faster on GPU.**

GPU throughput scales with worker count -- 4 is the sweet spot:

| `--decoder-workers` | GPU RTFx |
|---:|---:|
| 1 | 585× |
| 2 | 895× |
| 3 | 1100× |
| **4 (default)** | **1215×** |
| 6 | 1230× |
| 8 | 1241× |

## How the fast path works

Four optimizations stacked on top of each other:

### 1. Pre-allocated input / reused `MLFeatureProvider`

The decoder makes ~3000 tiny `MLModel.prediction(from:)` calls on a
17.5-min clip -- one per emitted symbol. Each call used to build a fresh
`MLDictionaryFeatureProvider`, which allocates a Swift dict + bridges
through `NSDictionary`. Over 3000 calls that was a non-trivial fraction
of the wall clock.

[`FeatureBag.swift`](Sources/ParakeetTDT/FeatureBag.swift) is a minimal
custom `MLFeatureProvider` built once per submodule and reused every
step. The `MLMultiArray`s it wraps (`input_ids`, `hidden`, `cell`,
`encoder_frame`, `decoder_state`) are also pre-allocated once and
overwritten in place -- no allocations in the hot loop.

### 2. `autoreleasepool` around each prediction

Core ML's output `MLMultiArray`s come out of a pool of IOSurface-backed
buffers. If you retain them past the next prediction call (via any stray
Swift reference), you blow the pool and the process crashes with
`Failed to allocate memory IOSurface object`. Wrapping each
`prediction(from:)` call in an `autoreleasepool` guarantees the
outputs are returned to the pool before the next iteration.

### 3. 3-stage pipeline across chunks

A 17.5-minute clip is 35 × 30-second chunks, and each chunk has three
sequential stages on wildly different hardware:

- **mel extraction** (CPU, `vDSP`) -- ~9 ms / chunk
- **encoder** (ANE / GPU / CPU depending on `computeUnits`) -- 22-170 ms
- **greedy TDT decode loop** (CPU, LSTM + joint) -- ~40 ms

Running them sequentially per chunk wastes the idle hardware. The ANE
is doing nothing while the CPU decodes, and the CPU is doing nothing
while the ANE encodes.

[`Pipeline.swift`](Sources/ParakeetTDT/Pipeline.swift) runs the three
stages on dedicated `DispatchQueue`s, with bounded blocking queues
connecting them. Chunk N's encoder runs while chunk N+1's mel extracts
and chunk N-1's decode loop runs -- all on different hardware.

### 4. Parallel decode worker pool

After (3), GPU builds became decode-bound (1.5 s decode vs 0.77 s
encoder). Solved by running the decode stage as a worker pool --
[`DecoderWorker.swift`](Sources/ParakeetTDT/DecoderWorker.swift) owns
its own per-worker buffer set (`input_ids`, `hidden`, `cell`,
`encoder_frame`, `decoder_state`, plus the two feature bags) so
concurrent decode threads don't alias state. The encoder + joint
`MLModel` objects are shared across workers; `MLModel.prediction(from:)`
is thread-safe per Apple's docs.

Default worker counts tuned per compute target on M5 Max:

- `.cpu` -> 1 worker (more than one contends with the on-CPU encoder)
- `.ane` -> 2 workers (encoder-bound; more doesn't help)
- `.gpu`, `.all` -> 4 workers (3× speedup over 1, near-flat past 4)

Overridable via `decoderWorkers:` on `ParakeetTranscriber.init(...)` or
`--decoder-workers N` on the CLI.

## What's still on the table

- **IOSurface-backed encoder input**: small additional win on GPU if
  the encoder becomes the new bottleneck. Not a priority at 1215× GPU.
- **vDSP_zvmags + vDSP_mmul in mel**: mel runs in parallel with the
  encoder so it's never on the critical path. Still, would be cleaner.
- **Fused decoder + joint mlmodel**: halves per-symbol dispatch count.
  Requires a reconversion in Python.

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
