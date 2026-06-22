# MLX Audio Lab

Minimal macOS SwiftUI app for testing different MLX audio models locally on
macOS and comparing their transcription performance.

The current build ships with
`mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` wired as the default ASR
model. More compatible MLX audio models will be added soon so the app can be
used as a small local benchmark lab rather than a single-model demo.

## Requirements

| Requirement | Notes |
|---|---|
| macOS | Apple Silicon recommended; the package targets macOS 14+. |
| Xcode | Required because MLX builds and bundles Metal shader resources. |
| Metal toolchain | If Xcode reports it missing, run `xcodebuild -downloadComponent MetalToolchain`. |
| Network | Needed on first run to download the model from Hugging Face. |

## Model storage

Models are downloaded from Hugging Face and stored in the local Hugging Face
cache. The current default model is placed under:

```text
~/.cache/huggingface/hub/mlx-audio/mlx-community_nemotron-3.5-asr-streaming-0.6b-8bit
```

Future compatible models will use the same Hugging Face cache area unless a
model-specific loader requires a different cache layout.

## Quick start

```bash
cd MLXAudioLab
./run.sh
```

Running `xcodebuild ... build` by itself only compiles the package. Use
`./run.sh` to build, package, sign, and open the macOS app bundle.

Click **Record**, speak, then click **Stop**. The first transcription downloads
and loads the model, so later runs are the useful generation-speed comparison.

## Local files

| Data | Location | GitHub note |
|---|---|---|
| Source | `Package.swift`, `Sources/`, `run.sh` | Safe to commit. |
| App icon | `Assets/AppIcon.svg`, `Sources/MLXAudioLab/Resources/AppIcon.icns` | Safe to commit. |
| Build output | `.derivedData/`, `.build/`, `.run/`, `*.app` | Ignored by `.gitignore`. |
| Runtime logs | `~/Library/Caches/MLXAudioLab/` | Local only; do not commit. |
| Temporary audio | macOS temporary directory | Deleted after transcription finishes. |
| Model cache | `~/.cache/huggingface/hub/mlx-audio/mlx-community_nemotron-3.5-asr-streaming-0.6b-8bit` | Local only; about 721 MB after download. |

The app does not require API keys. It does not write transcript text to its logs.

## License

MIT. See [LICENSE](./LICENSE).

## Troubleshooting

If the window closes unexpectedly, check the cache logs:

```bash
cat "$HOME/Library/Caches/MLXAudioLab/mlx-audio-lab.log"
cat "$HOME/Library/Caches/MLXAudioLab/mlx-audio-lab.stderr.log"
```
