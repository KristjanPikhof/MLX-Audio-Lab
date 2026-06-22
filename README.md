# MLX Audio Lab

Minimal macOS SwiftUI app for testing different MLX audio models locally on
macOS and comparing their transcription performance.

The goal is to provide a small local benchmark lab for MLX audio models: pick a
model, download it from Hugging Face when needed, record audio or import media,
and compare model load and generation speed. More compatible MLX audio models
will be added soon.

## Supported models

| Model | Hugging Face repo | Approx. download | Notes |
|---|---|---:|---|
| Nemotron 3.5 ASR Streaming 0.6B | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 721 MB | Smaller 8-bit MLX ASR model. |
| Parakeet TDT 0.6B v3 | `mlx-community/parakeet-tdt-0.6b-v3` | 2.51 GB | MLX conversion of NVIDIA Parakeet v3 for multilingual ASR comparison. |

## Requirements

| Requirement | Notes |
|---|---|
| macOS | Apple Silicon recommended; the package targets macOS 14+. |
| Xcode | Required because MLX builds and bundles Metal shader resources. |
| Metal toolchain | If Xcode reports it missing, run `xcodebuild -downloadComponent MetalToolchain`. |
| Network | Needed on first run to download the model from Hugging Face. |

## Model storage

Models are downloaded from Hugging Face and stored in the local Hugging Face
cache under the `mlx-audio` folder:

```text
~/.cache/huggingface/hub/mlx-audio/
```

The current model folders are:

```text
~/.cache/huggingface/hub/mlx-audio/mlx-community_nemotron-3.5-asr-streaming-0.6b-8bit
~/.cache/huggingface/hub/mlx-audio/mlx-community_parakeet-tdt-0.6b-v3
```

If `HF_HUB_CACHE` or `HF_HOME` is set, the Hugging Face cache root follows those
environment variables.

## Quick start

```bash
cd MLXAudioLab
./run.sh
```

Running `xcodebuild ... build` by itself only compiles the package. Use
`./run.sh` to build, package, sign, and open the macOS app bundle.

Click **Record**, speak, then click **Stop**. The recording is kept as the
current audio sample for this app session and is transcribed with the selected
model.

Click **Import Media** to select an audio or video file. The app extracts the
audio into a temporary session WAV before running the model. Imported files are
not transcribed automatically; click **Run Selected Model** when you are ready.

Current import targets are Apple-decoded audio/video media, including `.wav`,
`.mp3`, `.m4a`, `.aac`, `.mp4`, `.m4v`, `.mov`, `.aiff`, and `.caf` when the
file contains a readable audio track. Video files are used for audio only.

Use the model picker to switch models. The selected row shows whether the model
is already available on the computer or whether it still needs to be downloaded.
After switching models, click **Run Selected Model** to regenerate output from
the current audio sample.

Use the small trash button next to the model picker to delete the selected
downloaded model from the local Hugging Face cache.

## Local files

| Data | Location | GitHub note |
|---|---|---|
| Source | `Package.swift`, `Sources/`, `run.sh` | Safe to commit. |
| App icon | `Assets/AppIcon.svg`, `Sources/MLXAudioLab/Resources/AppIcon.icns` | Safe to commit. |
| Build output | `.derivedData/`, `.build/`, `.run/`, `*.app` | Ignored by `.gitignore`. |
| Runtime logs | `~/Library/Caches/MLXAudioLab/` | Local only; do not commit. |
| Temporary audio | macOS temporary directory | Recorded/imported media is converted to session-only WAV and deleted when cleared, replaced, or on next launch. |
| Model cache | `~/.cache/huggingface/hub/mlx-audio/` | Local only; large downloads are not part of the repo. |

The app does not require API keys. It does not write transcript text to its logs.

## License

MIT. See [LICENSE](./LICENSE).

## Troubleshooting

If the window closes unexpectedly, check the cache logs:

```bash
cat "$HOME/Library/Caches/MLXAudioLab/mlx-audio-lab.log"
cat "$HOME/Library/Caches/MLXAudioLab/mlx-audio-lab.stderr.log"
```
