# MLX Audio Lab

[![Swift 6.2](https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white)](Package.swift)
[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-0A84FF?logo=apple&logoColor=white)](Package.swift)
[![SwiftPM](https://img.shields.io/badge/build-SwiftPM-6E6E73)](Package.swift)
[![Apple Silicon recommended](https://img.shields.io/badge/Apple%20Silicon-recommended-111111?logo=apple&logoColor=white)](#requirements)
[![MLX local ASR](https://img.shields.io/badge/MLX-local%20ASR-00A67E)](#supported-models)
[![Models: Nemotron + Parakeet](https://img.shields.io/badge/models-Nemotron%20%2B%20Parakeet-7C3AED)](#supported-models)
[![Media import](https://img.shields.io/badge/import-audio%20%2B%20video-2563EB)](#quick-start)
[![TXT/MD export](https://img.shields.io/badge/export-TXT%20%2F%20MD-059669)](#quick-start)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Minimal macOS SwiftUI app for testing different MLX audio models locally on
macOS and comparing their transcription performance.

The goal is to provide a small local benchmark lab for MLX audio models: pick a
model, download it from Hugging Face when needed, record audio or import media,
and compare model load and generation speed. More compatible MLX audio models
will be added soon.

<img width="1232" height="894" alt="image" src="https://github.com/user-attachments/assets/10675ff8-7dbc-4b24-8f9b-98d04f9c6216" />

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

The app window has a minimum size of 1120 x 750.

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

The transcript panel has a **Follow** checkbox, enabled by default, to keep the
latest generated text in view while transcription is running.

The performance panel also shows transcript word, letter, character, and line
counts. It can copy the transcript or save it as `.txt` or `.md`.

Use the small trash button next to the model picker to delete the selected
downloaded model from the local Hugging Face cache.

## Performance notes

The app reports audio length, audio load, model load, generation, model-reported
time, and total time separately. These metrics update as audio load, model load,
and each decode chunk completes. In local tests, model load was usually under 2
seconds; long waits came from generation over long media files.

Imported media is normalized to 16 kHz mono WAV, then decoded in bounded
30-second model chunks. This keeps peak memory lower than the upstream default
20-minute decode window and is safer for long recordings. Very long files can
still take minutes because the current app loads the normalized audio sample
before running the model. The **Cancel** button stops after the current decode
chunk finishes, so it may take a few seconds to settle on slower runs.

## Local files

| Data | Location | GitHub note |
|---|---|---|
| Source | `Package.swift`, `Sources/`, `run.sh` | Safe to commit. |
| App icon | `Assets/AppIcon.svg`, `Sources/MLXAudioLab/Resources/AppIcon.icns` | Safe to commit. |
| Build output | `.derivedData/`, `.build/`, `.run/`, `*.app` | Ignored by `.gitignore`. |
| Runtime logs | `~/Library/Caches/MLXAudioLab/` | Local only; do not commit. |
| Temporary audio | macOS temporary directory | Recorded/imported media is converted to session-only WAV and deleted when cleared, replaced, or on next launch. |
| Model cache | `~/.cache/huggingface/hub/mlx-audio/` | Local only; large downloads are not part of the repo. |

The **Local paths** panel can copy or open the logs and model cache folders in
Finder. Opening a path creates the folder first if it does not exist yet.

The app does not require API keys. It does not write transcript text to its logs.

## License

MIT. See [LICENSE](./LICENSE).

## Troubleshooting

If the window closes unexpectedly, check the cache logs:

```bash
cat "$HOME/Library/Caches/MLXAudioLab/mlx-audio-lab.log"
cat "$HOME/Library/Caches/MLXAudioLab/mlx-audio-lab.stderr.log"
```
