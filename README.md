# MLX Audio Lab

[![Swift 6.2](https://img.shields.io/badge/Swift-6.2-F05138?logo=swift&logoColor=white)](Package.swift)
[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-0A84FF?logo=apple&logoColor=white)](Package.swift)
[![SwiftPM](https://img.shields.io/badge/build-SwiftPM-6E6E73)](Package.swift)
[![Apple Silicon recommended](https://img.shields.io/badge/Apple%20Silicon-recommended-111111?logo=apple&logoColor=white)](#requirements)
[![MLX local ASR](https://img.shields.io/badge/MLX-local%20ASR-00A67E)](#supported-models)
[![Models: 9 ASR families](https://img.shields.io/badge/models-9%20ASR%20families-7C3AED)](#supported-models)
[![Media import](https://img.shields.io/badge/import-audio%20%2B%20video-2563EB)](#quick-start)
[![TXT/MD export](https://img.shields.io/badge/export-TXT%20%2F%20MD-059669)](#quick-start)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Minimal macOS SwiftUI app for testing different MLX audio models locally on
macOS and comparing their transcription performance.

The goal is to provide a small local benchmark lab for MLX audio models: pick a
model, download it from Hugging Face when needed, record audio or import media,
and compare model load and generation speed. More compatible MLX audio models
will be added as the Swift runtime support matures.

<img width="1232" height="894" alt="image" src="https://github.com/user-attachments/assets/10675ff8-7dbc-4b24-8f9b-98d04f9c6216" />

## Supported models

These models appear in the app picker because the current pinned
`mlx-audio-swift` dependency exposes a compatible `STTGenerationModel` loader
for each family.

| Model | Hugging Face repo | Approx. download | Notes |
|---|---|---:|---|
| Nemotron 3.5 ASR Streaming 0.6B bf16 | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | 1.28 GB | Full-quality bf16 MLX conversion and recommended default. |
| Nemotron 3.5 ASR Streaming 0.6B 8-bit | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 756 MB | Smaller 8-bit MLX ASR model; model card says quality matches bf16. |
| Parakeet TDT 0.6B v3 | `mlx-community/parakeet-tdt-0.6b-v3` | 2.51 GB | MLX conversion of NVIDIA Parakeet v3 for multilingual ASR comparison. |
| Qwen3 ASR 0.6B 4-bit | `mlx-community/Qwen3-ASR-0.6B-4bit` | 708 MB | Compact Qwen3 ASR option for fast comparison. |
| Qwen3 ASR 1.7B 4-bit | `mlx-community/Qwen3-ASR-1.7B-4bit` | 1.6 GB | Larger Qwen3 ASR option for quality and speed checks. |
| Whisper Large v3 Turbo ASR fp16 | `mlx-community/whisper-large-v3-turbo-asr-fp16` | 1.61 GB | Whisper Turbo baseline converted for `mlx-audio`. |
| SenseVoice Small | `mlx-community/SenseVoiceSmall` | 936 MB | Non-autoregressive ASR with language, emotion, and event metadata. |
| GLM-ASR Nano 2512 4-bit | `mlx-community/GLM-ASR-Nano-2512-4bit` | 1.28 GB | English/Chinese ASR model with GLM decoder. |
| Granite 4.0 1B Speech 5-bit | `mlx-community/granite-4.0-1b-speech-5bit` | 2.22 GB | ASR and translation-style speech model. |
| Voxtral Mini 4B Realtime 4-bit | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | 3.13 GB | Heavy streaming STT model; benchmarked here through offline chunks. |
| Cohere Transcribe 03-2026 fp16 | `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | 3.85 GiB | Large community MLX conversion for experimental comparison. |

### Researched candidates

These Hugging Face MLX audio sources are worth tracking, but they are not in
the picker yet because they need a different UI path, are unusually heavy, or
need fresh-download verification with `mlx-audio-swift`.

| Candidate | Hugging Face repo | Status |
|---|---|---|
| Mega-ASR | `mlx-community/Mega-ASR-8bit` | Qwen3-derived routing model; promising but needs runtime validation. |
| VibeVoice ASR | `mlx-community/VibeVoice-ASR-4bit` | ASR plus diarization tags; likely too large for the first app list. |
| FireRed ASR 2 | `mlx-community/FireRedASR2-AED-mlx` | Swift class exists, but fresh download needs sidecar-file handling before picker use. |
| Qwen3 ForcedAligner | `mlx-community/Qwen3-ForcedAligner-0.6B-4bit` | Alignment model; needs reference text input, not just audio transcription. |
| Qwen2-Audio Instruct | `mlx-community/Qwen2-Audio-7B-Instruct-4bit` | Prompt-driven multimodal audio-text model, not a plain ASR row. |
| MiMo ASR | `mlx-community/MiMo-V2.5-ASR-MLX-4bit` | MLX ASR repo, but it points at a different `mlx-audio` fork. |
| Belle Whisper zh | `mlx-community/belle-whisper-large-v3-zh-8bit` | Whisper-family Chinese fine-tune; likely compatible but not yet tested. |

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
~/.cache/huggingface/hub/mlx-audio/mlx-community_nemotron-3.5-asr-streaming-0.6b
~/.cache/huggingface/hub/mlx-audio/mlx-community_nemotron-3.5-asr-streaming-0.6b-8bit
~/.cache/huggingface/hub/mlx-audio/mlx-community_parakeet-tdt-0.6b-v3
~/.cache/huggingface/hub/mlx-audio/mlx-community_Qwen3-ASR-0.6B-4bit
~/.cache/huggingface/hub/mlx-audio/mlx-community_Qwen3-ASR-1.7B-4bit
~/.cache/huggingface/hub/mlx-audio/mlx-community_whisper-large-v3-turbo-asr-fp16
~/.cache/huggingface/hub/mlx-audio/mlx-community_SenseVoiceSmall
~/.cache/huggingface/hub/mlx-audio/mlx-community_GLM-ASR-Nano-2512-4bit
~/.cache/huggingface/hub/mlx-audio/mlx-community_granite-4.0-1b-speech-5bit
~/.cache/huggingface/hub/mlx-audio/mlx-community_Voxtral-Mini-4B-Realtime-2602-4bit
~/.cache/huggingface/hub/mlx-audio/beshkenadze_cohere-transcribe-03-2026-mlx-fp16
```

If `HF_HUB_CACHE` or `HF_HOME` is set, the Hugging Face cache root follows those
environment variables.

The app uses the small trash button next to the model picker to remove the
selected model. Deletion removes both the app-specific `mlx-audio/<repo>` copy
and the matching standard Hugging Face `models--owner--repo` cache directory
when present. It does not delete recordings, imported samples, logs, or other
models.

## Quick start

```bash
cd MLXAudioLab
./run.sh
```

Running `xcodebuild ... build` by itself only compiles the package. Use
`./run.sh` to build, package, sign, and open the macOS app bundle.

The app window has a minimum size of 1120 x 775.

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

When a model is downloading, a progress banner appears at the top of the
window with the model name, percentage, downloaded bytes, and a linear progress
bar. App-managed downloads use the Hugging Face LFS transfer path so the banner
can update while large model files are still downloading. If a model already
exists in the standard Hugging Face cache, the app copies it into the
`mlx-audio` folder with chunked copy progress instead of a silent file copy.

The transcript panel has a **Follow** checkbox, enabled by default, to keep the
latest generated text in view while transcription is running.

The performance panel also shows transcript word, letter, character, and line
counts. It can copy the transcript or save it as `.txt` or `.md`.

Use the small trash button next to the model picker to delete the selected
downloaded model from local disk.

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
| Hugging Face cache | `~/.cache/huggingface/hub/models--*/` | Local only; selected model cache is deleted by the model trash button. |

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
