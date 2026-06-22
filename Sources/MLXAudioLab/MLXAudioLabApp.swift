import AVFoundation
import Darwin
import Foundation
import MLXAudioCore
import MLXAudioSTT
import Observation
import SwiftUI
import UniformTypeIdentifiers

enum AudioModelFamily: String, Sendable {
    case nemotron
    case parakeet
}

struct AudioModelOption: Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let repoID: String
    let family: AudioModelFamily
    let downloadSizeDescription: String
    let subtitle: String
    let languageHint: String?

    static let supported: [AudioModelOption] = [
        AudioModelOption(
            id: "nemotron-streaming-0.6b-8bit",
            displayName: "Nemotron 3.5 ASR Streaming 0.6B",
            repoID: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
            family: .nemotron,
            downloadSizeDescription: "~721 MB",
            subtitle: "8-bit MLX conversion; smaller download and good for quick local checks.",
            languageHint: "auto"
        ),
        AudioModelOption(
            id: "parakeet-tdt-0.6b-v3",
            displayName: "Parakeet TDT 0.6B v3",
            repoID: "mlx-community/parakeet-tdt-0.6b-v3",
            family: .parakeet,
            downloadSizeDescription: "~2.51 GB",
            subtitle: "MLX conversion of NVIDIA Parakeet v3; multilingual ASR comparison target.",
            languageHint: nil
        )
    ]
}

enum ModelLocalAvailability: Sendable {
    case available
    case notDownloaded
    case incomplete
}

enum ModelCache {
    static func rootDirectory() -> URL {
        let env = ProcessInfo.processInfo.environment
        let hubRoot: URL

        if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
            hubRoot = URL(fileURLWithPath: expandTilde(hubCache), isDirectory: true)
        } else if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            hubRoot = URL(fileURLWithPath: expandTilde(hfHome), isDirectory: true)
                .appending(path: "hub", directoryHint: .isDirectory)
        } else {
            hubRoot = FileManager.default.homeDirectoryForCurrentUser
                .appending(path: ".cache/huggingface/hub", directoryHint: .isDirectory)
        }

        return hubRoot.appending(path: "mlx-audio", directoryHint: .isDirectory)
    }

    static func directory(for option: AudioModelOption) -> URL {
        rootDirectory().appending(
            path: option.repoID.replacingOccurrences(of: "/", with: "_"),
            directoryHint: .isDirectory
        )
    }

    static func availability(for option: AudioModelOption) -> ModelLocalAvailability {
        let directory = directory(for: option)
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false

        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return .notDownloaded
        }

        let configURL = directory.appending(path: "config.json")
        guard fileManager.fileExists(atPath: configURL.path),
              let configData = try? Data(contentsOf: configURL),
              (try? JSONSerialization.jsonObject(with: configData)) != nil,
              containsNonEmptyFile(withExtension: "safetensors", in: directory)
        else {
            return .incomplete
        }

        return .available
    }

    private static func containsNonEmptyFile(withExtension fileExtension: String, in directory: URL) -> Bool {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        for case let fileURL as URL in enumerator
        where fileURL.pathExtension.lowercased() == fileExtension {
            let fileSize = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            if fileSize > 0 {
                return true
            }
        }

        return false
    }

    private static func expandTilde(_ path: String) -> String {
        NSString(string: path).expandingTildeInPath
    }
}

enum AudioSampleSource: String, Sendable {
    case recorded
    case imported

    var displayName: String {
        switch self {
        case .recorded:
            return "Recorded sample"
        case .imported:
            return "Imported WAV"
        }
    }

    var systemImage: String {
        switch self {
        case .recorded:
            return "record.circle"
        case .imported:
            return "waveform.badge.plus"
        }
    }
}

struct AudioSample: Identifiable, Sendable {
    let id: UUID
    let url: URL
    let source: AudioSampleSource
    let displayName: String
    let durationSeconds: Double
    let createdAt: Date
}

enum ProbeLog {
    static func configureProcessOutput() {
        let directory = logDirectory()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        _ = freopen(directory.appending(path: "mlx-audio-lab.stdout.log").path, "a", stdout)
        _ = freopen(directory.appending(path: "mlx-audio-lab.stderr.log").path, "a", stderr)
        setbuf(stdout, nil)
        setbuf(stderr, nil)
        write("app launched")
    }

    static func write(_ message: String) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "\(timestamp) \(message)\n"
        let url = logDirectory().appending(path: "mlx-audio-lab.log")

        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }

        guard let data = line.data(using: .utf8),
              let handle = try? FileHandle(forWritingTo: url)
        else {
            return
        }

        do {
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.close()
        } catch {
            try? handle.close()
        }
    }

    static func write(_ message: String, error: Error) {
        write("\(message): \(ProbeViewModel.describe(error))")
    }

    static func logDirectory() -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appending(path: "MLXAudioLab", directoryHint: .isDirectory)
    }
}

struct TranscriptionMetrics: Sendable {
    var audioSeconds: Double = 0
    var audioLoadSeconds: Double = 0
    var modelLoadSeconds: Double = 0
    var generationSeconds: Double = 0
    var modelReportedSeconds: Double = 0
    var totalSeconds: Double = 0
    var wasModelAlreadyLoaded = false
}

struct TranscriptionResult: Sendable {
    let text: String
    let metrics: TranscriptionMetrics
}

struct ModelPreparationResult: Sendable {
    let modelLoadSeconds: Double
    let wasModelAlreadyLoaded: Bool
}

actor AudioModelTranscriber {
    private var loadedModels: [String: any STTGenerationModel] = [:]

    func prepareModel(_ option: AudioModelOption) async throws -> ModelPreparationResult {
        ProbeLog.write("model prepare requested repo=\(option.repoID)")
        let start = ContinuousClock.now
        let wasLoaded = loadedModels[option.id] != nil
        _ = try await loadedModel(for: option)
        let seconds = Self.seconds(since: start)
        ProbeLog.write("model prepare complete repo=\(option.repoID) seconds=\(seconds) wasLoaded=\(wasLoaded)")
        return ModelPreparationResult(modelLoadSeconds: seconds, wasModelAlreadyLoaded: wasLoaded)
    }

    func transcribe(
        audioURL: URL,
        audioSeconds: Double,
        using option: AudioModelOption
    ) async throws -> TranscriptionResult {
        ProbeLog.write(
            "transcribe begin repo=\(option.repoID) audio=\(audioURL.lastPathComponent) audioSeconds=\(audioSeconds)"
        )
        let totalStart = ContinuousClock.now

        let audioLoadStart = ContinuousClock.now
        ProbeLog.write("audio load begin")
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16_000)
        let audioLoadSeconds = Self.seconds(since: audioLoadStart)
        ProbeLog.write("audio load complete seconds=\(audioLoadSeconds)")

        let modelLoadStart = ContinuousClock.now
        let wasLoaded = loadedModels[option.id] != nil
        let loadedModel = try await loadedModel(for: option)
        let modelLoadSeconds = Self.seconds(since: modelLoadStart)

        let generationStart = ContinuousClock.now
        ProbeLog.write("generation begin repo=\(option.repoID)")
        let output = loadedModel.generate(
            audio: audio,
            generationParameters: generationParameters(for: loadedModel, option: option)
        )
        let generationSeconds = Self.seconds(since: generationStart)
        ProbeLog.write("generation complete seconds=\(generationSeconds) textLength=\(output.text.count)")

        let metrics = TranscriptionMetrics(
            audioSeconds: audioSeconds,
            audioLoadSeconds: audioLoadSeconds,
            modelLoadSeconds: modelLoadSeconds,
            generationSeconds: generationSeconds,
            modelReportedSeconds: output.totalTime,
            totalSeconds: Self.seconds(since: totalStart),
            wasModelAlreadyLoaded: wasLoaded
        )

        let result = TranscriptionResult(text: output.text, metrics: metrics)
        ProbeLog.write("transcribe complete totalSeconds=\(metrics.totalSeconds)")
        return result
    }

    private func loadedModel(for option: AudioModelOption) async throws -> any STTGenerationModel {
        if let model = loadedModels[option.id] {
            ProbeLog.write("model already loaded repo=\(option.repoID)")
            return model
        }

        ProbeLog.write("model load begin repo=\(option.repoID)")
        let model: any STTGenerationModel
        switch option.family {
        case .nemotron:
            model = try await NemotronASRModel.fromPretrained(option.repoID)
        case .parakeet:
            model = try await ParakeetModel.fromPretrained(option.repoID)
        }

        loadedModels[option.id] = model
        ProbeLog.write("model load complete repo=\(option.repoID)")
        return model
    }

    private func generationParameters(
        for model: any STTGenerationModel,
        option: AudioModelOption
    ) -> STTGenerateParameters {
        let defaults = model.defaultGenerationParameters
        return STTGenerateParameters(
            maxTokens: defaults.maxTokens,
            temperature: defaults.temperature,
            topP: defaults.topP,
            topK: defaults.topK,
            verbose: false,
            language: option.languageHint,
            chunkDuration: defaults.chunkDuration,
            minChunkDuration: defaults.minChunkDuration,
            repetitionPenalty: defaults.repetitionPenalty,
            repetitionContextSize: defaults.repetitionContextSize
        )
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: ContinuousClock.now)
        return Double(duration.components.seconds) +
            Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }
}

@MainActor
@Observable
final class ProbeViewModel {
    var isRecording = false
    var isTranscribing = false
    var isPreparingModel = false
    var isImportingAudio = false
    var selectedModelID = AudioModelOption.supported[0].id {
        didSet {
            guard oldValue != selectedModelID else { return }
            refreshModelAvailability(updateStatus: true)
        }
    }
    var modelAvailability: [String: ModelLocalAvailability] = [:]
    var loadedModelIDs: Set<String> = []
    var currentSample: AudioSample?
    var status = "Ready"
    var transcript = ""
    var errorMessage: String?
    var metrics = TranscriptionMetrics()
    var recordingElapsedSeconds: Double = 0
    var logDirectoryPath = ProbeLog.logDirectory().path

    private let transcriber = AudioModelTranscriber()
    private var recorder: AVAudioRecorder?
    private var recordingStartedAt: Date?
    private var recordingURL: URL?
    private var timerTask: Task<Void, Never>?
    private var transcriptionTask: Task<Void, Never>?
    private var modelPreparationTask: Task<Void, Never>?

    init() {
        Self.cleanupTemporaryAudioFiles()
        refreshModelAvailability(updateStatus: true)
    }

    var primaryButtonTitle: String {
        if isRecording { return "Stop" }
        if isTranscribing { return "Working..." }
        return "Record"
    }

    var canPressPrimaryButton: Bool {
        if isRecording {
            return !isPreparingModel && !isTranscribing
        }
        return !isTranscribing && !isPreparingModel && selectedModelCanRecord
    }

    var modelControlsDisabled: Bool {
        isRecording || isTranscribing || isPreparingModel
    }

    var canImportAudio: Bool {
        !isRecording && !isTranscribing && !isPreparingModel
    }

    var selectedModel: AudioModelOption {
        AudioModelOption.supported.first { $0.id == selectedModelID } ?? AudioModelOption.supported[0]
    }

    var selectedModelAvailability: ModelLocalAvailability {
        modelAvailability[selectedModel.id] ?? ModelCache.availability(for: selectedModel)
    }

    var selectedModelIsLoaded: Bool {
        loadedModelIDs.contains(selectedModel.id)
    }

    var selectedModelStatusText: String {
        if selectedModelIsLoaded {
            return "Loaded for this session"
        }

        switch selectedModelAvailability {
        case .available:
            return "Available on this Mac"
        case .notDownloaded:
            return "Not downloaded"
        case .incomplete:
            return "Cache incomplete"
        }
    }

    var selectedModelStatusIcon: String {
        if selectedModelIsLoaded { return "bolt.fill" }

        switch selectedModelAvailability {
        case .available:
            return "checkmark.circle.fill"
        case .notDownloaded:
            return "arrow.down.circle"
        case .incomplete:
            return "exclamationmark.triangle.fill"
        }
    }

    var selectedModelActionTitle: String {
        if isPreparingModel { return "Preparing..." }
        if selectedModelIsLoaded { return "Selected" }

        switch selectedModelAvailability {
        case .available:
            return "Select"
        case .notDownloaded:
            return "Download & Select"
        case .incomplete:
            return "Repair & Select"
        }
    }

    var selectedModelActionIcon: String {
        if isPreparingModel { return "hourglass" }
        if selectedModelIsLoaded { return "checkmark" }

        switch selectedModelAvailability {
        case .available:
            return "checkmark.circle"
        case .notDownloaded:
            return "arrow.down.circle"
        case .incomplete:
            return "arrow.clockwise.circle"
        }
    }

    var canPrepareSelectedModel: Bool {
        !modelControlsDisabled && !selectedModelIsLoaded
    }

    var selectedModelCanRecord: Bool {
        selectedModelIsLoaded || selectedModelAvailability == .available
    }

    var canRunSelectedModel: Bool {
        !isRecording && !isTranscribing && !isPreparingModel && currentSample != nil && selectedModelCanRecord
    }

    var runSelectedModelDisabledText: String {
        if currentSample == nil {
            return "Record or import a WAV first"
        }
        if !selectedModelCanRecord {
            return "Download the selected model before running"
        }
        return ""
    }

    var currentSampleDescription: String {
        guard let currentSample else {
            return "Record or import a WAV first"
        }

        return "\(currentSample.source.displayName) · \(currentSample.displayName) · \(Self.formatSeconds(currentSample.durationSeconds))"
    }

    var currentSampleIcon: String {
        currentSample?.source.systemImage ?? "waveform"
    }

    var modelCacheRootPath: String {
        ModelCache.rootDirectory().path
    }

    func primaryButtonPressed() {
        if isRecording {
            stopRecordingAndTranscribe()
        } else {
            startRecording()
        }
    }

    func refreshModelAvailability(updateStatus: Bool = false) {
        modelAvailability = Dictionary(
            uniqueKeysWithValues: AudioModelOption.supported.map { option in
                (option.id, ModelCache.availability(for: option))
            }
        )

        if updateStatus, !isRecording, !isTranscribing, !isPreparingModel {
            status = idleStatusForSelectedModel()
        }
    }

    func beginImportAudio() {
        guard canImportAudio else { return }
        errorMessage = nil
        isImportingAudio = true
    }

    func handleImportedAudio(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let sourceURL = urls.first else {
                status = "Import canceled"
                return
            }

            do {
                let sample = try importAudioSample(from: sourceURL)
                replaceCurrentSample(with: sample)
                transcript = ""
                metrics = TranscriptionMetrics()
                recordingElapsedSeconds = 0
                errorMessage = nil
                status = "Ready to run"
                ProbeLog.write("audio import complete file=\(sample.url.lastPathComponent) seconds=\(sample.durationSeconds)")
            } catch {
                status = "Import failed"
                errorMessage = Self.describe(error)
                ProbeLog.write("audio import failed file=\(sourceURL.lastPathComponent)", error: error)
            }
        case .failure(let error):
            status = "Import failed"
            errorMessage = Self.describe(error)
            ProbeLog.write("audio import failed", error: error)
        }
    }

    func prepareSelectedModel() {
        guard canPrepareSelectedModel else { return }

        let option = selectedModel
        errorMessage = nil
        status = selectedModelAvailability == .available
            ? "Loading \(option.displayName)..."
            : "Downloading \(option.displayName)..."
        isPreparingModel = true
        ProbeLog.write("model action requested repo=\(option.repoID)")

        modelPreparationTask?.cancel()
        modelPreparationTask = Task {
            do {
                let result = try await transcriber.prepareModel(option)
                loadedModelIDs.insert(option.id)
                metrics.modelLoadSeconds = result.modelLoadSeconds
                metrics.wasModelAlreadyLoaded = result.wasModelAlreadyLoaded
                refreshModelAvailability()
                status = "\(option.displayName) selected"
            } catch {
                status = "Model preparation failed"
                errorMessage = Self.describe(error)
                ProbeLog.write("model preparation failed repo=\(option.repoID)", error: error)
            }
            isPreparingModel = false
        }
    }

    func runSelectedModelForCurrentSample() {
        guard let sample = currentSample else {
            status = "Record or import a WAV first"
            errorMessage = nil
            return
        }

        guard selectedModelCanRecord else {
            status = "Download the selected model before running"
            errorMessage = "\(selectedModel.displayName) is not available on this Mac yet."
            return
        }

        startTranscription(sample: sample, option: selectedModel)
    }

    func startRecording() {
        guard selectedModelCanRecord else {
            status = "Download the selected model before recording"
            errorMessage = "\(selectedModel.displayName) is not available on this Mac yet."
            return
        }

        let option = selectedModel
        ProbeLog.write("record requested")
        errorMessage = nil
        transcript = ""
        metrics = TranscriptionMetrics()
        recordingElapsedSeconds = 0
        status = "Requesting microphone access..."

        Task {
            let granted = await Self.requestMicrophoneAccess()
            ProbeLog.write("microphone access granted=\(granted)")
            guard granted else {
                status = "Microphone access denied"
                errorMessage = "Allow microphone access for Terminal or the built binary in System Settings, then run again."
                return
            }

            do {
                let url = try Self.makeRecordingURL()
                let recorder = try AVAudioRecorder(url: url, settings: Self.recordingSettings)
                recorder.isMeteringEnabled = true
                recorder.prepareToRecord()

                guard recorder.record() else {
                    throw NSError(
                        domain: "MLXAudioLab",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "AVAudioRecorder refused to start recording."]
                    )
                }

                self.recorder = recorder
                self.recordingURL = url
                self.recordingStartedAt = Date()
                self.isRecording = true
                self.status = "Recording with \(option.displayName)..."
                self.startTimer()
                ProbeLog.write("record started repo=\(option.repoID) recording=\(url.lastPathComponent)")
            } catch {
                self.status = "Recording failed"
                self.errorMessage = Self.describe(error)
                ProbeLog.write("record failed", error: error)
            }
        }
    }

    func stopRecordingAndTranscribe() {
        guard isRecording else { return }
        ProbeLog.write("stop requested")

        recorder?.stop()
        recorder = nil
        timerTask?.cancel()
        timerTask = nil

        let elapsed = recordingStartedAt.map { Date().timeIntervalSince($0) } ?? recordingElapsedSeconds
        recordingElapsedSeconds = elapsed
        isRecording = false

        guard let recordingURL else {
            status = "No recording found"
            ProbeLog.write("stop failed no recording url")
            return
        }

        let option = selectedModel
        self.recordingURL = nil
        let sample = AudioSample(
            id: UUID(),
            url: recordingURL,
            source: .recorded,
            displayName: recordingURL.lastPathComponent,
            durationSeconds: elapsed,
            createdAt: Date()
        )
        replaceCurrentSample(with: sample)
        startTranscription(sample: sample, option: option)
    }

    private func startTranscription(sample: AudioSample, option: AudioModelOption) {
        status = "Transcribing with \(option.displayName)..."
        isTranscribing = true
        errorMessage = nil
        transcript = ""
        metrics = TranscriptionMetrics()

        transcriptionTask?.cancel()
        transcriptionTask = Task {
            do {
                let result = try await transcriber.transcribe(
                    audioURL: sample.url,
                    audioSeconds: sample.durationSeconds,
                    using: option
                )
                self.loadedModelIDs.insert(option.id)
                self.refreshModelAvailability()
                self.transcript = result.text
                self.metrics = result.metrics
                self.status = result.text.isEmpty ? "Finished with empty output" : "Finished"
                ProbeLog.write("ui updated with transcription")
            } catch {
                self.status = "Transcription failed"
                self.errorMessage = Self.describe(error)
                ProbeLog.write("transcription failed", error: error)
            }
            self.isTranscribing = false
        }
    }

    func clearOutput() {
        deleteCurrentSample()
        transcript = ""
        errorMessage = nil
        status = idleStatusForSelectedModel()
        metrics = TranscriptionMetrics()
        recordingElapsedSeconds = 0
    }

    private func idleStatusForSelectedModel() -> String {
        if selectedModelIsLoaded {
            return "Ready"
        }

        switch selectedModelAvailability {
        case .available:
            return "Ready"
        case .notDownloaded:
            return "Download the selected model before recording or running"
        case .incomplete:
            return "Repair the selected model cache before recording or running"
        }
    }

    private func importAudioSample(from sourceURL: URL) throws -> AudioSample {
        guard sourceURL.pathExtension.lowercased() == "wav" else {
            throw Self.makeError("Only WAV files can be imported.")
        }

        let didStartAccessing = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }

        let directory = Self.recordingDirectory()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let destinationURL = directory.appending(path: "sample-upload-\(UUID().uuidString).wav")
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }

        do {
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            let durationSeconds = try Self.audioDurationSeconds(for: destinationURL)
            guard durationSeconds > 0 else {
                throw Self.makeError("The selected WAV file has no readable audio.")
            }

            return AudioSample(
                id: UUID(),
                url: destinationURL,
                source: .imported,
                displayName: sourceURL.lastPathComponent,
                durationSeconds: durationSeconds,
                createdAt: Date()
            )
        } catch {
            try? FileManager.default.removeItem(at: destinationURL)
            throw error
        }
    }

    private func replaceCurrentSample(with sample: AudioSample) {
        let previousSample = currentSample
        currentSample = sample

        if let previousSample, previousSample.url != sample.url {
            try? FileManager.default.removeItem(at: previousSample.url)
            ProbeLog.write("temporary sample replaced file=\(previousSample.url.lastPathComponent)")
        }
    }

    private func deleteCurrentSample() {
        guard let sample = currentSample else { return }
        currentSample = nil
        try? FileManager.default.removeItem(at: sample.url)
        ProbeLog.write("temporary sample deleted file=\(sample.url.lastPathComponent)")
    }

    private func startTimer() {
        timerTask?.cancel()
        timerTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                guard let self else { return }
                if let startedAt = self.recordingStartedAt {
                    self.recordingElapsedSeconds = Date().timeIntervalSince(startedAt)
                }
            }
        }
    }

    private static let recordingSettings: [String: Any] = [
        AVFormatIDKey: Int(kAudioFormatLinearPCM),
        AVSampleRateKey: 16_000.0,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16,
        AVLinearPCMIsFloatKey: false,
        AVLinearPCMIsBigEndianKey: false,
        AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
    ]

    private static func makeRecordingURL() throws -> URL {
        let directory = recordingDirectory()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appending(path: "recording-\(UUID().uuidString).wav")
    }

    private static func cleanupTemporaryAudioFiles() {
        let directory = recordingDirectory()
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else {
            return
        }

        for file in files where file.pathExtension.lowercased() == "wav" {
            try? FileManager.default.removeItem(at: file)
        }
    }

    private static func recordingDirectory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "MLXAudioLabRecordings", directoryHint: .isDirectory)
    }

    private static func audioDurationSeconds(for url: URL) throws -> Double {
        let audioFile = try AVAudioFile(forReading: url)
        let sampleRate = audioFile.fileFormat.sampleRate
        guard sampleRate > 0 else {
            throw makeError("The selected WAV file has an invalid sample rate.")
        }

        let durationSeconds = Double(audioFile.length) / sampleRate
        guard durationSeconds.isFinite, durationSeconds > 0 else {
            throw makeError("The selected WAV file has no readable audio.")
        }

        return durationSeconds
    }

    private nonisolated static func requestMicrophoneAccess() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    nonisolated static func formatSeconds(_ seconds: Double) -> String {
        String(format: "%.3f s", seconds)
    }

    private nonisolated static func makeError(_ description: String) -> NSError {
        NSError(
            domain: "MLXAudioLab",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: description]
        )
    }

    nonisolated static func describe(_ error: Error) -> String {
        if let localizedError = error as? LocalizedError,
           let description = localizedError.errorDescription {
            return description
        }
        return (error as NSError).localizedDescription
    }
}

struct ContentView: View {
    @Bindable var model: ProbeViewModel

    var body: some View {
        ZStack {
            LabBackdrop()

            if #available(macOS 26.0, *) {
                GlassEffectContainer(spacing: 18) {
                    workspace
                }
            } else {
                workspace
            }
        }
        .frame(minWidth: 1120, minHeight: 720)
        .onAppear {
            model.refreshModelAvailability(updateStatus: true)
        }
        .fileImporter(
            isPresented: $model.isImportingAudio,
            allowedContentTypes: [.wav],
            allowsMultipleSelection: false
        ) { result in
            model.handleImportedAudio(result)
        }
        .fileDialogMessage("Select a WAV file for local MLX audio testing.")
        .fileDialogConfirmationLabel("Import WAV")
    }

    private var workspace: some View {
        HStack(alignment: .top, spacing: 18) {
            LabSidebar(model: model)
                .frame(width: 318)

            TranscriptWorkspace(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            PerformancePanel(model: model)
                .frame(width: 256)
        }
        .padding(22)
    }
}

struct LabBackdrop: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(red: 0.08, green: 0.095, blue: 0.10),
                    Color(red: 0.11, green: 0.10, blue: 0.085)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            VStack(spacing: 0) {
                Color(red: 0.12, green: 0.42, blue: 0.38)
                    .opacity(0.18)
                    .frame(height: 160)
                    .blur(radius: 36)
                Spacer()
                Color(red: 0.58, green: 0.38, blue: 0.16)
                    .opacity(0.12)
                    .frame(height: 120)
                    .blur(radius: 30)
            }
        }
        .ignoresSafeArea()
    }
}

struct LabSidebar: View {
    @Bindable var model: ProbeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            BrandHeader()
            ModelSetupPanel(model: model)
            SampleControlPanel(model: model)
            SystemPathsPanel(model: model)
            Spacer(minLength: 0)
        }
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

struct BrandHeader: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: "waveform.and.magnifyingglass")
                    .font(.system(size: 22, weight: .semibold))
                    .frame(width: 42, height: 42)
                    .foregroundStyle(.white)
                    .labGlassPanel(cornerRadius: 12, tint: .teal.opacity(0.34), interactive: false)

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLX Audio Lab")
                        .font(.system(size: 26, weight: .semibold, design: .rounded))
                    Text("Local ASR benchmark")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 8) {
                Label("Local", systemImage: "lock")
                Label("WAV", systemImage: "waveform")
                Label("MLX", systemImage: "apple.terminal")
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .labGlassPanel(cornerRadius: 16, tint: .white.opacity(0.04), interactive: false)
    }
}

struct ModelSetupPanel: View {
    @Bindable var model: ProbeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionHeader(title: "Model", symbol: "cpu")

            Picker("Model", selection: $model.selectedModelID) {
                ForEach(AudioModelOption.supported) { option in
                    Text(option.displayName).tag(option.id)
                }
            }
            .labelsHidden()
            .disabled(model.modelControlsDisabled)

            VStack(alignment: .leading, spacing: 8) {
                Label(model.selectedModelStatusText, systemImage: model.selectedModelStatusIcon)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(statusTint)

                Text(model.selectedModel.repoID)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)

                HStack(spacing: 8) {
                    Text(model.selectedModel.downloadSizeDescription)
                    Text(model.selectedModel.family.rawValue.capitalized)
                }
                .font(.caption)
                .foregroundStyle(.tertiary)
            }

            HStack(spacing: 10) {
                Button {
                    model.prepareSelectedModel()
                } label: {
                    Label(model.selectedModelActionTitle, systemImage: model.selectedModelActionIcon)
                        .frame(maxWidth: .infinity)
                }
                .labButtonStyle(prominent: true)
                .disabled(!model.canPrepareSelectedModel)

                Button {
                    model.refreshModelAvailability(updateStatus: true)
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .frame(width: 28)
                }
                .labButtonStyle()
                .disabled(model.modelControlsDisabled)
                .help("Refresh local model availability")
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .labGlassPanel(cornerRadius: 16, tint: .teal.opacity(0.08), interactive: false)
    }

    private var statusTint: Color {
        if model.selectedModelIsLoaded { return .green }

        switch model.selectedModelAvailability {
        case .available:
            return .green
        case .notDownloaded:
            return .secondary
        case .incomplete:
            return .orange
        }
    }
}

struct SampleControlPanel: View {
    @Bindable var model: ProbeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionHeader(title: "Audio sample", symbol: "waveform")
            sampleSummary

            Button {
                model.primaryButtonPressed()
            } label: {
                Label(model.primaryButtonTitle, systemImage: model.isRecording ? "stop.fill" : "record.circle")
                    .frame(maxWidth: .infinity)
            }
            .labButtonStyle(prominent: true)
            .tint(model.isRecording ? .red : .accentColor)
            .disabled(!model.canPressPrimaryButton)
            .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")

            HStack(spacing: 10) {
                Button {
                    model.runSelectedModelForCurrentSample()
                } label: {
                    Label("Run", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .labButtonStyle(prominent: true)
                .disabled(!model.canRunSelectedModel)

                Button {
                    model.clearOutput()
                } label: {
                    Image(systemName: "trash")
                        .frame(width: 28)
                }
                .labButtonStyle()
                .disabled(model.isRecording || model.isTranscribing || model.isPreparingModel)
                .help("Clear sample and output")
            }

            Button {
                model.beginImportAudio()
            } label: {
                Label("Import WAV", systemImage: "square.and.arrow.down")
                    .frame(maxWidth: .infinity)
            }
            .labButtonStyle()
            .disabled(!model.canImportAudio)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .labGlassPanel(cornerRadius: 16, tint: .orange.opacity(0.07), interactive: false)
    }

    @ViewBuilder
    private var sampleSummary: some View {
        if let sample = model.currentSample {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: sample.source.systemImage)
                    .font(.title3)
                    .frame(width: 32, height: 32)
                    .foregroundStyle(.teal)

                VStack(alignment: .leading, spacing: 4) {
                    Text(sample.source.displayName)
                        .font(.callout.weight(.medium))
                    Text(sample.displayName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text("Length \(ProbeViewModel.formatSeconds(sample.durationSeconds))")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                Label(model.currentSampleDescription, systemImage: model.currentSampleIcon)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.secondary)
                if !model.runSelectedModelDisabledText.isEmpty {
                    Text(model.runSelectedModelDisabledText)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }
}

struct TranscriptWorkspace: View {
    @Bindable var model: ProbeViewModel
    @FocusState private var isTranscriptFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            workspaceHeader
            transcriptEditor
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .labGlassPanel(cornerRadius: 18, tint: .white.opacity(0.04), interactive: false)
    }

    private var workspaceHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Transcript")
                    .font(.system(size: 30, weight: .semibold, design: .rounded))
                Text(model.selectedModel.displayName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }

            Spacer()

            StatusCapsule(
                text: model.status,
                symbol: statusSymbol,
                tint: statusTint
            )

            if model.isRecording {
                Text(formatSeconds(model.recordingElapsedSeconds))
                    .font(.system(.title3, design: .monospaced, weight: .semibold))
                    .foregroundStyle(.red)
                    .monospacedDigit()
            }
        }
    }

    private var transcriptEditor: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $model.transcript)
                .font(.system(.body, design: .default))
                .focused($isTranscriptFocused)
                .scrollContentBackground(.hidden)
                .padding(14)
                .background(.black.opacity(0.12), in: .rect(cornerRadius: 12))
                .overlay {
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(.white.opacity(0.10), lineWidth: 1)
                }

            if model.transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isTranscriptFocused {
                Text("No output yet.")
                    .font(.body)
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 22)
                    .padding(.vertical, 22)
                    .allowsHitTesting(false)
            }
        }
    }

    private var statusSymbol: String {
        if model.isRecording { return "record.circle.fill" }
        if model.isTranscribing || model.isPreparingModel { return "hourglass" }
        if model.errorMessage != nil { return "exclamationmark.triangle.fill" }
        return "checkmark.circle.fill"
    }

    private var statusTint: Color {
        if model.isRecording { return .red }
        if model.isTranscribing || model.isPreparingModel { return .orange }
        if model.errorMessage != nil { return .red }
        return .green
    }

    private func formatSeconds(_ seconds: Double) -> String {
        ProbeViewModel.formatSeconds(seconds)
    }
}

struct PerformancePanel: View {
    let model: ProbeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionHeader(title: "Performance", symbol: "speedometer")

            VStack(spacing: 0) {
                MetricRow(title: "Audio length", value: formatSeconds(model.metrics.audioSeconds))
                MetricRow(title: "Audio load", value: formatSeconds(model.metrics.audioLoadSeconds))
                MetricRow(
                    title: model.metrics.wasModelAlreadyLoaded ? "Model cached" : "Model load",
                    value: formatSeconds(model.metrics.modelLoadSeconds)
                )
                MetricRow(title: "Generation", value: formatSeconds(model.metrics.generationSeconds))
                MetricRow(title: "Model reported", value: formatSeconds(model.metrics.modelReportedSeconds))
                MetricRow(title: "Total", value: formatSeconds(model.metrics.totalSeconds), showDivider: false)
            }

            if let errorMessage = model.errorMessage {
                ErrorNotice(message: errorMessage)
            }

            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(maxHeight: .infinity, alignment: .top)
        .labGlassPanel(cornerRadius: 16, tint: .green.opacity(0.06), interactive: false)
    }

    private func formatSeconds(_ seconds: Double) -> String {
        ProbeViewModel.formatSeconds(seconds)
    }
}

struct SystemPathsPanel: View {
    let model: ProbeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Local paths", symbol: "folder")
            PathLine(title: "Logs", value: model.logDirectoryPath)
            PathLine(title: "Models", value: model.modelCacheRootPath)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .labGlassPanel(cornerRadius: 16, tint: .white.opacity(0.03), interactive: false)
    }
}

struct SectionHeader: View {
    let title: String
    let symbol: String

    var body: some View {
        Label(title, systemImage: symbol)
            .font(.headline)
            .foregroundStyle(.primary)
    }
}

struct StatusCapsule: View {
    let text: String
    let symbol: String
    let tint: Color

    var body: some View {
        Label(text, systemImage: symbol)
            .font(.callout.weight(.medium))
            .lineLimit(1)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .foregroundStyle(tint)
            .labGlassPanel(cornerRadius: 14, tint: tint.opacity(0.12), interactive: false)
    }
}

struct MetricRow: View {
    let title: String
    let value: String
    var showDivider = true

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(title)
                    .font(.callout)
                    .foregroundStyle(.secondary)

                Spacer()

                Text(value)
                    .font(.system(.body, design: .monospaced, weight: .semibold))
                    .monospacedDigit()
                    .textSelection(.enabled)
            }
            .padding(.vertical, 10)

            if showDivider {
                Divider()
                    .opacity(0.45)
            }
        }
    }
}

struct ErrorNotice: View {
    let message: String

    var body: some View {
        Label {
            Text(message)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        } icon: {
            Image(systemName: "exclamationmark.triangle.fill")
        }
        .font(.caption)
        .foregroundStyle(.red)
        .padding(12)
        .labGlassPanel(cornerRadius: 12, tint: .red.opacity(0.10), interactive: false)
    }
}

struct PathLine: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption2.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
    }
}

private extension View {
    @ViewBuilder
    func labGlassPanel(
        cornerRadius: CGFloat,
        tint: Color,
        interactive: Bool
    ) -> some View {
        if #available(macOS 26.0, *) {
            if interactive {
                self.glassEffect(.regular.tint(tint).interactive(), in: .rect(cornerRadius: cornerRadius))
            } else {
                self.glassEffect(.regular.tint(tint), in: .rect(cornerRadius: cornerRadius))
            }
        } else {
            self
                .background(.ultraThinMaterial, in: .rect(cornerRadius: cornerRadius))
                .overlay {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .stroke(.white.opacity(0.12), lineWidth: 1)
                }
                .shadow(color: .black.opacity(0.18), radius: 18, y: 10)
        }
    }

    @ViewBuilder
    func labButtonStyle(prominent: Bool = false) -> some View {
        if #available(macOS 26.0, *) {
            if prominent {
                self.buttonStyle(.glassProminent)
            } else {
                self.buttonStyle(.glass)
            }
        } else {
            if prominent {
                self.buttonStyle(.borderedProminent)
            } else {
                self.buttonStyle(.bordered)
            }
        }
    }
}

@main
struct MLXAudioLabApp: App {
    @State private var model = ProbeViewModel()

    init() {
        ProbeLog.configureProcessOutput()
    }

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
        }
        .windowStyle(.titleBar)
    }
}
