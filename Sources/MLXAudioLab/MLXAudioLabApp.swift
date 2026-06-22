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
        cleanupTemporaryRecordings()

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
        status = "Transcribing with \(option.displayName)..."
        isTranscribing = true
        errorMessage = nil
        self.recordingURL = nil

        transcriptionTask?.cancel()
        transcriptionTask = Task {
            defer {
                try? FileManager.default.removeItem(at: recordingURL)
                ProbeLog.write("temporary recording deleted recording=\(recordingURL.lastPathComponent)")
            }

            do {
                let result = try await transcriber.transcribe(
                    recordingURL: recordingURL,
                    recordingSeconds: elapsed,
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
            return "Download the selected model before recording"
        case .incomplete:
            return "Repair the selected model cache before recording"
        }
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

    private func cleanupTemporaryRecordings() {
        let directory = Self.recordingDirectory()
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
        VStack(alignment: .leading, spacing: 18) {
            header
            modelPicker
            controls
            metricsView
            transcriptView
            footer
        }
        .padding(24)
        .frame(minWidth: 780, minHeight: 640)
        .onAppear {
            model.refreshModelAvailability(updateStatus: true)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("MLX Audio Lab")
                .font(.largeTitle)
                .fontWeight(.semibold)
            Text("Test local MLX audio models on macOS and compare transcription speed.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Picker("Model", selection: $model.selectedModelID) {
                    ForEach(AudioModelOption.supported) { option in
                        Text(option.displayName).tag(option.id)
                    }
                }
                .frame(maxWidth: 430)
                .disabled(model.modelControlsDisabled)

                Button {
                    model.prepareSelectedModel()
                } label: {
                    Label(model.selectedModelActionTitle, systemImage: model.selectedModelActionIcon)
                        .frame(minWidth: 150)
                }
                .disabled(!model.canPrepareSelectedModel)

                Button {
                    model.refreshModelAvailability(updateStatus: true)
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(model.modelControlsDisabled)

                Spacer()
            }

            HStack(spacing: 12) {
                Label(model.selectedModelStatusText, systemImage: model.selectedModelStatusIcon)
                    .foregroundStyle(model.selectedModelAvailability == .incomplete ? .orange : .secondary)
                Text(model.selectedModel.downloadSizeDescription)
                    .foregroundStyle(.secondary)
                Text(model.selectedModel.repoID)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
            .font(.callout)

            Text(model.selectedModel.subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .background(.quaternary.opacity(0.55), in: .rect(cornerRadius: 8))
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button {
                model.primaryButtonPressed()
            } label: {
                Label(model.primaryButtonTitle, systemImage: model.isRecording ? "stop.fill" : "record.circle")
                    .frame(minWidth: 120)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .tint(model.isRecording ? .red : .accentColor)
            .disabled(!model.canPressPrimaryButton)
            .accessibilityLabel(model.isRecording ? "Stop recording" : "Start recording")

            Button {
                model.clearOutput()
            } label: {
                Label("Clear", systemImage: "trash")
            }
            .controlSize(.large)
            .disabled(model.isRecording || model.isTranscribing || model.isPreparingModel)

            Spacer()

            VStack(alignment: .trailing, spacing: 2) {
                Text(model.status)
                    .font(.headline)
                Text(formatSeconds(model.recordingElapsedSeconds))
                    .font(.system(.title3, design: .monospaced))
                    .foregroundStyle(model.isRecording ? .red : .secondary)
            }
        }
    }

    private var metricsView: some View {
        Grid(alignment: .leading, horizontalSpacing: 28, verticalSpacing: 8) {
            GridRow {
                MetricCell(title: "Recording", value: formatSeconds(model.metrics.recordingSeconds))
                MetricCell(title: "Audio load", value: formatSeconds(model.metrics.audioLoadSeconds))
                MetricCell(
                    title: model.metrics.wasModelAlreadyLoaded ? "Model load cached" : "Model load",
                    value: formatSeconds(model.metrics.modelLoadSeconds)
                )
            }
            GridRow {
                MetricCell(title: "Generation", value: formatSeconds(model.metrics.generationSeconds))
                MetricCell(title: "Model reported", value: formatSeconds(model.metrics.modelReportedSeconds))
                MetricCell(title: "Total", value: formatSeconds(model.metrics.totalSeconds))
            }
        }
        .padding(14)
        .background(.quaternary.opacity(0.6), in: .rect(cornerRadius: 8))
    }

    private var transcriptView: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Output")
                .font(.headline)
            TextEditor(text: $model.transcript)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(8)
                .background(Color(nsColor: .textBackgroundColor), in: .rect(cornerRadius: 8))
                .overlay {
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.quaternary)
                }
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let errorMessage = model.errorMessage {
                Text(errorMessage)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            if !model.logDirectoryPath.isEmpty {
                Text("Logs: \(model.logDirectoryPath)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
            Text("Models: \(model.modelCacheRootPath)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
        }
    }

    private func formatSeconds(_ seconds: Double) -> String {
        String(format: "%.3f s", seconds)
    }
}

struct MetricCell: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
        }
        .frame(minWidth: 150, alignment: .leading)
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
