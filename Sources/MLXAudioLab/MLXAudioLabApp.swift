import AVFoundation
import Darwin
import Foundation
import MLXAudioCore
import MLXAudioSTT
import Observation
import SwiftUI

private let nemotronRepo = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"

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
    var recordingSeconds: Double = 0
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

actor NemotronTranscriber {
    private var model: NemotronASRModel?

    func transcribe(recordingURL: URL, recordingSeconds: Double) async throws -> TranscriptionResult {
        ProbeLog.write("transcribe begin recording=\(recordingURL.lastPathComponent) recordingSeconds=\(recordingSeconds)")
        let totalStart = ContinuousClock.now

        let audioLoadStart = ContinuousClock.now
        ProbeLog.write("audio load begin")
        let (_, audio) = try loadAudioArray(from: recordingURL, sampleRate: 16_000)
        let audioLoadSeconds = Self.seconds(since: audioLoadStart)
        ProbeLog.write("audio load complete seconds=\(audioLoadSeconds)")

        let wasLoaded = model != nil
        let modelLoadStart = ContinuousClock.now
        let loadedModel: NemotronASRModel
        if let model {
            ProbeLog.write("model already loaded")
            loadedModel = model
        } else {
            ProbeLog.write("model load begin repo=\(nemotronRepo)")
            let newModel = try await NemotronASRModel.fromPretrained(nemotronRepo)
            model = newModel
            loadedModel = newModel
            ProbeLog.write("model load complete")
        }
        let modelLoadSeconds = Self.seconds(since: modelLoadStart)

        let generationStart = ContinuousClock.now
        ProbeLog.write("generation begin")
        let output = loadedModel.generate(
            audio: audio,
            generationParameters: .init(language: "auto")
        )
        let generationSeconds = Self.seconds(since: generationStart)
        ProbeLog.write("generation complete seconds=\(generationSeconds) textLength=\(output.text.count)")

        let metrics = TranscriptionMetrics(
            recordingSeconds: recordingSeconds,
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
    var status = "Ready"
    var transcript = ""
    var errorMessage: String?
    var metrics = TranscriptionMetrics()
    var recordingElapsedSeconds: Double = 0
    var logDirectoryPath = ProbeLog.logDirectory().path

    private let transcriber = NemotronTranscriber()
    private var recorder: AVAudioRecorder?
    private var recordingStartedAt: Date?
    private var recordingURL: URL?
    private var timerTask: Task<Void, Never>?
    private var transcriptionTask: Task<Void, Never>?

    var primaryButtonTitle: String {
        if isRecording { return "Stop" }
        if isTranscribing { return "Working..." }
        return "Record"
    }

    var canPressPrimaryButton: Bool {
        !isTranscribing
    }

    func primaryButtonPressed() {
        if isRecording {
            stopRecordingAndTranscribe()
        } else {
            startRecording()
        }
    }

    func startRecording() {
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
                self.status = "Recording..."
                self.startTimer()
                ProbeLog.write("record started recording=\(url.lastPathComponent)")
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

        status = "Loading model and transcribing..."
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
                    recordingSeconds: elapsed
                )
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
        status = "Ready"
        metrics = TranscriptionMetrics()
        recordingElapsedSeconds = 0
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
            controls
            metricsView
            transcriptView
            footer
        }
        .padding(24)
        .frame(minWidth: 720, minHeight: 560)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("MLX Audio Lab")
                .font(.largeTitle)
                .fontWeight(.semibold)
            Text(nemotronRepo)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
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
            .disabled(model.isRecording || model.isTranscribing)

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
