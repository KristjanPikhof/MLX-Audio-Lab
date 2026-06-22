import AppKit
import AVFoundation
import Darwin
import Foundation
import HuggingFace
import MLXAudioCore
import MLXAudioSTT
import Observation
import SwiftUI
import UniformTypeIdentifiers

enum AudioModelFamily: String, Sendable {
    case nemotron = "Nemotron"
    case parakeet = "Parakeet"
    case qwen3ASR = "Qwen3 ASR"
    case whisper = "Whisper"
    case senseVoice = "SenseVoice"
    case glmASR = "GLM-ASR"
    case graniteSpeech = "Granite Speech"
    case voxtralRealtime = "Voxtral Realtime"
    case cohereTranscribe = "Cohere Transcribe"
}

struct AudioModelOption: Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let repoID: String
    let family: AudioModelFamily
    let downloadSizeDescription: String
    let downloadSizeBytes: Int64
    let subtitle: String
    let languageHint: String?
    let requiredFileNames: [String]

    static let supported: [AudioModelOption] = [
        AudioModelOption(
            id: "nemotron-streaming-0.6b-bf16",
            displayName: "Nemotron 3.5 ASR Streaming 0.6B bf16",
            repoID: "mlx-community/nemotron-3.5-asr-streaming-0.6b",
            family: .nemotron,
            downloadSizeDescription: "~1.28 GB",
            downloadSizeBytes: 1_276_296_562,
            subtitle: "Full-quality bf16 MLX conversion; larger than 8-bit but the recommended default.",
            languageHint: "auto",
            requiredFileNames: ["config.json", "model.safetensors", "vocab.txt"]
        ),
        AudioModelOption(
            id: "nemotron-streaming-0.6b-8bit",
            displayName: "Nemotron 3.5 ASR Streaming 0.6B 8-bit",
            repoID: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
            family: .nemotron,
            downloadSizeDescription: "~756 MB",
            downloadSizeBytes: 755_836_822,
            subtitle: "8-bit MLX conversion; smaller download and good for quick local checks.",
            languageHint: "auto",
            requiredFileNames: ["config.json", "model.safetensors", "vocab.txt"]
        ),
        AudioModelOption(
            id: "parakeet-tdt-0.6b-v3",
            displayName: "Parakeet TDT 0.6B v3",
            repoID: "mlx-community/parakeet-tdt-0.6b-v3",
            family: .parakeet,
            downloadSizeDescription: "~2.51 GB",
            downloadSizeBytes: 2_508_579_601,
            subtitle: "MLX conversion of NVIDIA Parakeet v3; multilingual ASR comparison target.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "vocab.txt"]
        ),
        AudioModelOption(
            id: "qwen3-asr-0.6b-4bit",
            displayName: "Qwen3 ASR 0.6B 4-bit",
            repoID: "mlx-community/Qwen3-ASR-0.6B-4bit",
            family: .qwen3ASR,
            downloadSizeDescription: "~708 MB",
            downloadSizeBytes: 708_000_000,
            subtitle: "Compact Qwen3 ASR conversion; good first comparison against Nemotron.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "merges.txt", "vocab.json"]
        ),
        AudioModelOption(
            id: "qwen3-asr-1.7b-4bit",
            displayName: "Qwen3 ASR 1.7B 4-bit",
            repoID: "mlx-community/Qwen3-ASR-1.7B-4bit",
            family: .qwen3ASR,
            downloadSizeDescription: "~1.6 GB",
            downloadSizeBytes: 1_600_000_000,
            subtitle: "Larger Qwen3 ASR checkpoint for quality and speed comparison.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "merges.txt", "vocab.json"]
        ),
        AudioModelOption(
            id: "whisper-large-v3-turbo-asr-fp16",
            displayName: "Whisper Large v3 Turbo ASR fp16",
            repoID: "mlx-community/whisper-large-v3-turbo-asr-fp16",
            family: .whisper,
            downloadSizeDescription: "~1.61 GB",
            downloadSizeBytes: 1_610_000_000,
            subtitle: "Whisper turbo baseline converted for mlx-audio.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "tokenizer.json", "merges.txt", "vocab.json"]
        ),
        AudioModelOption(
            id: "sensevoice-small",
            displayName: "SenseVoice Small",
            repoID: "mlx-community/SenseVoiceSmall",
            family: .senseVoice,
            downloadSizeDescription: "~936 MB",
            downloadSizeBytes: 936_000_000,
            subtitle: "Fast non-autoregressive ASR with language, emotion, and event metadata.",
            languageHint: nil,
            requiredFileNames: [
                "config.json",
                "model.safetensors",
                "am.mvn",
                "chn_jpn_yue_eng_ko_spectok.bpe.model"
            ]
        ),
        AudioModelOption(
            id: "glm-asr-nano-2512-4bit",
            displayName: "GLM-ASR Nano 2512 4-bit",
            repoID: "mlx-community/GLM-ASR-Nano-2512-4bit",
            family: .glmASR,
            downloadSizeDescription: "~1.28 GB",
            downloadSizeBytes: 1_280_000_000,
            subtitle: "Small GLM decoder ASR model; useful English/Chinese comparison target.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "tokenizer.json"]
        ),
        AudioModelOption(
            id: "granite-4.0-1b-speech-5bit",
            displayName: "Granite 4.0 1B Speech 5-bit",
            repoID: "mlx-community/granite-4.0-1b-speech-5bit",
            family: .graniteSpeech,
            downloadSizeDescription: "~2.22 GB",
            downloadSizeBytes: 2_220_000_000,
            subtitle: "IBM Granite speech model for ASR and translation-style experiments.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "tokenizer.json", "merges.txt", "vocab.json"]
        ),
        AudioModelOption(
            id: "voxtral-mini-4b-realtime-4bit",
            displayName: "Voxtral Mini 4B Realtime 4-bit",
            repoID: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            family: .voxtralRealtime,
            downloadSizeDescription: "~3.13 GB",
            downloadSizeBytes: 3_130_000_000,
            subtitle: "Heavy streaming STT model; this app benchmarks it through offline chunks.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "tekken.json"]
        ),
        AudioModelOption(
            id: "cohere-transcribe-03-2026-fp16",
            displayName: "Cohere Transcribe 03-2026 fp16",
            repoID: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
            family: .cohereTranscribe,
            downloadSizeDescription: "~3.85 GiB",
            downloadSizeBytes: 4_133_263_974,
            subtitle: "Community MLX conversion of Cohere Transcribe; large experimental baseline.",
            languageHint: nil,
            requiredFileNames: ["config.json", "model.safetensors", "tokenizer.model", "tokenizer_config.json"]
        )
    ]
}

enum ModelLocalAvailability: Sendable {
    case available
    case notDownloaded
    case incomplete
}

enum ModelCache {
    static func hubCacheRootDirectory() -> URL {
        let env = ProcessInfo.processInfo.environment

        if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
            return URL(fileURLWithPath: expandTilde(hubCache), isDirectory: true)
        }

        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: expandTilde(hfHome), isDirectory: true)
                .appending(path: "hub", directoryHint: .isDirectory)
        }

        return FileManager.default.homeDirectoryForCurrentUser
            .appending(path: ".cache/huggingface/hub", directoryHint: .isDirectory)
    }

    static func rootDirectory() -> URL {
        hubCacheRootDirectory().appending(path: "mlx-audio", directoryHint: .isDirectory)
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
              requiredFilesExist(for: option, in: directory),
              containsNonEmptyFile(withExtension: "safetensors", in: directory)
        else {
            return .incomplete
        }

        return .available
    }

    static func delete(_ option: AudioModelOption) throws {
        let fileManager = FileManager.default
        let candidates = [
            directory(for: option),
            huggingFaceRepositoryDirectory(for: option),
            huggingFaceLockDirectory(for: option)
        ].compactMap(\.self)

        for directory in candidates {
            try removeDirectoryIfPresent(directory, fileManager: fileManager)
        }
    }

    static func deleteAppCache(_ option: AudioModelOption) throws {
        try removeDirectoryIfPresent(directory(for: option), fileManager: .default)
    }

    private static func huggingFaceRepositoryDirectory(for option: AudioModelOption) -> URL? {
        let parts = option.repoID.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }

        return hubCacheRootDirectory()
            .appending(path: "models--\(parts[0])--\(parts[1])", directoryHint: .isDirectory)
    }

    private static func huggingFaceLockDirectory(for option: AudioModelOption) -> URL? {
        guard let repositoryDirectory = huggingFaceRepositoryDirectory(for: option) else { return nil }

        return hubCacheRootDirectory()
            .appending(path: ".locks", directoryHint: .isDirectory)
            .appending(path: repositoryDirectory.lastPathComponent, directoryHint: .isDirectory)
    }

    private static func requiredFilesExist(for option: AudioModelOption, in directory: URL) -> Bool {
        option.requiredFileNames.allSatisfy { fileName in
            let fileURL = directory.appending(path: fileName)
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        }
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

    private static func removeDirectoryIfPresent(_ directory: URL, fileManager: FileManager) throws {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return
        }

        try fileManager.removeItem(at: directory)
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
            return "Imported media"
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
    private static let writeLock = NSLock()

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

        writeLock.lock()
        defer {
            writeLock.unlock()
        }

        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }

        guard let data = line.data(using: .utf8),
              let handle = try? FileHandle(forWritingTo: url)
        else {
            return
        }

        do {
            defer {
                try? handle.close()
            }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
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

struct ModelDownloadProgress: Equatable, Sendable {
    let modelName: String
    let completedBytes: Int64
    let totalBytes: Int64

    init(modelName: String, progress: Progress, fallbackTotalBytes: Int64) {
        self.modelName = modelName
        completedBytes = max(progress.completedUnitCount, 0)
        totalBytes = max(progress.totalUnitCount, fallbackTotalBytes)
    }

    init(modelName: String, completedBytes: Int64, totalBytes: Int64) {
        self.modelName = modelName
        self.completedBytes = max(completedBytes, 0)
        self.totalBytes = max(totalBytes, 0)
    }

    func withCompletedBytes(_ bytes: Int64) -> ModelDownloadProgress {
        ModelDownloadProgress(
            modelName: modelName,
            completedBytes: bytes,
            totalBytes: totalBytes
        )
    }

    var fractionCompleted: Double? {
        guard totalBytes > 0 else { return nil }
        return min(max(Double(displayedCompletedBytes) / Double(totalBytes), 0), 1)
    }

    var percentageText: String {
        guard let fractionCompleted else { return "Preparing" }
        return "\(Int((fractionCompleted * 100).rounded(.down)))%"
    }

    var byteProgressText: String {
        let unit = ByteDisplayUnit.unit(for: max(totalBytes, displayedCompletedBytes))
        let completed = unit.string(fromByteCount: displayedCompletedBytes)
        guard totalBytes > 0 else { return completed }

        let total = unit.string(fromByteCount: totalBytes)
        return "\(completed) of \(total)"
    }

    private var displayedCompletedBytes: Int64 {
        guard totalBytes > 0 else { return completedBytes }
        return min(completedBytes, totalBytes)
    }

    private enum ByteDisplayUnit {
        case megabytes
        case gigabytes

        static func unit(for bytes: Int64) -> ByteDisplayUnit {
            bytes >= 1_073_741_824 ? .gigabytes : .megabytes
        }

        func string(fromByteCount bytes: Int64) -> String {
            let value = Double(max(bytes, 0)) / Double(bytesPerUnit)
            return "\(String(format: "%.1f", value)) \(suffix)"
        }

        private var bytesPerUnit: Int64 {
            switch self {
            case .megabytes:
                return 1_048_576
            case .gigabytes:
                return 1_073_741_824
            }
        }

        private var suffix: String {
            switch self {
            case .megabytes:
                return "MB"
            case .gigabytes:
                return "GB"
            }
        }
    }
}

private final class WAVConversionSession: @unchecked Sendable {
    private let reader: AVAssetReader
    private let output: AVAssetReaderAudioMixOutput
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let processingQueue = DispatchQueue(label: "MLXAudioLab.wav-conversion", qos: .userInitiated)

    private var continuation: CheckedContinuation<Void, Error>?
    private var didResume = false

    init(
        reader: AVAssetReader,
        output: AVAssetReaderAudioMixOutput,
        writer: AVAssetWriter,
        input: AVAssetWriterInput
    ) {
        self.reader = reader
        self.output = output
        self.writer = writer
        self.input = input
    }

    func run() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            self.continuation = continuation
            input.requestMediaDataWhenReady(on: processingQueue) {
                self.processAvailableMediaData()
            }
        }
    }

    private func processAvailableMediaData() {
        while input.isReadyForMoreMediaData {
            guard reader.status == .reading else {
                completeWriter()
                return
            }

            guard let sampleBuffer = output.copyNextSampleBuffer() else {
                completeWriter()
                return
            }

            guard input.append(sampleBuffer) else {
                reader.cancelReading()
                writer.cancelWriting()
                complete(.failure(writer.error ?? Self.makeError("The selected media file could not be converted to WAV.")))
                return
            }
        }
    }

    private func completeWriter() {
        input.markAsFinished()

        if reader.status == .failed || reader.status == .cancelled {
            writer.cancelWriting()
            complete(.failure(reader.error ?? Self.makeError("The selected media file could not be read.")))
            return
        }

        writer.finishWriting {
            self.processingQueue.async {
                guard self.writer.status == .completed else {
                    self.complete(
                        .failure(
                            self.writer.error ?? Self.makeError("The selected media file could not be converted to WAV.")
                        )
                    )
                    return
                }

                self.complete(.success(()))
            }
        }
    }

    private func complete(_ result: Result<Void, Error>) {
        guard !didResume else { return }
        didResume = true

        guard let continuation else { return }
        self.continuation = nil

        switch result {
        case .success:
            continuation.resume()
        case .failure(let error):
            continuation.resume(throwing: error)
        }
    }

    private static func makeError(_ description: String) -> NSError {
        NSError(
            domain: "MLXAudioLab",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: description]
        )
    }
}

private final class ModelDownloadProgressState: @unchecked Sendable {
    private let lock = NSLock()
    private let modelName: String
    private let totalBytes: Int64
    private var completedBytes: Int64 = 0
    private var currentFileBytes: Int64 = 0
    private var currentExpectedBytes: Int64 = 0

    init(modelName: String, totalBytes: Int64) {
        self.modelName = modelName
        self.totalBytes = totalBytes
    }

    func beginFile(expectedBytes: Int64) {
        lock.lock()
        defer {
            lock.unlock()
        }

        currentExpectedBytes = max(expectedBytes, 0)
        currentFileBytes = 0
    }

    func completeFile(expectedBytes: Int64) {
        lock.lock()
        defer {
            lock.unlock()
        }

        completedBytes += max(expectedBytes, 0)
        currentFileBytes = 0
        currentExpectedBytes = 0
    }

    func updateCurrentExpectedBytes(_ bytes: Int64) {
        guard bytes > 0 else { return }

        lock.lock()
        defer {
            lock.unlock()
        }

        currentExpectedBytes = max(currentExpectedBytes, bytes)
    }

    func updateCurrentFileBytes(_ bytes: Int64) {
        lock.lock()
        defer {
            lock.unlock()
        }

        currentFileBytes = max(currentFileBytes, bytes)
    }

    func snapshot() -> ModelDownloadProgress {
        lock.lock()
        defer {
            lock.unlock()
        }

        let currentBytes = min(
            max(currentFileBytes, 0),
            currentExpectedBytes > 0 ? currentExpectedBytes : Int64.max
        )
        return ModelDownloadProgress(
            modelName: modelName,
            completedBytes: completedBytes + currentBytes,
            totalBytes: totalBytes
        )
    }
}

private final class ModelDownloadTaskBox: @unchecked Sendable {
    private let lock = NSLock()
    private var task: URLSessionDownloadTask?

    func set(_ task: URLSessionDownloadTask) {
        lock.lock()
        self.task = task
        lock.unlock()
    }

    func resume() {
        lock.lock()
        let task = task
        lock.unlock()
        task?.resume()
    }

    func cancel() {
        lock.lock()
        let task = task
        lock.unlock()
        task?.cancel()
    }
}

private final class ModelFileDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let progressState: ModelDownloadProgressState
    private let temporaryDestination: URL
    private let repoID: String
    private let fileName: String
    private var continuation: CheckedContinuation<URL, Error>?
    private var downloadedURL: URL?
    private var finishError: Error?
    private var lastLoggedBytes: Int64 = 0

    init(
        progressState: ModelDownloadProgressState,
        temporaryDestination: URL,
        repoID: String,
        fileName: String
    ) {
        self.progressState = progressState
        self.temporaryDestination = temporaryDestination
        self.repoID = repoID
        self.fileName = fileName
    }

    func setContinuation(_ continuation: CheckedContinuation<URL, Error>) {
        lock.lock()
        self.continuation = continuation
        lock.unlock()
    }

    func urlSession(
        _: URLSession,
        downloadTask _: URLSessionDownloadTask,
        didWriteData _: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        if totalBytesExpectedToWrite > 0 {
            progressState.updateCurrentExpectedBytes(totalBytesExpectedToWrite)
        }
        progressState.updateCurrentFileBytes(totalBytesWritten)
        logProgressIfNeeded(bytes: totalBytesWritten, expectedBytes: totalBytesExpectedToWrite)
    }

    func urlSession(
        _: URLSession,
        downloadTask _: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            try? FileManager.default.removeItem(at: temporaryDestination)
            try FileManager.default.moveItem(at: location, to: temporaryDestination)

            lock.lock()
            downloadedURL = temporaryDestination
            lock.unlock()
        } catch {
            lock.lock()
            finishError = error
            lock.unlock()
        }
    }

    func urlSession(
        _: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            if (error as? URLError)?.code == .cancelled {
                finish(.failure(CancellationError()))
            } else {
                finish(.failure(error))
            }
            return
        }

        if let httpResponse = task.response as? HTTPURLResponse,
           !(200..<300).contains(httpResponse.statusCode) {
            finish(
                .failure(
                    NSError(
                        domain: "MLXAudioLab.ModelDownload",
                        code: httpResponse.statusCode,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Hugging Face returned HTTP \(httpResponse.statusCode) for \(fileName)."
                        ]
                    )
                )
            )
            return
        }

        lock.lock()
        let finishError = finishError
        let downloadedURL = downloadedURL
        lock.unlock()

        if let finishError {
            finish(.failure(finishError))
        } else if let downloadedURL {
            finish(.success(downloadedURL))
        } else {
            finish(
                .failure(
                    NSError(
                        domain: "MLXAudioLab.ModelDownload",
                        code: -1,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "The download finished without a temporary file for \(fileName)."
                        ]
                    )
                )
            )
        }
    }

    private func finish(_ result: Result<URL, Error>) {
        lock.lock()
        guard let continuation else {
            lock.unlock()
            return
        }
        self.continuation = nil
        lock.unlock()

        switch result {
        case .success(let url):
            continuation.resume(returning: url)
        case .failure(let error):
            continuation.resume(throwing: error)
        }
    }

    private func logProgressIfNeeded(bytes: Int64, expectedBytes: Int64) {
        let stepBytes: Int64 = 32 * 1024 * 1024
        let shouldLog: Bool

        lock.lock()
        if bytes - lastLoggedBytes >= stepBytes
            || (expectedBytes > 0 && bytes >= expectedBytes && lastLoggedBytes < expectedBytes) {
            lastLoggedBytes = bytes
            shouldLog = true
        } else {
            shouldLog = false
        }
        lock.unlock()

        if shouldLog {
            ProbeLog.write(
                "model file download progress repo=\(repoID) file=\(fileName) bytes=\(bytes) expected=\(expectedBytes)"
            )
        }
    }
}

struct TranscriptStatistics: Sendable {
    let words: Int
    let letters: Int
    let characters: Int
    let lines: Int

    static let empty = TranscriptStatistics(words: 0, letters: 0, characters: 0, lines: 0)

    static func calculate(from text: String) -> TranscriptStatistics {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else {
            return .empty
        }

        let words = trimmedText.split { character in
            !character.isLetter && !character.isNumber
        }.count
        let letters = trimmedText.filter(\.isLetter).count
        let lines = trimmedText.components(separatedBy: .newlines).count

        return TranscriptStatistics(
            words: words,
            letters: letters,
            characters: trimmedText.count,
            lines: lines
        )
    }
}

actor AudioModelTranscriber {
    private static let sampleRate = 16_000
    private static let safeDecodeChunkDurationSeconds: Float = 30

    private var loadedModels: [String: any STTGenerationModel] = [:]
    private var loadingModelIDs: Set<String> = []
    private var modelLoadErrors: [String: Error] = [:]
    private var modelLoadWaiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    func prepareModel(
        _ option: AudioModelOption,
        onDownloadProgress: (@MainActor @Sendable (ModelDownloadProgress?) -> Void)? = nil
    ) async throws -> ModelPreparationResult {
        ProbeLog.write("model prepare requested repo=\(option.repoID)")
        let start = ContinuousClock.now
        let wasLoaded = loadedModels[option.id] != nil
        if !wasLoaded {
            try await ensureModelCached(option, onDownloadProgress: onDownloadProgress)
            await onDownloadProgress?(nil)
        }
        _ = try await loadedModel(for: option)
        let seconds = Self.seconds(since: start)
        ProbeLog.write("model prepare complete repo=\(option.repoID) seconds=\(seconds) wasLoaded=\(wasLoaded)")
        return ModelPreparationResult(modelLoadSeconds: seconds, wasModelAlreadyLoaded: wasLoaded)
    }

    func unloadModel(id: String) {
        loadedModels[id] = nil
    }

    func transcribe(
        audioURL: URL,
        audioSeconds: Double,
        using option: AudioModelOption,
        onPartialTranscript: (@Sendable (String) async -> Void)? = nil,
        onMetricsUpdate: (@Sendable (TranscriptionMetrics) async -> Void)? = nil
    ) async throws -> TranscriptionResult {
        ProbeLog.write(
            "transcribe begin repo=\(option.repoID) audio=\(audioURL.lastPathComponent) audioSeconds=\(audioSeconds)"
        )
        let totalStart = ContinuousClock.now
        var metrics = TranscriptionMetrics(audioSeconds: audioSeconds)
        await onMetricsUpdate?(metrics)
        try Task.checkCancellation()

        let audioLoadStart = ContinuousClock.now
        ProbeLog.write("audio load begin")
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: Self.sampleRate)
        let audioLoadSeconds = Self.seconds(since: audioLoadStart)
        let totalSamples = audio.shape[0]
        guard totalSamples > 0 else {
            throw Self.makeError("The selected audio file has no readable samples.")
        }
        ProbeLog.write("audio load complete seconds=\(audioLoadSeconds) samples=\(totalSamples)")
        metrics.audioLoadSeconds = audioLoadSeconds
        metrics.totalSeconds = Self.seconds(since: totalStart)
        await onMetricsUpdate?(metrics)
        try Task.checkCancellation()

        let modelLoadStart = ContinuousClock.now
        let wasLoaded = loadedModels[option.id] != nil
        let loadedModel = try await loadedModel(for: option)
        let modelLoadSeconds = Self.seconds(since: modelLoadStart)
        metrics.modelLoadSeconds = modelLoadSeconds
        metrics.wasModelAlreadyLoaded = wasLoaded
        metrics.totalSeconds = Self.seconds(since: totalStart)
        await onMetricsUpdate?(metrics)
        try Task.checkCancellation()

        let generationStart = ContinuousClock.now
        let parameters = generationParameters(for: loadedModel, option: option)
        let chunkSamples = Self.chunkSampleCount(for: parameters)
        let totalChunks = Int(ceil(Double(totalSamples) / Double(chunkSamples)))
        ProbeLog.write(
            "generation begin repo=\(option.repoID) chunkSeconds=\(parameters.chunkDuration) chunks=\(totalChunks) samples=\(totalSamples)"
        )
        var transcriptParts: [String] = []
        var modelReportedSeconds = 0.0

        for chunkIndex in 0..<totalChunks {
            try Task.checkCancellation()

            let startSample = chunkIndex * chunkSamples
            let endSample = min(startSample + chunkSamples, totalSamples)
            let chunkAudio = audio[startSample..<endSample]
            let chunkDurationSeconds = Double(endSample - startSample) / Double(Self.sampleRate)
            ProbeLog.write(
                "generation chunk begin repo=\(option.repoID) index=\(chunkIndex + 1)/\(totalChunks) durationSeconds=\(chunkDurationSeconds)"
            )

            let output = loadedModel.generate(
                audio: chunkAudio,
                generationParameters: parameters
            )
            modelReportedSeconds += output.totalTime
            Self.updateGenerationMetrics(
                &metrics,
                generationStart: generationStart,
                modelReportedSeconds: modelReportedSeconds,
                totalStart: totalStart
            )
            await onMetricsUpdate?(metrics)

            try Task.checkCancellation()
            let chunkText = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !chunkText.isEmpty {
                transcriptParts.append(chunkText)
            }

            let partialTranscript = transcriptParts.joined(separator: " ")
            if !partialTranscript.isEmpty {
                await onPartialTranscript?(partialTranscript)
                ProbeLog.write(
                    "partial transcript emitted repo=\(option.repoID) index=\(chunkIndex + 1)/\(totalChunks) textLength=\(partialTranscript.count)"
                )
            }

            ProbeLog.write(
                "generation chunk complete repo=\(option.repoID) index=\(chunkIndex + 1)/\(totalChunks) textLength=\(output.text.count)"
            )
        }

        let transcript = transcriptParts.joined(separator: " ")
        let generationSeconds = Self.seconds(since: generationStart)
        ProbeLog.write("generation complete seconds=\(generationSeconds) textLength=\(transcript.count)")

        Self.updateGenerationMetrics(
            &metrics,
            generationSeconds: generationSeconds,
            modelReportedSeconds: modelReportedSeconds,
            totalStart: totalStart
        )
        await onMetricsUpdate?(metrics)

        let result = TranscriptionResult(text: transcript, metrics: metrics)
        ProbeLog.write("transcribe complete totalSeconds=\(metrics.totalSeconds)")
        return result
    }

    private func loadedModel(for option: AudioModelOption) async throws -> any STTGenerationModel {
        if let model = loadedModels[option.id] {
            ProbeLog.write("model already loaded repo=\(option.repoID)")
            return model
        }

        if loadingModelIDs.contains(option.id) {
            await waitForModelLoad(id: option.id)
            try Task.checkCancellation()

            if let model = loadedModels[option.id] {
                ProbeLog.write("model joined in-flight load repo=\(option.repoID)")
                return model
            }

            if let error = modelLoadErrors[option.id] {
                throw error
            }

            throw Self.makeError("The in-flight model load finished without a loaded model.")
        }

        ProbeLog.write("model load begin repo=\(option.repoID)")
        loadingModelIDs.insert(option.id)
        modelLoadErrors[option.id] = nil
        defer {
            loadingModelIDs.remove(option.id)
            resumeModelLoadWaiters(id: option.id)
        }

        do {
            let model = try await Self.loadModel(for: option)
            loadedModels[option.id] = model
            ProbeLog.write("model load complete repo=\(option.repoID)")
            return model
        } catch {
            modelLoadErrors[option.id] = error
            throw error
        }
    }

    private func waitForModelLoad(id: String) async {
        await withCheckedContinuation { continuation in
            modelLoadWaiters[id, default: []].append(continuation)
        }
    }

    private func resumeModelLoadWaiters(id: String) {
        let waiters = modelLoadWaiters.removeValue(forKey: id) ?? []
        for waiter in waiters {
            waiter.resume()
        }
    }

    private static func loadModel(for option: AudioModelOption) async throws -> any STTGenerationModel {
        let model: any STTGenerationModel
        switch option.family {
        case .nemotron:
            model = try await NemotronASRModel.fromPretrained(option.repoID)
        case .parakeet:
            model = try await ParakeetModel.fromPretrained(option.repoID)
        case .qwen3ASR:
            model = try await Qwen3ASRModel.fromPretrained(option.repoID)
        case .whisper:
            model = try await WhisperModel.fromPretrained(option.repoID)
        case .senseVoice:
            model = try await SenseVoiceModel.fromPretrained(option.repoID)
        case .glmASR:
            model = try await GLMASRModel.fromPretrained(option.repoID)
        case .graniteSpeech:
            model = try await GraniteSpeechModel.fromPretrained(option.repoID)
        case .voxtralRealtime:
            model = try await VoxtralRealtimeModel.fromPretrained(option.repoID)
        case .cohereTranscribe:
            model = try await CohereTranscribeModel.fromPretrained(option.repoID)
        }

        return model
    }

    private func ensureModelCached(
        _ option: AudioModelOption,
        onDownloadProgress: (@MainActor @Sendable (ModelDownloadProgress?) -> Void)?
    ) async throws {
        guard ModelCache.availability(for: option) != .available else { return }

        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        let cache = HubCache.default
        let client: HubClient
        if let hfToken, !hfToken.isEmpty {
            client = HubClient(host: HubClient.defaultHost, bearerToken: hfToken, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }

        guard let repoID = Repo.ID(rawValue: option.repoID) else {
            throw Self.makeError("Invalid repository ID: \(option.repoID)")
        }

        if ModelCache.availability(for: option) == .incomplete {
            try ModelCache.deleteAppCache(option)
        }

        ProbeLog.write("model cache download begin repo=\(option.repoID)")
        try await downloadModelSnapshot(
            client: client,
            repoID: repoID,
            option: option,
            hfToken: hfToken,
            onDownloadProgress: onDownloadProgress
        )
        ProbeLog.write("model cache download complete repo=\(option.repoID)")
    }

    private func downloadModelSnapshot(
        client: HubClient,
        repoID: Repo.ID,
        option: AudioModelOption,
        hfToken: String?,
        onDownloadProgress: (@MainActor @Sendable (ModelDownloadProgress?) -> Void)?
    ) async throws {
        let modelDirectory = ModelCache.directory(for: option)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)

        let entries = try await downloadableEntries(client: client, repoID: repoID, option: option)
        guard entries.contains(where: { URL(fileURLWithPath: $0.path).pathExtension.lowercased() == "safetensors" }) else {
            throw Self.makeError("No model.safetensors file was found in \(option.repoID).")
        }

        let totalBytes = entries.reduce(Int64(0)) { partial, entry in
            partial + max(Int64(entry.size ?? 1), 1)
        }
        let progressState = ModelDownloadProgressState(
            modelName: option.displayName,
            totalBytes: max(totalBytes, 1)
        )
        let progressSampler = startModelProgressSampler(
            state: progressState,
            onDownloadProgress: onDownloadProgress
        )

        await emit(state: progressState, onDownloadProgress: onDownloadProgress)

        do {
            for entry in entries {
                try Task.checkCancellation()

                let fileBytes = max(Int64(entry.size ?? 1), 1)
                progressState.beginFile(expectedBytes: fileBytes)
                let destination = modelDirectory.appending(path: entry.path)

                if let cachedPath = client.cache?.cachedFilePath(
                    repo: repoID,
                    kind: .model,
                    revision: "main",
                    filename: entry.path
                ) {
                    ProbeLog.write(
                        "model file copy begin repo=\(option.repoID) file=\(entry.path) bytes=\(fileBytes)"
                    )
                    try await copyFileWithProgress(
                        from: cachedPath,
                        to: destination,
                        progressState: progressState
                    )
                    ProbeLog.write(
                        "model file copy complete repo=\(option.repoID) file=\(entry.path) bytes=\(fileBytes)"
                    )
                } else {
                    ProbeLog.write(
                        "model file download begin repo=\(option.repoID) file=\(entry.path) bytes=\(fileBytes)"
                    )
                    try await downloadFileWithProgress(
                        client: client,
                        repoID: repoID,
                        repoIDText: option.repoID,
                        entry: entry,
                        destination: destination,
                        expectedBytes: fileBytes,
                        hfToken: hfToken,
                        progressState: progressState
                    )
                    ProbeLog.write(
                        "model file download complete repo=\(option.repoID) file=\(entry.path) bytes=\(fileBytes)"
                    )
                }
                progressState.completeFile(expectedBytes: fileBytes)
                await emit(state: progressState, onDownloadProgress: onDownloadProgress)
            }
        } catch {
            await stopDownloadProgressSampler(progressSampler)
            throw error
        }

        await stopDownloadProgressSampler(progressSampler)

        await emit(state: progressState, onDownloadProgress: onDownloadProgress)

        guard ModelCache.availability(for: option) == .available else {
            throw Self.makeError("The model download finished, but required files are still missing.")
        }
    }

    private func downloadableEntries(
        client: HubClient,
        repoID: Repo.ID,
        option: AudioModelOption
    ) async throws -> [Git.TreeEntry] {
        let patterns = Set([
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.wav"
        ] + option.requiredFileNames)
        let entries = try await client.listFiles(
            in: repoID,
            kind: .model,
            revision: "main",
            recursive: true
        )

        return entries
            .filter { entry in
                guard entry.type == .file else { return false }
                return patterns.contains { pattern in
                    fnmatch(pattern, entry.path, 0) == 0
                }
            }
            .sorted {
                let leftSize = $0.size ?? 0
                let rightSize = $1.size ?? 0
                if leftSize == rightSize {
                    return $0.path < $1.path
                }
                return leftSize > rightSize
            }
    }

    private func copyFileWithProgress(
        from source: URL,
        to destination: URL,
        progressState: ModelDownloadProgressState
    ) async throws {
        try await Task.detached(priority: .utility) {
            try Self.copyFileWithProgressSynchronously(
                from: source,
                to: destination,
                progressState: progressState
            )
        }.value
    }

    private nonisolated static func copyFileWithProgressSynchronously(
        from source: URL,
        to destination: URL,
        progressState: ModelDownloadProgressState
    ) throws {
        let fileManager = FileManager.default
        let resolvedSource = source.resolvingSymlinksInPath()
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let temporaryDestination = destination
            .deletingLastPathComponent()
            .appending(path: ".\(destination.lastPathComponent).partial-\(UUID().uuidString)")
        fileManager.createFile(atPath: temporaryDestination.path, contents: nil)

        let input = try FileHandle(forReadingFrom: resolvedSource)
        let output = try FileHandle(forWritingTo: temporaryDestination)
        var didFinish = false
        defer {
            try? input.close()
            try? output.close()
            if !didFinish {
                try? fileManager.removeItem(at: temporaryDestination)
            }
        }

        while true {
            try Task.checkCancellation()
            guard let data = try input.read(upToCount: 1024 * 1024),
                  !data.isEmpty
            else {
                break
            }

            try output.write(contentsOf: data)
            let writtenBytes = try output.offset()
            progressState.updateCurrentFileBytes(Int64(writtenBytes))
        }

        try? fileManager.removeItem(at: destination)
        try fileManager.moveItem(at: temporaryDestination, to: destination)
        didFinish = true
    }

    private func downloadFileWithProgress(
        client: HubClient,
        repoID: Repo.ID,
        repoIDText: String,
        entry: Git.TreeEntry,
        destination: URL,
        expectedBytes: Int64,
        hfToken: String?,
        progressState: ModelDownloadProgressState
    ) async throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let temporaryDestination = destination
            .deletingLastPathComponent()
            .appending(path: ".\(destination.lastPathComponent).download-\(UUID().uuidString)")
        try? fileManager.removeItem(at: temporaryDestination)

        var request = URLRequest(url: resolveURL(client: client, repoID: repoID, entry: entry))
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if let userAgent = client.userAgent {
            request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        }
        let bearerToken = if let hfToken, !hfToken.isEmpty {
            hfToken
        } else {
            await client.bearerToken
        }
        if let bearerToken, !bearerToken.isEmpty {
            request.setValue("Bearer \(bearerToken)", forHTTPHeaderField: "Authorization")
        }

        let delegate = ModelFileDownloadDelegate(
            progressState: progressState,
            temporaryDestination: temporaryDestination,
            repoID: repoIDText,
            fileName: entry.path
        )
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        let taskBox = ModelDownloadTaskBox()
        taskBox.set(session.downloadTask(with: request))

        defer {
            session.invalidateAndCancel()
            try? fileManager.removeItem(at: temporaryDestination)
        }

        let downloadedTemporaryURL = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<URL, Error>) in
                delegate.setContinuation(continuation)
                taskBox.resume()
            }
        } onCancel: {
            taskBox.cancel()
        }

        let downloadedBytes = fileSize(at: downloadedTemporaryURL)
        if downloadedBytes > 0 {
            progressState.updateCurrentFileBytes(downloadedBytes)
        } else if expectedBytes > 1 {
            throw Self.makeError("Downloaded \(entry.path) was empty.")
        }

        try? fileManager.removeItem(at: destination)
        try fileManager.moveItem(at: downloadedTemporaryURL, to: destination)
    }

    private func resolveURL(client: HubClient, repoID: Repo.ID, entry: Git.TreeEntry) -> URL {
        client.host
            .appending(path: repoID.namespace)
            .appending(path: repoID.name)
            .appending(path: "resolve")
            .appending(component: "main")
            .appending(path: entry.path)
    }

    private func fileSize(at url: URL) -> Int64 {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        return Int64(size)
    }

    private func startModelProgressSampler(
        state: ModelDownloadProgressState,
        onDownloadProgress: (@MainActor @Sendable (ModelDownloadProgress?) -> Void)?
    ) -> Task<Void, Never>? {
        guard let onDownloadProgress else { return nil }

        return Task.detached(priority: .utility) { [onDownloadProgress, state] in
            while !Task.isCancelled {
                await onDownloadProgress(state.snapshot())

                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    private func emit(
        state: ModelDownloadProgressState,
        onDownloadProgress: (@MainActor @Sendable (ModelDownloadProgress?) -> Void)?
    ) async {
        await onDownloadProgress?(state.snapshot())
    }

    private func stopDownloadProgressSampler(_ sampler: Task<Void, Never>?) async {
        sampler?.cancel()
        if let sampler {
            _ = await sampler.result
        }
    }

    private func generationParameters(
        for model: any STTGenerationModel,
        option: AudioModelOption
    ) -> STTGenerateParameters {
        let defaults = model.defaultGenerationParameters
        let chunkDuration = defaults.chunkDuration <= 0
            ? defaults.chunkDuration
            : min(defaults.chunkDuration, Self.safeDecodeChunkDurationSeconds)
        return STTGenerateParameters(
            maxTokens: defaults.maxTokens,
            temperature: defaults.temperature,
            topP: defaults.topP,
            topK: defaults.topK,
            verbose: false,
            language: option.languageHint ?? defaults.language,
            chunkDuration: chunkDuration,
            minChunkDuration: defaults.minChunkDuration,
            repetitionPenalty: defaults.repetitionPenalty,
            repetitionContextSize: defaults.repetitionContextSize
        )
    }

    private static func updateGenerationMetrics(
        _ metrics: inout TranscriptionMetrics,
        generationStart: ContinuousClock.Instant,
        modelReportedSeconds: Double,
        totalStart: ContinuousClock.Instant
    ) {
        updateGenerationMetrics(
            &metrics,
            generationSeconds: seconds(since: generationStart),
            modelReportedSeconds: modelReportedSeconds,
            totalStart: totalStart
        )
    }

    private static func updateGenerationMetrics(
        _ metrics: inout TranscriptionMetrics,
        generationSeconds: Double,
        modelReportedSeconds: Double,
        totalStart: ContinuousClock.Instant
    ) {
        metrics.generationSeconds = generationSeconds
        metrics.modelReportedSeconds = modelReportedSeconds
        metrics.totalSeconds = seconds(since: totalStart)
    }

    private static func chunkSampleCount(for parameters: STTGenerateParameters) -> Int {
        let chunkDuration = parameters.chunkDuration > 0
            ? Double(parameters.chunkDuration)
            : Double(safeDecodeChunkDurationSeconds)
        return max(1, Int(chunkDuration * Double(sampleRate)))
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: ContinuousClock.now)
        return Double(duration.components.seconds) +
            Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }

    private static func makeError(_ message: String) -> NSError {
        NSError(
            domain: "MLXAudioLab",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }
}

@MainActor
@Observable
final class ProbeViewModel {
    var isRecording = false
    var isStartingRecording = false
    var isTranscribing = false
    var isCancellingTranscription = false
    var isPreparingModel = false
    var isImportingAudio = false
    var isProcessingImport = false
    var isDeletingModel = false
    var isConfirmingModelDeletion = false
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
    var hasTranscriptOutput = false
    var transcriptStatistics = TranscriptStatistics.empty
    var transcript = ""
    var errorMessage: String?
    var transcriptExportMessage: String?
    var pathActionMessage: String?
    var shouldFollowTranscript = true
    var modelDownloadProgress: ModelDownloadProgress?
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
    private var lastTranscriptStatisticsUpdate = Date.distantPast

    private static let transcriptStatisticsUpdateInterval: TimeInterval = 0.5

    init() {
        Self.cleanupTemporaryAudioFiles()
        refreshModelAvailability(updateStatus: true)
    }

    nonisolated static var supportedImportContentTypes: [UTType] {
        [.audio, .movie]
    }

    var primaryButtonTitle: String {
        if isRecording { return "Stop" }
        if isStartingRecording { return "Starting..." }
        if isCancellingTranscription { return "Cancelling..." }
        if isTranscribing || isProcessingImport { return "Working..." }
        return "Record"
    }

    var canPressPrimaryButton: Bool {
        if isRecording {
            return !hasRecordingStopBlocker
        }

        return !hasNonRecordingWork && selectedModelCanRecord
    }

    var modelControlsDisabled: Bool {
        isRecording || hasNonRecordingWork
    }

    var canImportAudio: Bool {
        !isRecording && !hasNonRecordingWork
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

    var canDeleteSelectedModel: Bool {
        !modelControlsDisabled && (selectedModelIsLoaded || selectedModelAvailability != .notDownloaded)
    }

    var selectedModelCanRecord: Bool {
        selectedModelIsLoaded || selectedModelAvailability == .available
    }

    var canRunSelectedModel: Bool {
        !isRecording
            && !hasNonRecordingWork
            && currentSample != nil
            && selectedModelCanRecord
    }

    var canCancelTranscription: Bool {
        isTranscribing && !isCancellingTranscription
    }

    var runSelectedModelDisabledText: String {
        if currentSample == nil {
            return "Record or import media first"
        }
        if !selectedModelCanRecord {
            return "Download the selected model before running"
        }
        return ""
    }

    var currentSampleDescription: String {
        guard let currentSample else {
            return "Record or import media first"
        }

        return "\(currentSample.source.displayName) · \(currentSample.displayName) · \(Self.formatSeconds(currentSample.durationSeconds))"
    }

    var currentSampleIcon: String {
        currentSample?.source.systemImage ?? "waveform"
    }

    var modelCacheRootPath: String {
        ModelCache.rootDirectory().path
    }

    var canClearOutput: Bool {
        !isRecording && !hasNonRecordingWork
    }

    private var hasRecordingStopBlocker: Bool {
        isPreparingModel || isTranscribing || isProcessingImport || isDeletingModel
    }

    private var hasNonRecordingWork: Bool {
        isStartingRecording || hasRecordingStopBlocker
    }

    private func setTranscript(_ text: String, refreshStatistics: Bool = false) {
        transcript = text
        hasTranscriptOutput = Self.containsVisibleText(text)
        refreshTranscriptStatistics(force: refreshStatistics || !isTranscribing || !hasTranscriptOutput)
    }

    private func refreshTranscriptStatistics(force: Bool = false) {
        let now = Date()
        guard force || now.timeIntervalSince(lastTranscriptStatisticsUpdate) >= Self.transcriptStatisticsUpdateInterval else {
            return
        }

        transcriptStatistics = TranscriptStatistics.calculate(from: transcript)
        lastTranscriptStatisticsUpdate = now
    }

    private nonisolated static func containsVisibleText(_ text: String) -> Bool {
        text.contains { character in
            !character.isWhitespace
        }
    }

    func primaryButtonPressed() {
        if isRecording {
            stopRecordingAndTranscribe()
        } else {
            startRecording()
        }
    }

    func refreshModelAvailability(updateStatus: Bool = false) {
        let option = selectedModel
        modelAvailability[option.id] = ModelCache.availability(for: option)

        if updateStatus, !isRecording, !isStartingRecording, !isTranscribing, !isPreparingModel {
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

            status = "Importing media..."
            errorMessage = nil
            isProcessingImport = true

            Task {
                do {
                    let sample = try await Self.importAudioSample(from: sourceURL)
                    replaceCurrentSample(with: sample)
                    setTranscript("", refreshStatistics: true)
                    transcriptExportMessage = nil
                    metrics = TranscriptionMetrics(audioSeconds: sample.durationSeconds)
                    recordingElapsedSeconds = 0
                    errorMessage = nil
                    status = "Ready to run"
                    ProbeLog.write(
                        "audio import complete file=\(sample.url.lastPathComponent) seconds=\(sample.durationSeconds)"
                    )
                } catch {
                    status = "Import failed"
                    errorMessage = Self.describe(error)
                    ProbeLog.write("audio import failed file=\(sourceURL.lastPathComponent)", error: error)
                }

                isProcessingImport = false
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
        updateModelDownloadProgress(nil)
        status = selectedModelAvailability == .available
            ? "Loading \(option.displayName)..."
            : "Downloading \(option.displayName)..."
        isPreparingModel = true
        ProbeLog.write("model action requested repo=\(option.repoID)")

        modelPreparationTask?.cancel()
        modelPreparationTask = Task {
            do {
                let result = try await transcriber.prepareModel(option) { [weak self] progress in
                    guard let self else { return }
                    updateModelDownloadProgress(progress)
                    if let modelDownloadProgress {
                        status = "Downloading \(option.displayName) \(modelDownloadProgress.percentageText)"
                    } else if isPreparingModel {
                        status = "Loading \(option.displayName)..."
                    }
                }
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
            updateModelDownloadProgress(nil)
            isPreparingModel = false
        }
    }

    private func updateModelDownloadProgress(_ progress: ModelDownloadProgress?) {
        guard let progress else {
            modelDownloadProgress = nil
            return
        }

        if let current = modelDownloadProgress,
           current.modelName == progress.modelName,
           current.completedBytes > progress.completedBytes {
            modelDownloadProgress = progress.withCompletedBytes(current.completedBytes)
        } else {
            modelDownloadProgress = progress
        }
    }

    func requestDeleteSelectedModel() {
        guard canDeleteSelectedModel else { return }
        isConfirmingModelDeletion = true
    }

    func deleteSelectedModel() {
        guard canDeleteSelectedModel else { return }

        let option = selectedModel
        isDeletingModel = true
        errorMessage = nil
        status = "Deleting \(option.displayName)..."
        ProbeLog.write("model delete requested repo=\(option.repoID)")

        Task {
            do {
                await transcriber.unloadModel(id: option.id)
                try await Task.detached(priority: .userInitiated) {
                    try ModelCache.delete(option)
                }.value

                loadedModelIDs.remove(option.id)
                refreshModelAvailability()
                status = "\(option.displayName) deleted from local cache"
                ProbeLog.write("model delete complete repo=\(option.repoID)")
            } catch {
                status = "Model delete failed"
                errorMessage = Self.describe(error)
                ProbeLog.write("model delete failed repo=\(option.repoID)", error: error)
            }

            isDeletingModel = false
        }
    }

    func runSelectedModelForCurrentSample() {
        guard let sample = currentSample else {
            status = "Record or import media first"
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

    func cancelTranscription() {
        guard isTranscribing else { return }

        isCancellingTranscription = true
        status = "Cancelling transcription..."
        errorMessage = nil
        transcriptionTask?.cancel()
        ProbeLog.write("transcription cancel requested")
    }

    func startRecording() {
        guard !isStartingRecording else { return }

        guard selectedModelCanRecord else {
            status = "Download the selected model before recording"
            errorMessage = "\(selectedModel.displayName) is not available on this Mac yet."
            return
        }

        let option = selectedModel
        ProbeLog.write("record requested")
        errorMessage = nil
        setTranscript("", refreshStatistics: true)
        transcriptExportMessage = nil
        metrics = TranscriptionMetrics()
        recordingElapsedSeconds = 0
        status = "Requesting microphone access..."
        isStartingRecording = true

        Task {
            defer {
                self.isStartingRecording = false
            }

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

        let wallClockElapsed = recordingStartedAt.map { Date().timeIntervalSince($0) } ?? recordingElapsedSeconds
        recordingElapsedSeconds = wallClockElapsed
        isRecording = false

        guard let recordingURL else {
            status = "No recording found"
            ProbeLog.write("stop failed no recording url")
            return
        }

        let durationSeconds = (try? Self.audioDurationSeconds(for: recordingURL)) ?? wallClockElapsed
        recordingElapsedSeconds = durationSeconds

        let option = selectedModel
        self.recordingURL = nil
        let sample = AudioSample(
            id: UUID(),
            url: recordingURL,
            source: .recorded,
            displayName: recordingURL.lastPathComponent,
            durationSeconds: durationSeconds,
            createdAt: Date()
        )
        replaceCurrentSample(with: sample)
        startTranscription(sample: sample, option: option)
    }

    private func startTranscription(sample: AudioSample, option: AudioModelOption) {
        status = "Transcribing with \(option.displayName)..."
        isTranscribing = true
        isCancellingTranscription = false
        errorMessage = nil
        setTranscript("", refreshStatistics: true)
        transcriptExportMessage = nil
        metrics = TranscriptionMetrics(audioSeconds: sample.durationSeconds)

        transcriptionTask?.cancel()
        transcriptionTask = Task {
            do {
                let result = try await transcriber.transcribe(
                    audioURL: sample.url,
                    audioSeconds: sample.durationSeconds,
                    using: option,
                    onPartialTranscript: { partialTranscript in
                        await MainActor.run {
                            self.setTranscript(partialTranscript)
                        }
                    },
                    onMetricsUpdate: { updatedMetrics in
                        await MainActor.run {
                            self.metrics = updatedMetrics
                        }
                    }
                )
                self.loadedModelIDs.insert(option.id)
                self.refreshModelAvailability()
                self.setTranscript(result.text, refreshStatistics: true)
                self.metrics = result.metrics
                self.status = result.text.isEmpty ? "Finished with empty output" : "Finished"
                ProbeLog.write("ui updated with transcription")
            } catch is CancellationError {
                self.status = "Cancelled"
                self.errorMessage = nil
                ProbeLog.write("transcription cancelled")
            } catch {
                self.status = "Transcription failed"
                self.errorMessage = Self.describe(error)
                ProbeLog.write("transcription failed", error: error)
            }
            self.isCancellingTranscription = false
            self.isTranscribing = false
            self.transcriptionTask = nil
        }
    }

    func clearOutput() {
        deleteCurrentSample()
        setTranscript("", refreshStatistics: true)
        errorMessage = nil
        transcriptExportMessage = nil
        status = idleStatusForSelectedModel()
        metrics = TranscriptionMetrics()
        recordingElapsedSeconds = 0
    }

    func copyTranscript() {
        guard hasTranscriptOutput else { return }

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(transcript, forType: .string)
        transcriptExportMessage = "Copied transcript"
    }

    func saveTranscriptAsText() {
        saveTranscript(
            content: transcript,
            fileExtension: "txt",
            contentType: .plainText
        )
    }

    func saveTranscriptAsMarkdown() {
        saveTranscript(
            content: markdownTranscript(),
            fileExtension: "md",
            contentType: Self.markdownContentType
        )
    }

    func copyPath(_ path: String, title: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(path, forType: .string)
        pathActionMessage = "\(title) path copied"
    }

    func openPathInFinder(_ path: String, title: String) {
        let url = URL(fileURLWithPath: path, isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            guard NSWorkspace.shared.open(url) else {
                pathActionMessage = "Could not open \(title)"
                return
            }
            pathActionMessage = "Opened \(title) in Finder"
        } catch {
            pathActionMessage = "Could not open \(title): \(Self.describe(error))"
        }
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

    private func saveTranscript(content: String, fileExtension: String, contentType: UTType) {
        guard hasTranscriptOutput else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [contentType]
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        panel.nameFieldStringValue = suggestedTranscriptFileName(fileExtension: fileExtension)

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
            transcriptExportMessage = "Saved \(url.lastPathComponent)"
        } catch {
            transcriptExportMessage = "Save failed: \(Self.describe(error))"
        }
    }

    private func markdownTranscript() -> String {
        let sampleName = currentSample?.displayName ?? "No sample"
        let audioLength = Self.formatSeconds(metrics.audioSeconds)

        return """
        # Transcript

        - Model: \(selectedModel.displayName)
        - Sample: \(sampleName)
        - Audio length: \(audioLength)

        ## Text

        \(transcript)
        """
    }

    private func suggestedTranscriptFileName(fileExtension: String) -> String {
        let displayName = currentSample?.displayName ?? "transcript"
        let stem = URL(fileURLWithPath: displayName).deletingPathExtension().lastPathComponent
        let rawBaseName = stem.isEmpty ? "transcript" : stem
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let sanitized = rawBaseName.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let baseName = String(sanitized).trimmingCharacters(in: CharacterSet(charactersIn: "-"))

        return "\(baseName.isEmpty ? "transcript" : baseName)-transcript.\(fileExtension)"
    }

    private nonisolated static var markdownContentType: UTType {
        UTType(filenameExtension: "md") ?? .plainText
    }

    private nonisolated static func importAudioSample(from sourceURL: URL) async throws -> AudioSample {
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
            try await convertMediaToWAV(from: sourceURL, to: destinationURL)
            let durationSeconds = try Self.audioDurationSeconds(for: destinationURL)
            guard durationSeconds > 0 else {
                throw Self.makeError("The selected media file has no readable audio.")
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

    private nonisolated static func convertMediaToWAV(from sourceURL: URL, to destinationURL: URL) async throws {
        let asset = AVURLAsset(url: sourceURL)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard !audioTracks.isEmpty else {
            throw makeError("The selected file does not contain a readable audio track.")
        }

        try await writeAudioTracksToWAV(asset: asset, audioTracks: audioTracks, destinationURL: destinationURL)
    }

    private nonisolated static func writeAudioTracksToWAV(
        asset: AVAsset,
        audioTracks: [AVAssetTrack],
        destinationURL: URL
    ) async throws {
        let audioSettings = wavConversionSettings()
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderAudioMixOutput(audioTracks: audioTracks, audioSettings: audioSettings)
        output.alwaysCopiesSampleData = false

        guard reader.canAdd(output) else {
            throw makeError("This media file's audio track cannot be prepared for conversion.")
        }
        reader.add(output)

        let writer = try AVAssetWriter(outputURL: destinationURL, fileType: .wav)
        let input = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        input.expectsMediaDataInRealTime = false

        guard writer.canAdd(input) else {
            throw makeError("The WAV writer could not be configured for this media file.")
        }
        writer.add(input)

        guard writer.startWriting() else {
            throw writer.error ?? makeError("The WAV writer failed to start.")
        }

        guard reader.startReading() else {
            writer.cancelWriting()
            throw reader.error ?? makeError("The selected media file could not be read.")
        }

        writer.startSession(atSourceTime: .zero)

        let conversionSession = WAVConversionSession(
            reader: reader,
            output: output,
            writer: writer,
            input: input
        )
        try await conversionSession.run()
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

    private nonisolated static let recordingSessionDirectoryName = "session-\(UUID().uuidString)"

    private static func makeRecordingURL() throws -> URL {
        let directory = recordingDirectory()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appending(path: "recording-\(UUID().uuidString).wav")
    }

    private nonisolated static func cleanupTemporaryAudioFiles() {
        let rootDirectory = recordingRootDirectory()
        let currentDirectory = recordingDirectory()
        let staleCutoff = Date().addingTimeInterval(-24 * 60 * 60)
        guard let sessionDirectories = try? FileManager.default.contentsOfDirectory(
            at: rootDirectory,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey]
        ) else {
            return
        }

        for directory in sessionDirectories {
            if isStaleRecordingDirectory(directory, currentDirectory: currentDirectory, staleCutoff: staleCutoff) {
                try? FileManager.default.removeItem(at: directory)
            }
        }
    }

    private nonisolated static func isStaleRecordingDirectory(
        _ directory: URL,
        currentDirectory: URL,
        staleCutoff: Date
    ) -> Bool {
        guard directory != currentDirectory,
              let values = try? directory.resourceValues(forKeys: [.contentModificationDateKey, .isDirectoryKey]),
              values.isDirectory == true
        else {
            return false
        }

        return (values.contentModificationDate ?? .distantPast) < staleCutoff
    }

    private nonisolated static func recordingDirectory() -> URL {
        recordingRootDirectory()
            .appending(path: recordingSessionDirectoryName, directoryHint: .isDirectory)
    }

    private nonisolated static func recordingRootDirectory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "MLXAudioLabRecordings", directoryHint: .isDirectory)
    }

    private nonisolated static func audioDurationSeconds(for url: URL) throws -> Double {
        let audioFile = try AVAudioFile(forReading: url)
        let sampleRate = audioFile.fileFormat.sampleRate
        guard sampleRate > 0 else {
            throw makeError("The selected media file has an invalid sample rate.")
        }

        let durationSeconds = Double(audioFile.length) / sampleRate
        guard durationSeconds.isFinite, durationSeconds > 0 else {
            throw makeError("The selected media file has no readable audio.")
        }

        return durationSeconds
    }

    private nonisolated static func wavConversionSettings() -> [String: Any] {
        [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
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

    private enum Layout {
        static let edgeInset: CGFloat = 24
        static let topChromeCompensation: CGFloat = 16
        static let columnSpacing: CGFloat = 20
        static let sidebarWidth: CGFloat = 320
        static let performanceWidth: CGFloat = 260
    }

    var body: some View {
        ZStack {
            LabBackdrop()

            if #available(macOS 26.0, *) {
                GlassEffectContainer(spacing: Layout.columnSpacing) {
                    workspace
                }
            } else {
                workspace
            }

            if let progress = model.modelDownloadProgress {
                VStack {
                    DownloadProgressBanner(progress: progress)
                    Spacer()
                }
                .padding(.horizontal, Layout.edgeInset)
                .padding(.top, 8)
                .transition(.move(edge: .top).combined(with: .opacity))
                .zIndex(2)
            }
        }
        .animation(.easeInOut(duration: 0.18), value: model.modelDownloadProgress != nil)
        .frame(minWidth: 1120, minHeight: 820)
        .onAppear {
            model.refreshModelAvailability(updateStatus: true)
        }
        .fileImporter(
            isPresented: $model.isImportingAudio,
            allowedContentTypes: ProbeViewModel.supportedImportContentTypes,
            allowsMultipleSelection: false
        ) { result in
            model.handleImportedAudio(result)
        }
        .fileDialogMessage("Select an audio or video file. The app will extract audio into a temporary WAV sample.")
        .fileDialogConfirmationLabel("Import Media")
        .confirmationDialog(
            "Delete downloaded model?",
            isPresented: $model.isConfirmingModelDeletion,
            titleVisibility: .visible
        ) {
            Button("Delete \(model.selectedModel.displayName)", role: .destructive) {
                model.deleteSelectedModel()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This removes the selected model from the app cache and the matching Hugging Face cache. It does not delete recordings or imported samples.")
        }
    }

    private var workspace: some View {
        HStack(alignment: .top, spacing: Layout.columnSpacing) {
            LabSidebar(model: model)
                .frame(width: Layout.sidebarWidth)

            TranscriptWorkspace(model: model)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            PerformancePanel(model: model)
                .frame(width: Layout.performanceWidth)
        }
        .padding(.horizontal, Layout.edgeInset)
        .padding(.bottom, Layout.edgeInset)
        .padding(.top, Layout.edgeInset + Layout.topChromeCompensation)
    }
}

struct DownloadProgressBanner: View {
    let progress: ModelDownloadProgress

    private enum Layout {
        static let progressTextWidth: CGFloat = 220
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 10) {
                Label("Downloading \(progress.modelName)", systemImage: "arrow.down.circle.fill")
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)

                Spacer(minLength: 12)

                Text("\(progress.percentageText)  \(progress.byteProgressText)")
                    .font(.caption.monospacedDigit().weight(.medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .frame(width: Layout.progressTextWidth, alignment: .trailing)
            }

            ProgressView(value: progress.fractionCompleted)
                .progressViewStyle(.linear)
                .tint(.orange)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(maxWidth: 680)
        .labGlassPanel(cornerRadius: 12, tint: .orange.opacity(0.16), interactive: false)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Downloading \(progress.modelName)")
        .accessibilityValue("\(progress.percentageText), \(progress.byteProgressText)")
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
                Label("Media", systemImage: "waveform")
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

            HStack(spacing: 10) {
                Picker("Model", selection: $model.selectedModelID) {
                    ForEach(AudioModelOption.supported) { option in
                        Text(option.displayName).tag(option.id)
                    }
                }
                .labelsHidden()
                .disabled(model.modelControlsDisabled)

                Button {
                    model.requestDeleteSelectedModel()
                } label: {
                    Image(systemName: "trash")
                        .frame(width: 28)
                }
                .labButtonStyle()
                .disabled(!model.canDeleteSelectedModel)
                .help("Delete selected model from the local Hugging Face cache")
            }

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
                    Text(model.selectedModel.family.rawValue)
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
                if model.isTranscribing {
                    Button {
                        model.cancelTranscription()
                    } label: {
                        Label(model.isCancellingTranscription ? "Cancelling..." : "Cancel", systemImage: "xmark.circle.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .labButtonStyle(prominent: true)
                    .tint(.orange)
                    .disabled(!model.canCancelTranscription)
                    .accessibilityLabel("Cancel transcription")
                    .help("Cancel transcription after the current decode chunk finishes")
                } else {
                    Button {
                        model.runSelectedModelForCurrentSample()
                    } label: {
                        Label("Run", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .labButtonStyle(prominent: true)
                    .disabled(!model.canRunSelectedModel)
                }

                Button {
                    model.clearOutput()
                } label: {
                    Image(systemName: "trash")
                        .frame(width: 28)
                }
                .labButtonStyle()
                .disabled(!model.canClearOutput)
                .help("Clear sample and output")
            }

            Button {
                model.beginImportAudio()
            } label: {
                Label(model.isProcessingImport ? "Importing..." : "Import Media", systemImage: "square.and.arrow.down")
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
    private let transcriptEndID = "transcript-end"

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
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Transcript")
                    .font(.system(size: 30, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)

                Text(model.selectedModel.displayName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(alignment: .center, spacing: 10) {
                Toggle(isOn: $model.shouldFollowTranscript) {
                    Label("Follow", systemImage: "arrow.down.to.line")
                }
                .toggleStyle(.checkbox)
                .controlSize(.small)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: true, vertical: false)
                .help("Follow transcript output while it is generated")

                StatusCapsule(
                    text: statusTitle,
                    symbol: statusSymbol,
                    tint: statusTint,
                    isProgressing: statusShowsProgress
                )
                .help(model.status)

                Spacer(minLength: 8)

                if model.isRecording {
                    Text(formatSeconds(model.recordingElapsedSeconds))
                        .font(.system(.title3, design: .monospaced, weight: .semibold))
                        .foregroundStyle(.red)
                        .monospacedDigit()
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }
            }
        }
    }

    private var transcriptEditor: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 0) {
                    Group {
                        if model.hasTranscriptOutput {
                            Text(model.transcript)
                                .foregroundStyle(.primary)
                                .textSelection(.enabled)
                        } else {
                            Text("No output yet.")
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .font(.system(.body, design: .default))
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(18)

                    Color.clear
                        .frame(height: 1)
                        .id(transcriptEndID)
                }
            }
            .scrollIndicators(.visible)
            .onChange(of: model.transcript) {
                scrollToBottom(proxy)
            }
            .onChange(of: model.shouldFollowTranscript) {
                scrollToBottom(proxy, animated: false)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.black.opacity(0.12), in: .rect(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(.white.opacity(0.10), lineWidth: 1)
        }
    }

    private var statusTitle: String {
        if model.isRecording { return "Recording" }
        if model.isStartingRecording { return "Starting" }
        if model.isProcessingImport { return "Importing" }
        if model.isPreparingModel { return "Preparing model" }
        if model.isDeletingModel { return "Deleting model" }
        if model.isCancellingTranscription { return "Cancelling" }
        if model.isTranscribing { return "Transcribing" }
        if model.errorMessage != nil { return "Needs attention" }
        if model.status.localizedCaseInsensitiveContains("finished") { return "Finished" }
        if model.status.localizedCaseInsensitiveContains("cancelled") { return "Cancelled" }
        return "Ready"
    }

    private var statusSymbol: String {
        if model.isRecording { return "record.circle.fill" }
        if model.isStartingRecording { return "record.circle" }
        if model.status.localizedCaseInsensitiveContains("cancelled") { return "xmark.circle.fill" }
        if model.errorMessage != nil { return "exclamationmark.triangle.fill" }
        return "checkmark.circle.fill"
    }

    private var statusTint: Color {
        if model.isRecording { return .red }
        if model.isStartingRecording { return .orange }
        if statusShowsProgress { return .orange }
        if model.status.localizedCaseInsensitiveContains("cancelled") { return .orange }
        if model.errorMessage != nil { return .red }
        return .green
    }

    private var statusShowsProgress: Bool {
        model.isTranscribing
            || model.isStartingRecording
            || model.isCancellingTranscription
            || model.isPreparingModel
            || model.isProcessingImport
            || model.isDeletingModel
    }

    private func formatSeconds(_ seconds: Double) -> String {
        ProbeViewModel.formatSeconds(seconds)
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool = true) {
        guard model.shouldFollowTranscript else { return }

        if animated {
            withAnimation(.easeOut(duration: 0.18)) {
                proxy.scrollTo(transcriptEndID, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(transcriptEndID, anchor: .bottom)
        }
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

            Divider()
                .opacity(0.45)
                .padding(.vertical, 2)

            TranscriptDataSection(model: model)

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

struct TranscriptDataSection: View {
    let model: ProbeViewModel

    var body: some View {
        let stats = model.transcriptStatistics
        let hasOutput = model.hasTranscriptOutput

        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Transcript", symbol: "text.quote")

            VStack(spacing: 0) {
                MetricRow(title: "Words", value: formatCount(stats.words))
                MetricRow(title: "Letters", value: formatCount(stats.letters))
                MetricRow(title: "Characters", value: formatCount(stats.characters))
                MetricRow(title: "Lines", value: formatCount(stats.lines), showDivider: false)
            }

            VStack(spacing: 8) {
                Button {
                    model.copyTranscript()
                } label: {
                    Label("Copy Text", systemImage: "doc.on.doc")
                        .frame(maxWidth: .infinity)
                }
                .labButtonStyle()
                .disabled(!hasOutput)
                .help("Copy transcript text to the clipboard")

                HStack(spacing: 8) {
                    Button {
                        model.saveTranscriptAsText()
                    } label: {
                        Label("Save TXT", systemImage: "doc.text")
                            .frame(maxWidth: .infinity)
                    }
                    .labButtonStyle()
                    .disabled(!hasOutput)
                    .help("Save transcript as a text file")

                    Button {
                        model.saveTranscriptAsMarkdown()
                    } label: {
                        Label("Save MD", systemImage: "doc.richtext")
                            .frame(maxWidth: .infinity)
                    }
                    .labButtonStyle()
                    .disabled(!hasOutput)
                    .help("Save transcript as a Markdown file")
                }
            }

            if let message = model.transcriptExportMessage {
                let isFailure = message.localizedCaseInsensitiveContains("failed")
                Label(message, systemImage: isFailure ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(isFailure ? .red : .secondary)
                    .lineLimit(2)
            }
        }
    }

    private func formatCount(_ count: Int) -> String {
        count.formatted(.number)
    }
}

struct SystemPathsPanel: View {
    let model: ProbeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            SectionHeader(title: "Local paths", symbol: "folder")
            PathActionRow(title: "Logs", value: model.logDirectoryPath, model: model)
            PathActionRow(title: "Models", value: model.modelCacheRootPath, model: model)

            if let message = model.pathActionMessage {
                let isFailure = message.localizedCaseInsensitiveContains("could not")
                Label(message, systemImage: isFailure ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(isFailure ? .red : .secondary)
                    .lineLimit(2)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
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
    var isProgressing = false

    var body: some View {
        HStack(spacing: 8) {
            if isProgressing {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.72)
                    .tint(tint)
                    .frame(width: 13, height: 13)
            } else {
                Image(systemName: symbol)
                    .font(.caption.weight(.semibold))
            }

            Text(text)
                .lineLimit(1)
        }
        .font(.caption.weight(.semibold))
        .padding(.horizontal, 11)
        .padding(.vertical, 7)
        .foregroundStyle(tint)
        .fixedSize(horizontal: true, vertical: false)
        .labGlassPanel(cornerRadius: 13, tint: tint.opacity(0.10), interactive: false)
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

struct PathActionRow: View {
    let title: String
    let value: String
    let model: ProbeViewModel

    private enum Layout {
        static let actionButtonWidth: CGFloat = 64
        static let actionButtonHeight: CGFloat = 34
        static let actionIconWidth: CGFloat = 24
    }

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Text(displayPath)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }

            Spacer(minLength: 8)

            actionButton(
                symbol: "doc.on.doc",
                help: "Copy \(title.lowercased()) path",
                accessibilityLabel: "Copy \(title) path"
            ) {
                model.copyPath(value, title: title)
            }

            actionButton(
                symbol: "folder",
                help: "Open \(title.lowercased()) folder in Finder",
                accessibilityLabel: "Open \(title) folder in Finder"
            ) {
                model.openPathInFinder(value, title: title)
            }
        }
        .padding(.vertical, 3)
    }

    private func actionButton(
        symbol: String,
        help: String,
        accessibilityLabel: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.body.weight(.medium))
                .frame(width: Layout.actionIconWidth)
                .frame(width: Layout.actionButtonWidth, height: Layout.actionButtonHeight)
                .contentShape(Rectangle())
        }
        .labButtonStyle()
        .help(help)
        .accessibilityLabel(accessibilityLabel)
    }

    private var displayPath: String {
        NSString(string: value).abbreviatingWithTildeInPath
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

private final class ApplicationDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct MLXAudioLabApp: App {
    @NSApplicationDelegateAdaptor(ApplicationDelegate.self) private var applicationDelegate
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
