//
//  AudioFileService.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import AVFoundation
import AppKit

// MARK: - 音频文件服务

/// 负责文件选择、音频读取/写入、格式转换
enum AudioFileService {

    // MARK: - 常量

    /// 标准化输出采样率（与 FFmpeg 降噪输出保持一致）
    static let targetSampleRate: Double = 16000.0

    /// 波形可视化的采样点数量
    static let waveformSampleCount: Int = 200

    // MARK: - 文件选择

    /// 打开文件选择面板，让用户选择 MP3/音频文件
    @MainActor
    static func openFilePicker() async -> [URL] {
        let panel = NSOpenPanel()
        panel.title = "选择音频文件"
        panel.message = "选择一个或多个 MP3 音频文件"
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.mp3, .audio]

        let response = panel.runModal()
        guard response == .OK else { return [] }
        return panel.urls
    }

    /// 打开保存面板，让用户选择导出位置
    @MainActor
    static func openSavePanel(suggestedName: String) async -> URL? {
        let panel = NSSavePanel()
        panel.title = "导出降噪后的音频"
        panel.nameFieldStringValue = suggestedName
        panel.allowedContentTypes = [.wav]
        panel.canCreateDirectories = true

        let response = panel.runModal()
        guard response == .OK else { return nil }
        return panel.url
    }

    // MARK: - 音频读取

    /// 读取音频文件信息（时长），不加载完整数据
    static func getAudioDuration(url: URL) throws -> TimeInterval {
        let audioFile = try AVAudioFile(forReading: url)
        let sampleRate = audioFile.processingFormat.sampleRate
        let frameCount = Double(audioFile.length)
        return frameCount / sampleRate
    }

    /// 读取音频文件并转换为 16kHz 单声道 Float32 PCM 数据
    static func loadAndResample(url: URL) throws -> [Float] {
        let sourceFile = try AVAudioFile(forReading: url)
        let sourceFormat = sourceFile.processingFormat

        // 目标格式：16kHz 单声道 Float32
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioFileServiceError.formatCreationFailed
        }

        // 读取源文件全部数据
        let sourceFrameCount = AVAudioFrameCount(sourceFile.length)
        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: sourceFrameCount
        ) else {
            throw AudioFileServiceError.bufferCreationFailed
        }
        try sourceFile.read(into: sourceBuffer)

        // 如果已经是目标格式，直接返回
        if sourceFormat.sampleRate == targetSampleRate
            && sourceFormat.channelCount == 1
        {
            return bufferToFloatArray(sourceBuffer)
        }

        // 创建转换器
        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw AudioFileServiceError.converterCreationFailed
        }

        // 计算目标缓冲区大小
        let ratio = targetSampleRate / sourceFormat.sampleRate
        let targetFrameCount = AVAudioFrameCount(Double(sourceFrameCount) * ratio)
        guard let targetBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: targetFrameCount
        ) else {
            throw AudioFileServiceError.bufferCreationFailed
        }

        // 执行转换
        var isDone = false
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if isDone {
                outStatus.pointee = .noDataNow
                return nil
            }
            isDone = true
            outStatus.pointee = .haveData
            return sourceBuffer
        }

        var error: NSError?
        converter.convert(to: targetBuffer, error: &error, withInputFrom: inputBlock)
        if let error {
            throw AudioFileServiceError.conversionFailed(error.localizedDescription)
        }

        return bufferToFloatArray(targetBuffer)
    }

    // MARK: - 便捷方法

    /// 从音频文件 URL 直接加载并提取波形采样点（用于可视化）
    /// - Parameter url: 音频文件 URL
    /// - Returns: 降采样后的波形 RMS 采样点数组
    static func loadWaveformFromFile(url: URL) throws -> [Float] {
        let audioData = try loadAndResample(url: url)
        return extractWaveformSamples(from: audioData)
    }

    /// 生成降噪输出文件的临时 URL
    /// - Parameter originalFileName: 原始文件名
    /// - Returns: 临时目录下的 WAV 文件 URL
    static func generateTempOutputURL(originalFileName: String) -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let baseName = (originalFileName as NSString).deletingPathExtension
        let outputName = "\(baseName)_denoised.wav"
        let outputURL = tempDir.appendingPathComponent(outputName)

        // 如果文件已存在则删除
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try? FileManager.default.removeItem(at: outputURL)
        }

        return outputURL
    }

    // MARK: - 波形数据提取

    /// 从音频数据中提取用于可视化的波形采样点
    static func extractWaveformSamples(from audioData: [Float], count: Int = waveformSampleCount) -> [Float] {
        guard !audioData.isEmpty else { return [] }

        let chunkSize = max(1, audioData.count / count)
        var samples: [Float] = []
        samples.reserveCapacity(count)

        for i in 0..<count {
            let start = i * chunkSize
            let end = min(start + chunkSize, audioData.count)
            guard start < audioData.count else { break }

            // 取每个 chunk 的 RMS 值
            let chunk = Array(audioData[start..<end])
            let rms = sqrt(chunk.reduce(0) { $0 + $1 * $1 } / Float(chunk.count))
            samples.append(rms)
        }

        return samples
    }

    // MARK: - 音频写入

    /// 将 Float32 PCM 数据写入 WAV 文件
    static func writeWAV(data: [Float], sampleRate: Double = targetSampleRate, to url: URL) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioFileServiceError.formatCreationFailed
        }

        let frameCount = AVAudioFrameCount(data.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioFileServiceError.bufferCreationFailed
        }
        buffer.frameLength = frameCount

        // 拷贝数据到缓冲区
        if let channelData = buffer.floatChannelData?[0] {
            data.withUnsafeBufferPointer { src in
                channelData.update(from: src.baseAddress!, count: data.count)
            }
        }

        let outputFile = try AVAudioFile(forWriting: url, settings: format.settings)
        try outputFile.write(from: buffer)
    }

    /// 将处理后的音频保存到临时目录，返回临时文件 URL
    static func saveToTempFile(data: [Float], originalFileName: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let baseName = (originalFileName as NSString).deletingPathExtension
        let outputName = "\(baseName)_denoised.wav"
        let outputURL = tempDir.appendingPathComponent(outputName)

        // 如果文件已存在则删除
        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        try writeWAV(data: data, to: outputURL)
        return outputURL
    }

    /// 将临时文件复制到用户选择的位置
    static func exportFile(from sourceURL: URL, to destinationURL: URL) throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
    }

    // MARK: - 私有辅助方法

    /// 将 AVAudioPCMBuffer 转换为 Float 数组
    private static func bufferToFloatArray(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else { return [] }
        let frameLength = Int(buffer.frameLength)

        if buffer.format.channelCount == 1 {
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameLength))
        }

        // 多声道 -> 混合为单声道
        var mono = [Float](repeating: 0, count: frameLength)
        let channelCount = Int(buffer.format.channelCount)
        for ch in 0..<channelCount {
            let chData = channelData[ch]
            for i in 0..<frameLength {
                mono[i] += chData[i]
            }
        }
        let scale = 1.0 / Float(channelCount)
        for i in 0..<frameLength {
            mono[i] *= scale
        }
        return mono
    }
}

// MARK: - 错误类型

enum AudioFileServiceError: LocalizedError {
    case formatCreationFailed
    case bufferCreationFailed
    case converterCreationFailed
    case conversionFailed(String)
    case fileNotFound(String)

    var errorDescription: String? {
        switch self {
        case .formatCreationFailed:
            return "无法创建音频格式"
        case .bufferCreationFailed:
            return "无法创建音频缓冲区"
        case .converterCreationFailed:
            return "无法创建音频转换器"
        case .conversionFailed(let msg):
            return "音频转换失败: \(msg)"
        case .fileNotFound(let path):
            return "找不到文件: \(path)"
        }
    }
}
