//
//  FFmpegDenoiser.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/14.
//

import Foundation

// MARK: - FFmpeg 降噪引擎

/// 基于 FFmpeg arnndn 滤镜 (RNNoise) 的音频降噪引擎
///
/// arnndn (Audio Recurrent Neural Network Denoiser) 使用 RNNoise 神经网络
/// 专门针对人声进行降噪，能有效去除背景噪声同时保留语音质量。
///
/// 处理流程: 输入文件 → FFmpeg Process → arnndn 滤镜 → 输出 WAV 文件
final class FFmpegDenoiser: Sendable {

    // MARK: - 常量

    /// 输出音频采样率 (Hz)
    static let outputSampleRate: Double = 16000.0

    // MARK: - 属性

    /// 降噪强度 (0.0 ~ 1.0)，映射到 arnndn 的 mix 参数
    private let denoiseStrength: Float

    /// FFmpeg 二进制文件路径
    private let ffmpegURL: URL

    /// RNNoise 模型文件路径
    private let modelURL: URL

    // MARK: - 初始化

    /// 初始化降噪引擎
    /// - Parameter strength: 降噪强度 (0.0 ~ 1.0)，默认 1.0（全强度）
    init(strength: Float = 1.0) throws {
        self.denoiseStrength = max(0.0, min(1.0, strength))

        // 定位 Bundle 内的 FFmpeg 二进制
        guard let ffmpegPath = Bundle.main.path(forResource: "ffmpeg", ofType: nil) else {
            throw FFmpegDenoiserError.ffmpegNotFound
        }
        self.ffmpegURL = URL(fileURLWithPath: ffmpegPath)

        // 定位 Bundle 内的 RNNoise 模型文件
        guard let modelPath = Bundle.main.path(forResource: "std", ofType: "rnnn") else {
            throw FFmpegDenoiserError.modelNotFound
        }
        self.modelURL = URL(fileURLWithPath: modelPath)

        // 确保 ffmpeg 有执行权限
        let fileManager = FileManager.default
        if !fileManager.isExecutableFile(atPath: ffmpegPath) {
            try? fileManager.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: ffmpegPath
            )
        }
    }

    // MARK: - 主处理方法

    /// 对音频文件执行降噪处理
    /// - Parameters:
    ///   - inputURL: 输入音频文件 URL
    ///   - outputURL: 输出 WAV 文件 URL
    ///   - duration: 音频总时长（秒），用于计算进度
    ///   - onProgress: 进度回调 (0.0 ~ 1.0)
    func process(
        inputURL: URL,
        outputURL: URL,
        duration: TimeInterval,
        onProgress: @escaping @Sendable (Double) -> Void
    ) throws {
        // 构建 FFmpeg 参数
        let arguments = buildArguments(
            inputPath: inputURL.path,
            outputPath: outputURL.path
        )

        // 创建 Process
        let process = Process()
        process.executableURL = ffmpegURL
        process.arguments = arguments

        // 设置环境变量，避免 FFmpeg 尝试读取终端
        var environment = ProcessInfo.processInfo.environment
        environment["AV_LOG_FORCE_NOCOLOR"] = "1"
        process.environment = environment

        // 捕获 stdout（-progress pipe:1 输出到 stdout）
        let stdoutPipe = Pipe()
        process.standardOutput = stdoutPipe

        // 捕获 stderr（FFmpeg 日志输出）
        let stderrPipe = Pipe()
        process.standardError = stderrPipe

        // 总时长（微秒），用于计算进度
        let totalDurationUs = duration * 1_000_000

        // 异步读取 stdout 解析进度
        let progressHandler = stdoutPipe.fileHandleForReading
        progressHandler.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let output = String(data: data, encoding: .utf8)
            else { return }

            // 解析 -progress 输出的 key=value 行
            let lines = output.components(separatedBy: .newlines)
            for line in lines {
                if line.hasPrefix("out_time_us=") {
                    let valueStr = line.replacingOccurrences(of: "out_time_us=", with: "")
                    if let timeUs = Double(valueStr), totalDurationUs > 0 {
                        let progress = min(1.0, max(0.0, timeUs / totalDurationUs))
                        onProgress(progress)
                    }
                }
            }
        }

        // 启动 FFmpeg 进程
        do {
            try process.run()
        } catch {
            progressHandler.readabilityHandler = nil
            throw FFmpegDenoiserError.processLaunchFailed(error.localizedDescription)
        }

        // 等待进程完成
        process.waitUntilExit()

        // 清理读取回调
        progressHandler.readabilityHandler = nil

        // 检查退出状态
        let exitCode = process.terminationStatus
        if exitCode != 0 {
            // 读取 stderr 获取错误信息
            let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
            let stderrOutput = String(data: stderrData, encoding: .utf8) ?? "未知错误"

            // 提取最后几行有用的错误信息
            let errorLines = stderrOutput
                .components(separatedBy: .newlines)
                .filter { !$0.isEmpty }
                .suffix(5)
                .joined(separator: "\n")

            throw FFmpegDenoiserError.processFailed(exitCode: exitCode, message: errorLines)
        }

        // 验证输出文件存在
        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw FFmpegDenoiserError.outputFileMissing
        }

        // #region agent log
        do {
            let logPath2 = "/Users/Jason/Desktop/VoiceClean/.cursor/debug.log"
            let fileSize = (try? FileManager.default.attributesOfItem(atPath: outputURL.path)[.size] as? Int) ?? -1
            let successLog = "{\"location\":\"FFmpegDenoiser.swift:process-success\",\"message\":\"FFmpeg completed\",\"data\":{\"exitCode\":0,\"outputExists\":true,\"outputSize\":\(fileSize)},\"hypothesisId\":\"E\",\"timestamp\":\(Int(Date().timeIntervalSince1970 * 1000))}\n"
            if let sData = successLog.data(using: .utf8), let fh = FileHandle(forWritingAtPath: logPath2) { fh.seekToEndOfFile(); fh.write(sData); fh.closeFile() }
        }
        // #endregion

        // 最终进度 100%
        onProgress(1.0)
    }

    // MARK: - 私有方法

    /// 构建 FFmpeg 命令行参数
    private func buildArguments(inputPath: String, outputPath: String) -> [String] {
        // 构建 arnndn 滤镜字符串
        // mix 参数: 1.0 = 完全降噪, 0.0 = 原始信号
        let mixValue = String(format: "%.2f", denoiseStrength)
        let filterChain = "arnndn=m=\(modelURL.path):mix=\(mixValue)"

        // #region agent log
        do {
            let logPath = "/Users/Jason/Desktop/VoiceClean/.cursor/debug.log"
            let logEntry = "{\"location\":\"FFmpegDenoiser.swift:buildArguments\",\"message\":\"FFmpeg args built\",\"data\":{\"inputPath\":\"\(inputPath)\",\"outputPath\":\"\(outputPath)\",\"filterChain\":\"\(filterChain)\",\"mixValue\":\"\(mixValue)\"},\"hypothesisId\":\"E\",\"timestamp\":\(Int(Date().timeIntervalSince1970 * 1000))}\n"
            if !FileManager.default.fileExists(atPath: logPath) { FileManager.default.createFile(atPath: logPath, contents: nil) }
            if let data = logEntry.data(using: .utf8), let fh = FileHandle(forWritingAtPath: logPath) { fh.seekToEndOfFile(); fh.write(data); fh.closeFile() }
        }
        // #endregion

        return [
            "-y",                       // 覆盖输出文件
            "-i", inputPath,            // 输入文件
            "-af", filterChain,         // 音频滤镜链
            "-ar", "16000",             // 输出采样率 16kHz
            "-ac", "1",                 // 单声道
            "-c:a", "pcm_f32le",        // Float32 PCM 编码器（WAV 格式）
            "-f", "wav",                // 输出格式 WAV
            "-progress", "pipe:1",      // 进度输出到 stdout
            "-loglevel", "error",       // 只输出错误日志到 stderr
            outputPath                  // 输出文件路径
        ]
    }
}

// MARK: - 错误类型

enum FFmpegDenoiserError: LocalizedError {
    case ffmpegNotFound
    case modelNotFound
    case processLaunchFailed(String)
    case processFailed(exitCode: Int32, message: String)
    case outputFileMissing

    var errorDescription: String? {
        switch self {
        case .ffmpegNotFound:
            return "找不到 FFmpeg 可执行文件 — 请确保 ffmpeg 已添加到项目 Resources 中"
        case .modelNotFound:
            return "找不到 RNNoise 模型文件 (std.rnnn) — 请确保模型文件已添加到项目 Resources 中"
        case .processLaunchFailed(let msg):
            return "FFmpeg 进程启动失败: \(msg)"
        case .processFailed(let code, let msg):
            return "FFmpeg 处理失败 (退出码 \(code)): \(msg)"
        case .outputFileMissing:
            return "FFmpeg 处理完成但输出文件不存在"
        }
    }
}
