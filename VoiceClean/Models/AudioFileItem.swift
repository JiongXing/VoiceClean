//
//  AudioFileItem.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import Foundation

// MARK: - 处理状态

/// 音频文件的降噪处理状态
enum ProcessingStatus: Equatable {
    /// 空闲，等待处理
    case idle
    /// 正在处理，附带进度 (0.0 ~ 1.0)
    case processing(Double)
    /// 处理完成，附带输出文件的临时 URL
    case completed(URL)
    /// 处理失败，附带错误描述
    case failed(String)

    static func == (lhs: ProcessingStatus, rhs: ProcessingStatus) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle):
            return true
        case (.processing(let a), .processing(let b)):
            return a == b
        case (.completed(let a), .completed(let b)):
            return a == b
        case (.failed(let a), .failed(let b)):
            return a == b
        default:
            return false
        }
    }

    /// 当前进度值（仅在 .processing 状态时有意义）
    var progress: Double {
        if case .processing(let p) = self { return p }
        return 0
    }

    /// 是否处理完成
    var isCompleted: Bool {
        if case .completed = self { return true }
        return false
    }

    /// 是否正在处理
    var isProcessing: Bool {
        if case .processing = self { return true }
        return false
    }

    /// 输出文件 URL（仅在 .completed 状态时有值）
    var outputURL: URL? {
        if case .completed(let url) = self { return url }
        return nil
    }

    /// 状态的显示文本
    var displayText: String {
        switch self {
        case .idle:
            return "等待处理"
        case .processing(let p):
            return "处理中 \(Int(p * 100))%"
        case .completed:
            return "已完成"
        case .failed(let msg):
            return "失败: \(msg)"
        }
    }
}

// MARK: - 音频文件数据模型

/// 表示一个待处理的音频文件
struct AudioFileItem: Identifiable {
    let id: UUID
    let url: URL
    let fileName: String
    let duration: TimeInterval
    /// 原始音频波形采样点（用于可视化，已降采样）
    var waveformSamples: [Float]
    /// 处理后的波形采样点
    var processedWaveformSamples: [Float]
    /// 处理状态
    var status: ProcessingStatus

    init(url: URL, duration: TimeInterval, waveformSamples: [Float] = []) {
        self.id = UUID()
        self.url = url
        self.fileName = url.lastPathComponent
        self.duration = duration
        self.waveformSamples = waveformSamples
        self.processedWaveformSamples = []
        self.status = .idle
    }

    /// 格式化时长显示（HH:MM:SS）
    var formattedDuration: String {
        let hours = Int(duration) / 3600
        let minutes = (Int(duration) % 3600) / 60
        let seconds = Int(duration) % 60
        if hours > 0 {
            return String(format: "%d:%02d:%02d", hours, minutes, seconds)
        }
        return String(format: "%d:%02d", minutes, seconds)
    }
}
