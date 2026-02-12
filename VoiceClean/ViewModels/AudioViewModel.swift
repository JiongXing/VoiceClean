//
//  AudioViewModel.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import Foundation
import Observation

// MARK: - 主 ViewModel

/// 管理音频文件列表和降噪处理的核心 ViewModel
@Observable
@MainActor
final class AudioViewModel {

    // MARK: - 公开状态

    /// 已添加的音频文件列表
    var audioFiles: [AudioFileItem] = []

    /// 是否正在处理
    var isProcessing: Bool = false

    /// 降噪强度 (0.0 ~ 1.0)
    var denoiseStrength: Double = 0.8

    /// 全局错误消息（用于 Alert 展示）
    var errorMessage: String?
    var showError: Bool = false

    /// 当前选中的文件 ID（用于详情/波形展示）
    var selectedFileID: UUID?

    // MARK: - 计算属性

    /// 是否有文件可以处理
    var hasFilesToProcess: Bool {
        audioFiles.contains { $0.status == .idle || $0.status.displayText.hasPrefix("失败") }
    }

    /// 是否有文件已完成处理
    var hasCompletedFiles: Bool {
        audioFiles.contains { $0.status.isCompleted }
    }

    /// 整体处理进度
    var overallProgress: Double {
        guard !audioFiles.isEmpty else { return 0 }
        let total = audioFiles.reduce(0.0) { sum, file in
            switch file.status {
            case .completed: return sum + 1.0
            case .processing(let p): return sum + p
            default: return sum
            }
        }
        return total / Double(audioFiles.count)
    }

    // MARK: - 文件管理

    /// 通过文件选择面板添加文件
    func addFiles() async {
        let urls = await AudioFileService.openFilePicker()
        await addFiles(from: urls)
    }

    /// 通过 URL 列表添加文件（支持拖拽）
    func addFiles(from urls: [URL]) async {
        for url in urls {
            // 避免重复添加
            guard !audioFiles.contains(where: { $0.url == url }) else { continue }

            // 检查文件扩展名
            let ext = url.pathExtension.lowercased()
            guard ["mp3", "m4a", "wav", "aac", "aiff", "flac"].contains(ext) else { continue }

            do {
                let duration = try AudioFileService.getAudioDuration(url: url)

                // 预加载波形数据
                let audioData = try AudioFileService.loadAndResample(url: url)
                let waveform = AudioFileService.extractWaveformSamples(from: audioData)

                let item = AudioFileItem(
                    url: url,
                    duration: duration,
                    waveformSamples: waveform
                )
                audioFiles.append(item)
            } catch {
                showErrorMessage("无法读取文件 \(url.lastPathComponent): \(error.localizedDescription)")
            }
        }
    }

    /// 移除指定文件
    func removeFile(_ item: AudioFileItem) {
        audioFiles.removeAll { $0.id == item.id }
        if selectedFileID == item.id {
            selectedFileID = nil
        }
    }

    /// 移除所有文件
    func removeAll() {
        audioFiles.removeAll()
        selectedFileID = nil
    }

    // MARK: - 降噪处理

    /// 处理所有待处理的文件
    func processAll() async {
        guard !isProcessing else { return }
        isProcessing = true
        defer { isProcessing = false }

        for i in audioFiles.indices {
            guard audioFiles[i].status == .idle
                || audioFiles[i].status.displayText.hasPrefix("失败")
            else { continue }

            await processFile(at: i)
        }
    }

    /// 处理单个文件
    func processSingleFile(_ item: AudioFileItem) async {
        guard let index = audioFiles.firstIndex(where: { $0.id == item.id }) else { return }
        guard !isProcessing else { return }

        isProcessing = true
        defer { isProcessing = false }

        await processFile(at: index)
    }

    /// 导出单个已完成的文件
    func exportFile(_ item: AudioFileItem) async {
        guard case .completed(let tempURL) = item.status else { return }

        let suggestedName = (item.fileName as NSString).deletingPathExtension + "_denoised.wav"
        guard let saveURL = await AudioFileService.openSavePanel(suggestedName: suggestedName) else { return }

        do {
            try AudioFileService.exportFile(from: tempURL, to: saveURL)
        } catch {
            showErrorMessage("导出失败: \(error.localizedDescription)")
        }
    }

    /// 导出所有已完成的文件
    func exportAll() async {
        for item in audioFiles where item.status.isCompleted {
            await exportFile(item)
        }
    }

    // MARK: - 私有方法

    /// 处理指定索引的文件
    private func processFile(at index: Int) async {
        guard index < audioFiles.count else { return }

        audioFiles[index].status = .processing(0)

        let url = audioFiles[index].url
        let fileName = audioFiles[index].fileName
        let strength = Float(denoiseStrength)
        let fileIndex = index

        do {
            // 所有重操作通过 GCD 在全局队列执行，确保不在主线程
            let result: (waveform: [Float], tempURL: URL) = try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        // 加载并重采样音频
                        let audioData = try AudioFileService.loadAndResample(url: url)

                        // 初始化降噪引擎 + 执行降噪
                        let denoiser = try AudioDenoiser(strength: strength)
                        let processedData = try denoiser.process(audioData: audioData) { progress in
                            DispatchQueue.main.async {
                                self.audioFiles[fileIndex].status = .processing(progress)
                            }
                        }

                        // 提取波形 + 保存临时文件
                        let waveform = AudioFileService.extractWaveformSamples(from: processedData)
                        let tempURL = try AudioFileService.saveToTempFile(
                            data: processedData,
                            originalFileName: fileName
                        )

                        continuation.resume(returning: (waveform, tempURL))
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }

            guard index < audioFiles.count else { return }
            audioFiles[index].processedWaveformSamples = result.waveform
            audioFiles[index].status = .completed(result.tempURL)

        } catch {
            guard index < audioFiles.count else { return }
            audioFiles[index].status = .failed(error.localizedDescription)
            showErrorMessage("处理 \(audioFiles[index].fileName) 失败: \(error.localizedDescription)")
        }
    }

    /// 显示错误消息
    private func showErrorMessage(_ message: String) {
        errorMessage = message
        showError = true
    }
}
