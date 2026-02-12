//
//  AudioDenoiser.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import Accelerate
import CoreML
import Foundation

// MARK: - 降噪引擎

/// 基于 DTLN Core ML 模型的音频降噪引擎
///
/// DTLN (Dual-Signal Transformation LSTM Network) 包含两个子模型:
/// - Model 1 (分离网络): 频谱域，输入幅度谱 → LSTM×2 → sigmoid 掩码
/// - Model 2 (增强网络): 特征域，输入时域帧 → Conv1D编码 → LayerNorm → LSTM×2
///                        → sigmoid掩码 → Multiply → Conv1D解码 → 增强帧
///
/// 处理流程: 分帧 → FFT → Model1(掩码) → 应用掩码+IFFT → Model2(增强) → Overlap-Add
final class AudioDenoiser: Sendable {

    // MARK: - 常量

    /// FFT 帧长度（DTLN 固定为 512）
    private let blockLen: Int = 512
    /// 帧移步长（DTLN 固定为 128）
    private let blockShift: Int = 128
    /// 频谱的频率 bin 数量 (blockLen / 2 + 1)
    private let numBins: Int = 257
    /// LSTM 隐藏层单元数
    private let numUnits: Int = 128
    /// LSTM 层数（每个模型 2 层）
    private let numLayers: Int = 2

    /// 降噪强度 (0.0 ~ 1.0)
    private let denoiseStrength: Float

    // MARK: - Core ML 模型

    private let model1: MLModel
    private let model2: MLModel

    // MARK: - Core ML 输出 key 名称

    /// Model 1 输出 key（通过 shape 识别 + 扰动测试确定的映射）
    private let m1MaskKey = "mask_output"
    /// Model 2 输出 key
    private let m2FrameKey = "frame_output"

    // MARK: - 初始化

    /// 初始化降噪引擎
    /// - Parameter strength: 降噪强度 (0.0 ~ 1.0)，默认 1.0（全强度）
    init(strength: Float = 1.0) throws {
        self.denoiseStrength = max(0.0, min(1.0, strength))

        let config = MLModelConfiguration()
        config.computeUnits = .all

        self.model1 = try Self.loadModel(named: "DTLN_model_1", configuration: config)
        self.model2 = try Self.loadModel(named: "DTLN_model_2", configuration: config)
    }

    /// 加载 Core ML 模型，优先使用 Xcode 预编译的 .mlmodelc，否则从 .mlpackage 运行时编译
    private static func loadModel(named name: String, configuration: MLModelConfiguration) throws -> MLModel {
        // Xcode 将 .mlpackage 自动编译为 .mlmodelc 放入 Bundle，直接加载即可
        if let compiledURL = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }

        // 回退：从 .mlpackage 运行时编译（开发调试场景）
        if let packageURL = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            let compiledURL = try MLModel.compileModel(at: packageURL)
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }

        throw DenoiserError.modelNotFound(name)
    }

    // MARK: - 主处理方法

    /// 对音频数据执行降噪处理
    /// - Parameters:
    ///   - audioData: 16kHz 单声道 Float32 PCM 数据
    ///   - onProgress: 进度回调 (0.0 ~ 1.0)
    /// - Returns: 降噪后的 Float32 PCM 数据
    func process(audioData: [Float], onProgress: @Sendable (Double) -> Void) throws -> [Float] {
        let totalSamples = audioData.count
        guard totalSamples > blockLen else {
            throw DenoiserError.audioTooShort
        }

        // ── 输出缓冲区 ──
        var outputBuffer = [Float](repeating: 0, count: totalSamples)

        // ── 初始化 LSTM 状态 (全零) ──
        // Model 1: 2 层 LSTM，每层 h 和 c
        var m1States: [MLMultiArray] = try (0..<(numLayers * 2)).map { _ in
            try createZeroArray(shape: [1, numUnits as NSNumber])
        }
        // Model 2: 2 层 LSTM，每层 h 和 c
        var m2States: [MLMultiArray] = try (0..<(numLayers * 2)).map { _ in
            try createZeroArray(shape: [1, numUnits as NSNumber])
        }

        // ── 初始化 FFT ──
        let log2n = vDSP_Length(log2(Float(blockLen)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw DenoiserError.fftSetupFailed
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // ── 汉宁窗 ──
        var window = [Float](repeating: 0, count: blockLen)
        vDSP_hann_window(&window, vDSP_Length(blockLen), Int32(vDSP_HANN_NORM))

        // ── 计算总帧数 ──
        let numFrames = max(0, (totalSamples - blockLen) / blockShift + 1)

        // ── 逐帧处理 ──
        for frameIdx in 0..<numFrames {
            let offset = frameIdx * blockShift

            // ── 提取当前帧 ──
            var inputFrame = [Float](repeating: 0, count: blockLen)
            let remaining = min(blockLen, totalSamples - offset)
            for i in 0..<remaining {
                inputFrame[i] = audioData[offset + i]
            }

            // ── FFT → 幅度谱 + 相位谱 ──
            let (magnitude, phase) = performFFT(inputFrame, setup: fftSetup, log2n: log2n)

            // ── Model 1: 频谱掩码估计 ──
            let magInput = try createMLMultiArray(shape: [1, 1, numBins as NSNumber], from: magnitude)

            var model1Dict: [String: Any] = ["mag_input": magInput]
            for i in 0..<numLayers {
                model1Dict["m1_state_h_\(i)"] = m1States[i * 2]
                model1Dict["m1_state_c_\(i)"] = m1States[i * 2 + 1]
            }

            let m1Provider = try MLDictionaryFeatureProvider(dictionary: model1Dict)
            let m1Output = try model1.prediction(from: m1Provider)

            // 提取掩码
            guard let maskArray = m1Output.featureValue(for: m1MaskKey)?.multiArrayValue else {
                throw DenoiserError.modelOutputMissing("mask from Model 1")
            }

            // 更新 Model 1 的 LSTM 状态
            for i in 0..<numLayers {
                if let h = m1Output.featureValue(for: "m1_out_h_\(i)")?.multiArrayValue {
                    m1States[i * 2] = h
                }
                if let c = m1Output.featureValue(for: "m1_out_c_\(i)")?.multiArrayValue {
                    m1States[i * 2 + 1] = c
                }
            }

            // ── 应用掩码到频谱并 IFFT 回时域 ──
            var maskedMag = [Float](repeating: 0, count: numBins)
            for i in 0..<numBins {
                var mask = Float(truncating: maskArray[[0, 0, i as NSNumber] as [NSNumber]])
                mask = adjustMask(mask)
                maskedMag[i] = magnitude[i] * mask
            }

            let estimatedFrame = performIFFT(maskedMag, phase: phase, setup: fftSetup, log2n: log2n)

            // ── Model 2: 特征域增强 ──
            // Model 2 输入是 IFFT 后的时域帧
            let frameInput = try createMLMultiArray(shape: [1, 1, blockLen as NSNumber], from: estimatedFrame)

            var model2Dict: [String: Any] = ["frame_input": frameInput]
            for i in 0..<numLayers {
                model2Dict["m2_state_h_\(i)"] = m2States[i * 2]
                model2Dict["m2_state_c_\(i)"] = m2States[i * 2 + 1]
            }

            let m2Provider = try MLDictionaryFeatureProvider(dictionary: model2Dict)
            let m2Output = try model2.prediction(from: m2Provider)

            // 提取增强帧 (Model 2 直接输出增强后的时域帧)
            guard let enhancedArray = m2Output.featureValue(for: m2FrameKey)?.multiArrayValue else {
                throw DenoiserError.modelOutputMissing("enhanced frame from Model 2")
            }

            // 更新 Model 2 的 LSTM 状态
            for i in 0..<numLayers {
                if let h = m2Output.featureValue(for: "m2_out_h_\(i)")?.multiArrayValue {
                    m2States[i * 2] = h
                }
                if let c = m2Output.featureValue(for: "m2_out_c_\(i)")?.multiArrayValue {
                    m2States[i * 2 + 1] = c
                }
            }

            // ── Overlap-Add: 将增强帧加窗并叠加到输出 ──
            for i in 0..<blockLen {
                let enhanced = Float(truncating: enhancedArray[[0, 0, i as NSNumber] as [NSNumber]])
                let idx = offset + i
                if idx < totalSamples {
                    outputBuffer[idx] += enhanced * window[i]
                }
            }

            // 上报进度
            if frameIdx % 50 == 0 || frameIdx == numFrames - 1 {
                let progress = Double(frameIdx + 1) / Double(numFrames)
                onProgress(progress)
            }
        }

        return outputBuffer
    }

    // MARK: - FFT / IFFT

    /// 对一帧时域信号执行 FFT，返回幅度谱和相位谱
    private func performFFT(_ frame: [Float], setup: FFTSetup, log2n: vDSP_Length) -> ([Float], [Float]) {
        var realPart = [Float](repeating: 0, count: blockLen / 2)
        var imagPart = [Float](repeating: 0, count: blockLen / 2)
        var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)

        frame.withUnsafeBufferPointer { bufferPtr in
            bufferPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: blockLen / 2) { complexPtr in
                vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(blockLen / 2))
            }
        }

        vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

        // 缩放
        var scale: Float = 1.0 / Float(blockLen)
        vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(blockLen / 2))
        vDSP_vsmul(imagPart, 1, &scale, &imagPart, 1, vDSP_Length(blockLen / 2))

        // 提取幅度和相位
        var magnitude = [Float](repeating: 0, count: numBins)
        var phase = [Float](repeating: 0, count: numBins)

        // DC
        magnitude[0] = abs(realPart[0])
        phase[0] = realPart[0] >= 0 ? 0 : .pi

        // Nyquist (packed in imagPart[0])
        magnitude[numBins - 1] = abs(imagPart[0])
        phase[numBins - 1] = imagPart[0] >= 0 ? 0 : .pi

        // 中间频率 bin
        for i in 1..<(numBins - 1) {
            let re = realPart[i]
            let im = imagPart[i]
            magnitude[i] = sqrt(re * re + im * im)
            phase[i] = atan2(im, re)
        }

        return (magnitude, phase)
    }

    /// 从幅度谱和相位谱执行 IFFT，返回时域帧
    private func performIFFT(_ magnitude: [Float], phase: [Float], setup: FFTSetup, log2n: vDSP_Length) -> [Float] {
        var realPart = [Float](repeating: 0, count: blockLen / 2)
        var imagPart = [Float](repeating: 0, count: blockLen / 2)

        // DC (pack into realPart[0])
        realPart[0] = magnitude[0] * cos(phase[0]) * Float(blockLen)
        // Nyquist (pack into imagPart[0])
        imagPart[0] = magnitude[numBins - 1] * cos(phase[numBins - 1]) * Float(blockLen)

        // 中间频率 bin: 从极坐标转直角坐标
        for i in 1..<(numBins - 1) {
            realPart[i] = magnitude[i] * cos(phase[i]) * Float(blockLen)
            imagPart[i] = magnitude[i] * sin(phase[i]) * Float(blockLen)
        }

        var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
        vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))

        // 转回交错格式
        var result = [Float](repeating: 0, count: blockLen)
        result.withUnsafeMutableBufferPointer { outBuf in
            outBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: blockLen / 2) { complexPtr in
                vDSP_ztoc(&splitComplex, 1, complexPtr, 2, vDSP_Length(blockLen / 2))
            }
        }

        // 缩放 IFFT 结果
        var ifftScale: Float = 1.0 / 2.0
        var scaled = [Float](repeating: 0, count: blockLen)
        vDSP_vsmul(result, 1, &ifftScale, &scaled, 1, vDSP_Length(blockLen))

        return scaled
    }

    // MARK: - 私有辅助方法

    /// 根据降噪强度调整掩码值
    private func adjustMask(_ mask: Float) -> Float {
        if denoiseStrength >= 1.0 { return mask }
        return mask * denoiseStrength + (1.0 - denoiseStrength)
    }

    /// 创建全零 MLMultiArray
    private func createZeroArray(shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let count = array.count
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = 0
        }
        return array
    }

    /// 从 Float 数组创建 MLMultiArray
    private func createMLMultiArray(shape: [NSNumber], from data: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let count = min(array.count, data.count)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            ptr[i] = data[i]
        }
        return array
    }
}

// MARK: - 错误类型

enum DenoiserError: LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case audioTooShort
    case fftSetupFailed
    case modelOutputMissing(String)
    case processingFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "找不到模型文件: \(name).mlpackage — 请确保已将模型文件添加到项目中"
        case .modelLoadFailed(let msg):
            return "模型加载失败: \(msg)"
        case .audioTooShort:
            return "音频文件太短，无法处理"
        case .fftSetupFailed:
            return "FFT 初始化失败"
        case .modelOutputMissing(let name):
            return "模型输出缺失: \(name)"
        case .processingFailed(let msg):
            return "处理失败: \(msg)"
        }
    }
}
