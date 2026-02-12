//
//  WaveformView.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import SwiftUI

// MARK: - 波形可视化视图

/// 显示音频波形的视图，支持原始和处理后的波形对比
struct WaveformView: View {

    /// 波形采样数据
    let samples: [Float]

    /// 波形颜色
    var color: Color = .accentColor

    /// 波形线条宽度
    var lineWidth: CGFloat = 1.5

    /// 是否镜像显示（上下对称）
    var mirrored: Bool = true

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let height = geometry.size.height
            let midY = height / 2

            if samples.isEmpty {
                // 空状态：显示一条中间线
                Path { path in
                    path.move(to: CGPoint(x: 0, y: midY))
                    path.addLine(to: CGPoint(x: width, y: midY))
                }
                .stroke(color.opacity(0.3), lineWidth: 0.5)
            } else {
                // 绘制波形
                Canvas { context, size in
                    let barCount = samples.count
                    guard barCount > 0 else { return }

                    let barWidth = max(1, size.width / CGFloat(barCount))
                    let maxAmplitude = samples.max() ?? 1.0
                    let normalizer: Float = maxAmplitude > 0 ? maxAmplitude : 1.0

                    for i in 0..<barCount {
                        let normalized = CGFloat(samples[i] / normalizer)
                        let barHeight = max(1, normalized * (size.height / 2 - 2))
                        let x = CGFloat(i) * barWidth

                        let rect: CGRect
                        if mirrored {
                            rect = CGRect(
                                x: x,
                                y: midY - barHeight,
                                width: max(1, barWidth - 0.5),
                                height: barHeight * 2
                            )
                        } else {
                            rect = CGRect(
                                x: x,
                                y: size.height - barHeight,
                                width: max(1, barWidth - 0.5),
                                height: barHeight
                            )
                        }

                        let roundedRect = RoundedRectangle(cornerRadius: barWidth / 3)
                            .path(in: rect)

                        // 渐变透明度：中间高、两端低
                        let opacity = 0.4 + 0.6 * Double(normalized)
                        context.fill(roundedRect, with: .color(color.opacity(opacity)))
                    }
                }
            }
        }
    }
}

// MARK: - 波形对比视图

/// 并排显示原始和处理后的波形
struct WaveformComparisonView: View {

    let originalSamples: [Float]
    let processedSamples: [Float]

    var body: some View {
        VStack(spacing: 8) {
            // 原始波形
            VStack(alignment: .leading, spacing: 4) {
                Label("原始音频", systemImage: "waveform")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                WaveformView(samples: originalSamples, color: .secondary)
                    .frame(height: 50)
            }

            // 处理后波形
            VStack(alignment: .leading, spacing: 4) {
                Label("降噪后", systemImage: "waveform.path.ecg")
                    .font(.caption)
                    .foregroundStyle(Color.accentColor)

                WaveformView(
                    samples: processedSamples.isEmpty ? originalSamples : processedSamples,
                    color: .accentColor
                )
                .frame(height: 50)
                .opacity(processedSamples.isEmpty ? 0.3 : 1.0)
            }
        }
    }
}

#Preview {
    let sampleData: [Float] = (0..<200).map { _ in Float.random(in: 0.05...1.0) }
    let processedData: [Float] = sampleData.map { max(0.02, $0 * 0.4) }

    return VStack(spacing: 20) {
        WaveformView(samples: sampleData, color: .blue)
            .frame(height: 80)

        WaveformComparisonView(
            originalSamples: sampleData,
            processedSamples: processedData
        )
    }
    .padding()
    .frame(width: 500)
}
