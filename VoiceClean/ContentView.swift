//
//  ContentView.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import SwiftUI

// MARK: - 主界面

struct ContentView: View {

    @State private var viewModel = AudioViewModel()

    var body: some View {
        VStack(spacing: 0) {
            // 顶部标题栏
            headerView
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 12)

            Divider()
                .padding(.horizontal, 16)

            // 主内容区域
            ScrollView {
                VStack(spacing: 16) {
                    // 拖拽区域
                    DropZoneView(
                        onDrop: { urls in
                            Task { await viewModel.addFiles(from: urls) }
                        },
                        onTap: {
                            Task { await viewModel.addFiles() }
                        }
                    )

                    // 文件列表
                    if !viewModel.audioFiles.isEmpty {
                        fileListSection
                    }

                    // 波形预览（选中文件时显示）
                    if let selectedFile = selectedFile {
                        waveformSection(for: selectedFile)
                    }

                    // 降噪控制
                    if !viewModel.audioFiles.isEmpty {
                        controlsSection
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }

            // 底部操作栏
            if !viewModel.audioFiles.isEmpty {
                Divider()
                    .padding(.horizontal, 16)

                bottomBar
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
            }
        }
        .frame(minWidth: 600, minHeight: 450)
        .background(Color(nsColor: .windowBackgroundColor))
        .alert("错误", isPresented: $viewModel.showError) {
            Button("确定", role: .cancel) {}
        } message: {
            if let msg = viewModel.errorMessage {
                Text(msg)
            }
        }
    }

    // MARK: - 当前选中的文件

    private var selectedFile: AudioFileItem? {
        guard let id = viewModel.selectedFileID else {
            return viewModel.audioFiles.first
        }
        return viewModel.audioFiles.first { $0.id == id }
    }

    // MARK: - 顶部标题栏

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 8) {
                    Image(systemName: "waveform.path.ecg")
                        .font(.title2)
                        .foregroundStyle(Color.accentColor)

                    Text("VoiceClean")
                        .font(.title2)
                        .fontWeight(.semibold)
                }

                Text("音频降噪 · 突出人声")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            // 文件数量统计
            if !viewModel.audioFiles.isEmpty {
                HStack(spacing: 12) {
                    Label(
                        "\(viewModel.audioFiles.count) 个文件",
                        systemImage: "doc.on.doc"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    let completed = viewModel.audioFiles.filter { $0.status.isCompleted }.count
                    if completed > 0 {
                        Label(
                            "\(completed) 已完成",
                            systemImage: "checkmark.circle"
                        )
                        .font(.caption)
                        .foregroundStyle(.green)
                    }
                }
            }
        }
    }

    // MARK: - 文件列表区域

    private var fileListSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("文件列表", systemImage: "list.bullet")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundStyle(.secondary)

                Spacer()

                if viewModel.audioFiles.count > 1 && !viewModel.isProcessing {
                    Button("清空全部", role: .destructive) {
                        viewModel.removeAll()
                    }
                    .font(.caption)
                    .buttonStyle(.borderless)
                }
            }

            FileListView(
                files: viewModel.audioFiles,
                selectedID: viewModel.selectedFileID ?? viewModel.audioFiles.first?.id,
                onSelect: { id in
                    viewModel.selectedFileID = id
                },
                onRemove: { file in
                    viewModel.removeFile(file)
                },
                onExport: { file in
                    Task { await viewModel.exportFile(file) }
                }
            )
            .frame(minHeight: 60, maxHeight: 200)
            .background {
                RoundedRectangle(cornerRadius: 10)
                    .fill(.ultraThinMaterial)
            }
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    // MARK: - 波形预览区域

    private func waveformSection(for file: AudioFileItem) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("波形预览 — \(file.fileName)", systemImage: "waveform")
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            WaveformComparisonView(
                originalSamples: file.waveformSamples,
                processedSamples: file.processedWaveformSamples
            )
            .padding(12)
            .background {
                RoundedRectangle(cornerRadius: 10)
                    .fill(.ultraThinMaterial)
            }
        }
    }

    // MARK: - 降噪控制区域

    private var controlsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("降噪强度", systemImage: "slider.horizontal.3")
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                Text("轻度")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 30)

                Slider(value: $viewModel.denoiseStrength, in: 0.1...1.0, step: 0.1)
                    .tint(.accentColor)
                    .disabled(viewModel.isProcessing)

                Text("强力")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 30)

                Text("\(Int(viewModel.denoiseStrength * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 36)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background {
                RoundedRectangle(cornerRadius: 10)
                    .fill(.ultraThinMaterial)
            }
        }
    }

    // MARK: - 底部操作栏

    private var bottomBar: some View {
        HStack {
            // 整体进度
            if viewModel.isProcessing {
                ProgressView(value: viewModel.overallProgress)
                    .progressViewStyle(.linear)
                    .frame(maxWidth: 200)
                    .tint(.accentColor)

                Text("\(Int(viewModel.overallProgress * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 36)
            }

            Spacer()

            HStack(spacing: 12) {
                // 导出全部
                if viewModel.hasCompletedFiles {
                    Button {
                        Task { await viewModel.exportAll() }
                    } label: {
                        Label("全部导出", systemImage: "square.and.arrow.up")
                    }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.isProcessing)
                }

                // 开始降噪
                Button {
                    Task { await viewModel.processAll() }
                } label: {
                    Label(
                        viewModel.isProcessing ? "处理中..." : "开始降噪",
                        systemImage: viewModel.isProcessing ? "hourglass" : "wand.and.stars"
                    )
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isProcessing || !viewModel.hasFilesToProcess)
            }
        }
    }
}

#Preview {
    ContentView()
        .frame(width: 800, height: 560)
}
