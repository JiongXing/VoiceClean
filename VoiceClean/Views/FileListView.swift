//
//  FileListView.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import SwiftUI

// MARK: - 文件列表视图

/// 显示已添加的音频文件列表
struct FileListView: View {

    let files: [AudioFileItem]
    let selectedID: UUID?
    var onSelect: (UUID) -> Void
    var onRemove: (AudioFileItem) -> Void
    var onExport: (AudioFileItem) -> Void

    var body: some View {
        if files.isEmpty {
            emptyStateView
        } else {
            ScrollView {
                LazyVStack(spacing: 6) {
                    ForEach(files) { file in
                        AudioFileRow(
                            file: file,
                            isSelected: file.id == selectedID,
                            onRemove: { onRemove(file) },
                            onExport: { onExport(file) }
                        )
                        .onTapGesture {
                            onSelect(file.id)
                        }
                    }
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 4)
            }
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: 8) {
            Image(systemName: "music.note.list")
                .font(.title2)
                .foregroundStyle(.tertiary)
            Text("暂无音频文件")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, minHeight: 80)
    }
}

// MARK: - 单个文件行

/// 文件列表中的单行，显示文件信息和状态
struct AudioFileRow: View {

    let file: AudioFileItem
    let isSelected: Bool
    var onRemove: () -> Void
    var onExport: () -> Void

    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 12) {
            // 状态图标
            statusIcon
                .frame(width: 28, height: 28)

            // 文件信息
            VStack(alignment: .leading, spacing: 2) {
                Text(file.fileName)
                    .font(.system(.body, design: .default, weight: .medium))
                    .lineLimit(1)
                    .truncationMode(.middle)

                HStack(spacing: 8) {
                    Text(file.formattedDuration)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text(file.status.displayText)
                        .font(.caption)
                        .foregroundStyle(statusColor)
                }
            }

            Spacer()

            // 进度条（处理中时显示）
            if case .processing(let progress) = file.status {
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .frame(width: 80)
                    .tint(.accentColor)
            }

            // 操作按钮
            HStack(spacing: 4) {
                if file.status.isCompleted {
                    Button(action: onExport) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 13))
                    }
                    .buttonStyle(.borderless)
                    .help("导出文件")
                }

                if !file.status.isProcessing {
                    Button(action: onRemove) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.borderless)
                    .opacity(isHovered ? 1 : 0)
                    .help("移除文件")
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? Color.accentColor.opacity(0.12) : (isHovered ? Color.primary.opacity(0.04) : .clear))
        }
        .overlay {
            if isSelected {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.accentColor.opacity(0.3), lineWidth: 1)
            }
        }
        .onHover { hovering in
            isHovered = hovering
        }
        .animation(.easeInOut(duration: 0.15), value: isHovered)
        .animation(.easeInOut(duration: 0.15), value: isSelected)
    }

    // MARK: - 子视图

    @ViewBuilder
    private var statusIcon: some View {
        switch file.status {
        case .idle:
            Image(systemName: "music.note")
                .foregroundStyle(.secondary)
                .font(.system(size: 14))

        case .processing:
            ProgressView()
                .controlSize(.small)

        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.system(size: 16))

        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
                .font(.system(size: 14))
        }
    }

    private var statusColor: Color {
        switch file.status {
        case .idle: return .secondary
        case .processing: return .accentColor
        case .completed: return .green
        case .failed: return .red
        }
    }
}

#Preview {
    let files: [AudioFileItem] = [
        AudioFileItem(url: URL(fileURLWithPath: "/test/lecture_01.mp3"), duration: 2730),
        AudioFileItem(url: URL(fileURLWithPath: "/test/lecture_02.mp3"), duration: 4815),
        AudioFileItem(url: URL(fileURLWithPath: "/test/short_clip.mp3"), duration: 300),
    ]

    return FileListView(
        files: files,
        selectedID: files.first?.id,
        onSelect: { _ in },
        onRemove: { _ in },
        onExport: { _ in }
    )
    .frame(width: 500, height: 300)
    .padding()
}
