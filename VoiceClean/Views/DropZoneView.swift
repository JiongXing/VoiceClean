//
//  DropZoneView.swift
//  VoiceClean
//
//  Created by jxing on 2026/2/12.
//

import SwiftUI
import UniformTypeIdentifiers

// MARK: - 拖拽区域视图

/// 文件拖拽和选择区域，支持拖入 MP3 文件或点击选择
struct DropZoneView: View {

    /// 是否正在拖拽悬停
    @State private var isTargeted = false

    /// 图标动画
    @State private var iconBounce = false

    /// 拖入文件的回调
    var onDrop: ([URL]) -> Void

    /// 点击选择文件的回调
    var onTap: () -> Void

    var body: some View {
        ZStack {
            // 背景
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .strokeBorder(
                            isTargeted
                                ? Color.accentColor
                                : Color.secondary.opacity(0.3),
                            style: StrokeStyle(
                                lineWidth: isTargeted ? 2.5 : 1.5,
                                dash: [8, 4]
                            )
                        )
                )

            // 内容
            VStack(spacing: 12) {
                Image(systemName: "waveform.badge.plus")
                    .font(.system(size: 40, weight: .light))
                    .foregroundStyle(
                        isTargeted
                            ? Color.accentColor
                            : .secondary
                    )
                    .symbolEffect(.bounce, value: iconBounce)
                    .scaleEffect(isTargeted ? 1.15 : 1.0)
                    .animation(.spring(response: 0.3), value: isTargeted)

                VStack(spacing: 4) {
                    Text("拖拽音频文件到此处")
                        .font(.headline)
                        .foregroundStyle(.primary)

                    Text("或点击选择文件")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    Text("支持 MP3、M4A、WAV、AAC、FLAC 格式")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .padding(.top, 2)
                }
            }
            .padding()
        }
        .frame(minHeight: 140, maxHeight: 160)
        .onTapGesture {
            iconBounce.toggle()
            onTap()
        }
        .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
            handleDrop(providers: providers)
            return true
        }
        .animation(.easeInOut(duration: 0.2), value: isTargeted)
    }

    // MARK: - 拖拽处理

    private func handleDrop(providers: [NSItemProvider]) {
        var urls: [URL] = []
        let group = DispatchGroup()

        for provider in providers {
            guard provider.canLoadObject(ofClass: URL.self) else { continue }
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url {
                    let ext = url.pathExtension.lowercased()
                    if ["mp3", "m4a", "wav", "aac", "aiff", "flac"].contains(ext) {
                        urls.append(url)
                    }
                }
                group.leave()
            }
        }

        group.notify(queue: .main) {
            if !urls.isEmpty {
                onDrop(urls)
            }
        }
    }
}

#Preview {
    DropZoneView(
        onDrop: { urls in print("Dropped: \(urls)") },
        onTap: { print("Tapped") }
    )
    .padding()
    .frame(width: 500, height: 200)
}
