# VoiceClean 项目 — iOS 面试参考文档

## 一、项目概述

VoiceClean 是一款 **macOS 本地音频降噪应用**，核心场景为处理讲座、会议等以人声为主的录音文件，通过深度学习模型实现背景噪声抑制、人声增强，最终导出高质量的音频文件。

**技术亮点**：
- 基于 DTLN（Dual-Signal Transformation LSTM Network）深度学习模型，通过 Core ML 在本地 GPU/Neural Engine 上推理，全程离线、保护隐私
- 使用 Accelerate 框架（vDSP）实现高性能 FFT/IFFT 信号处理
- 纯 SwiftUI 构建的现代 macOS 界面，支持拖拽导入、波形可视化、实时进度反馈
- 采用 MVVM + Observation 框架的现代架构，async/await 驱动的异步处理流程

---

## 二、架构设计

### 2.1 整体分层架构

```
┌─────────────────────────────────────────────────┐
│                   View Layer                     │
│  ContentView / DropZoneView / FileListView /     │
│  WaveformView / WaveformComparisonView           │
└──────────────────────┬──────────────────────────┘
                       │ @Observable 绑定
┌──────────────────────▼──────────────────────────┐
│                ViewModel Layer                    │
│              AudioViewModel                       │
│  - 状态管理 (audioFiles, isProcessing, progress)  │
│  - 业务逻辑编排 (addFiles, processAll, export)    │
└──────┬──────────────────────────────┬───────────┘
       │                              │
┌──────▼──────────┐     ┌────────────▼────────────┐
│ AudioFileService │     │    AudioDenoiser        │
│ - 文件选择(NS*Panel) │  │ - Core ML 推理          │
│ - AVAudioFile 读写  │  │ - vDSP FFT/IFFT        │
│ - 采样率转换         │  │ - LSTM 状态管理          │
│ - 波形数据提取       │  │ - Overlap-Add 重建       │
└─────────────────┘     └─────────────────────────┘
       │                              │
┌──────▼──────────────────────────────▼───────────┐
│              系统框架层                            │
│  AVFoundation / CoreML / Accelerate / AppKit     │
└─────────────────────────────────────────────────┘
```

### 2.2 核心设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| UI 框架 | SwiftUI | macOS 26.2 部署目标，充分利用声明式 UI 优势 |
| 状态管理 | `@Observable` (Observation) | 比 ObservableObject/Combine 更轻量、性能更好 |
| 降噪方案 | DTLN + Core ML | 模型仅 ~3MB，推理速度快，效果经 DNS Challenge 验证 |
| 信号处理 | Accelerate/vDSP | Apple 原生 SIMD 加速框架，FFT 性能远超纯 Swift 实现 |
| 并发模型 | async/await + Task.detached | 主线程保持 UI 响应，后台线程执行 CPU 密集型推理 |
| 服务层设计 | enum 静态方法 | AudioFileService 无状态，enum 防止意外实例化 |

---

## 三、核心技术实现

### 3.1 DTLN 降噪模型原理

DTLN 是一个**双路径 LSTM 网络**，包含两个级联的子网络：

```
                          音频帧 (512 samples)
                                │
                     ┌──────────▼──────────┐
                     │       FFT (vDSP)     │
                     │  → 幅度谱 + 相位谱    │
                     └──────────┬──────────┘
                                │ 幅度谱 (257 bins)
                     ┌──────────▼──────────┐
                     │   Model 1 (Core ML)  │
                     │   LSTM×2 → sigmoid   │
                     │   → 频谱掩码 (0~1)    │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   掩码 × 幅度谱       │
                     │   + 原始相位 → IFFT   │
                     │   → 估计时域帧        │
                     └──────────┬──────────┘
                                │ 时域帧 (512 samples)
                     ┌──────────▼──────────┐
                     │   Model 2 (Core ML)  │
                     │   Conv1D 编码 (256d)  │
                     │   LayerNorm → LSTM×2 │
                     │   → sigmoid 掩码      │
                     │   → Multiply          │
                     │   → Conv1D 解码 (512d)│
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Overlap-Add        │
                     │   汉宁窗 + 帧叠加     │
                     └──────────┬──────────┘
                                │
                          降噪后的音频
```

**关键参数**：帧长 512、帧移 128、FFT 点数 512、LSTM 隐藏层 128 单元 × 2 层、采样率 16kHz。

### 3.2 Core ML 集成要点

模型转换链路：`TensorFlow .h5 → Keras 重建 → coremltools → .mlpackage`

转换过程中解决的关键问题：
1. **Keras 3 兼容性**：TF 2.20 内置 Keras 3，不允许直接对 KerasTensor 使用 `tf.*` 操作，必须包裹在 `Lambda` 层中
2. **LSTM 控制流**：coremltools 无法处理 Keras 3 LSTM 的 while loop，通过设置 `unroll=True` 消除循环（因为 seq_len=1，展开无额外开销）
3. **有状态推理**：LSTM 隐藏状态需在帧间传递，每个模型有 4 个状态张量（2 层 × h/c），作为模型的输入/输出显式管理
4. **权重分片**：完整模型 20 个权重张量，前 8 个属于 Model 1（LSTM×2 + Dense），后 12 个属于 Model 2（Conv1D + LayerNorm + LSTM×2 + Dense + Conv1D）

### 3.3 音频处理管线（AudioDenoiser）

```swift
// 核心处理循环 — 逐帧推理
for frameIdx in 0..<numFrames {
    // 1. 提取帧
    let inputFrame = extractFrame(audioData, at: frameIdx * blockShift)

    // 2. FFT（vDSP 加速）
    let (magnitude, phase) = performFFT(inputFrame)

    // 3. Model 1: 频谱掩码
    let mask = model1.predict(magnitude, states: m1States)
    // → 更新 m1States（LSTM 隐藏态跨帧传递）

    // 4. 应用掩码 + IFFT
    let maskedMag = magnitude * mask * denoiseStrength
    let estimatedFrame = performIFFT(maskedMag, phase: phase)

    // 5. Model 2: 特征域增强
    let enhancedFrame = model2.predict(estimatedFrame, states: m2States)
    // → 更新 m2States

    // 6. Overlap-Add（汉宁窗加权叠加）
    overlapAdd(enhancedFrame, to: outputBuffer, at: offset)
}
```

### 3.4 并发设计

```swift
// ViewModel (MainActor) 中的处理调度
func processFile(at index: Int) async {
    audioFiles[index].status = .processing(0)

    let processedData = try await Task.detached(priority: .userInitiated) {
        // 后台线程：CPU 密集型推理
        let denoiser = try AudioDenoiser(strength: strength)
        return try denoiser.process(audioData: audioData) { progress in
            // 进度回调：跨线程回到 MainActor 更新 UI
            Task { @MainActor in
                self.audioFiles[index].status = .processing(progress)
            }
        }
    }.value

    audioFiles[index].status = .completed(tempURL)
}
```

关键设计：
- `AudioViewModel` 标记为 `@MainActor`，确保 UI 状态更新的线程安全
- `AudioDenoiser` 标记为 `Sendable`（final class + 全部不可变属性），安全在 `Task.detached` 中使用
- 进度回调通过 `Task { @MainActor in ... }` 从后台跨越到主线程
- 使用 `[weak self]` 避免闭包对 ViewModel 的强引用循环

### 3.5 音频 I/O（AudioFileService）

```
MP3/M4A/WAV 输入
       │
  AVAudioFile 读取
       │
  AVAudioConverter 重采样 → 16kHz 单声道 Float32
       │
  [Float] PCM 数据 → 送入降噪引擎
       │
  降噪后 [Float] → AVAudioFile 写入 → .wav 输出
```

设计为 `enum` 的无状态服务类，所有方法为 `static`，防止意外实例化。多声道自动混合为单声道。

---

## 四、UI 设计

### 4.1 界面结构

- **DropZoneView**：拖拽区域，虚线边框 + `symbolEffect(.bounce)` 动画，支持 `onDrop` + 点击选择
- **FileListView**：`LazyVStack` 文件列表，悬停显示删除按钮，选中状态高亮
- **WaveformView**：`Canvas` 绘制的波形可视化，RMS 采样 + 镜像显示
- **WaveformComparisonView**：处理前/后波形对比
- **降噪强度 Slider**：通过线性插值 `mask * strength + (1 - strength)` 控制掩码力度

### 4.2 macOS 适配

- `NSOpenPanel` / `NSSavePanel` 文件选择
- `.ultraThinMaterial` 毛玻璃背景
- `.onHover` 悬停效果
- `WindowGroup.defaultSize` / `.windowResizability` 窗口尺寸控制
- App Sandbox 配置 `ENABLE_USER_SELECTED_FILES = readwrite`

---

## 五、面试问答

### Q1：为什么选择 DTLN 而不是其他降噪方案（如传统频谱减法或更大的模型）？

**回答**：这是一个 **效果、体积、延迟** 三者的权衡。

传统频谱减法（Spectral Subtraction）虽然计算量小，但只能处理平稳噪声（如恒定的空调嗡嗡声），对非平稳噪声（如键盘敲击、翻书声）几乎无效，而且容易引入"音乐噪声"伪影。

更大的模型如 Demucs（Facebook）效果确实更好，但模型体积 100MB+，不适合嵌入到桌面 App 中分发。

DTLN 的优势在于：
1. **模型极小**（~3MB），两个子模型合计约 88 万参数
2. **专为语音设计**，在 DNS Challenge（微软深度降噪挑战赛）中取得过前十名
3. **双路径架构**：Model 1 在频谱域估计掩码处理宽带噪声，Model 2 在学习的特征域做精细增强，两级级联互补
4. **因果模型**（causal），可扩展为实时流式处理

综合来看，DTLN 在"够用的效果"和"极低的资源消耗"之间达到了很好的平衡。

---

### Q2：AudioDenoiser 被标记为 Sendable，但内部持有 MLModel 引用，这是如何保证线程安全的？

**回答**：`AudioDenoiser` 是一个 `final class`，它的所有存储属性都是 `let`（不可变的）：

```swift
final class AudioDenoiser: Sendable {
    private let blockLen: Int = 512
    private let denoiseStrength: Float
    private let model1: MLModel
    private let model2: MLModel
    // ...
}
```

关键点：
1. `MLModel` 本身是线程安全的 — Apple 文档明确说明 `MLModel.prediction(from:)` 可以从任意线程调用，内部有适当的同步机制
2. 所有可变状态（LSTM 隐藏态、FFT 缓冲区）都是 `process()` 方法内的**局部变量**，不是实例属性，所以天然线程安全
3. `denoiseStrength` 在 `init` 时确定后不再变化

这种设计保证了每次调用 `process()` 都是独立的、无副作用的。同时因为在 `Task.detached` 中每次新建 `AudioDenoiser` 实例，不存在多线程共享同一实例的情况。

如果未来需要复用同一实例并发处理多个文件，当前设计也是安全的，因为 `process()` 不修改任何实例状态。

---

### Q3：为什么 AudioFileService 用 enum 而不是 struct 或 class？

**回答**：这是 Swift 中一种**命名空间模式**（Caseless Enum Pattern）。

```swift
enum AudioFileService {
    static func openFilePicker() async -> [URL] { ... }
    static func loadAndResample(url: URL) throws -> [Float] { ... }
    // ...
}
```

用 `enum` 而非 `struct` 或 `class` 的原因：
- `AudioFileService` 是一个**纯工具集**，没有任何实例状态需要管理
- `enum` 没有 case 时**无法被实例化**（`AudioFileService()` 编译报错），从语义上强制表达"这是一个命名空间，不是对象"
- 如果用 `struct`，开发者可能误写 `let service = AudioFileService()` 创建无意义的实例
- 如果用 `class`，还会引入引用语义的复杂性

这种模式在 Apple 自己的代码中也有使用（如 `DispatchQueue.main`），是 Swift 社区广泛认可的最佳实践。

---

### Q4：vDSP FFT 的 packed format 是什么？为什么 DC 和 Nyquist 分量要特殊处理？

**回答**：这涉及 vDSP 实数 FFT 的**紧凑存储格式**。

对一个长度为 N 的实数信号做 FFT，理论上输出 N/2+1 个复数频率分量。但由于实数信号的 FFT 具有共轭对称性，vDSP 的 `vDSP_fft_zrip`（in-place real FFT）使用一种特殊的 packed 格式来节省内存：

- `realPart[0]` 存储 **DC 分量**（0 Hz）的实部（虚部恒为 0）
- `imagPart[0]` 存储 **Nyquist 分量**（fs/2）的实部（虚部也恒为 0）
- `realPart[1..N/2-1]` 和 `imagPart[1..N/2-1]` 正常存储中间频率的实部和虚部

所以在代码中需要特殊处理：

```swift
// DC
magnitude[0] = abs(realPart[0])
phase[0] = realPart[0] >= 0 ? 0 : .pi

// Nyquist (packed in imagPart[0])
magnitude[numBins - 1] = abs(imagPart[0])
phase[numBins - 1] = imagPart[0] >= 0 ? 0 : .pi
```

DC 和 Nyquist 分量的虚部必为 0，相位只有 0 或 π（正或负实数），这是实数 FFT 的数学性质决定的。如果不正确处理这两个特殊位置，IFFT 重建后的信号会出现严重失真。

---

### Q5：如果要将这个 App 从 macOS 移植到 iOS，需要做哪些改动？

**回答**：核心降噪引擎（`AudioDenoiser`）和文件服务的业务逻辑**完全不需要改动**，它们只依赖 `AVFoundation`、`CoreML`、`Accelerate` 这些跨平台框架。

需要改动的部分：

1. **文件选择 UI**：
   - macOS 使用 `NSOpenPanel` / `NSSavePanel`（AppKit）
   - iOS 需要改为 `UIDocumentPickerViewController` 或 SwiftUI 的 `.fileImporter()` / `.fileExporter()` 修饰符
   - 这是改动量最大的一处

2. **Sandbox 权限**：
   - macOS 使用 `ENABLE_USER_SELECTED_FILES`
   - iOS 需要在 Info.plist 中配置 `UIFileSharingEnabled` 和 Document Types

3. **UI 适配**：
   - `.ultraThinMaterial` 在 iOS 上也可用，但视觉效果不同
   - `.onHover` 在 iOS 上无效（无鼠标），需移除
   - 窗口尺寸控制（`.defaultSize`、`.windowResizability`）移除
   - 拖拽（`.onDrop`）在 iOS 上可用但交互方式不同，建议改为以 `.fileImporter()` 为主

4. **App 生命周期**：
   - iOS 需要处理后台中断（`UIApplication.willResignActiveNotification`），在降噪处理中可能需要申请 Background Task
   - 或使用 `BGProcessingTask` 在后台继续处理长音频

5. **性能考虑**：
   - iPhone 的 Neural Engine 同样支持 Core ML，性能可能更好（Apple 在 iPhone 芯片上投入更多 ML 硬件）
   - 但需要注意内存限制：iOS 对单个 App 的内存使用限制更严格，处理超长音频（>1 小时）时可能需要分段加载

整体来说，由于架构做了良好的分层，移植工作量预计只占总代码量的 15-20%，主要集中在 View 层和文件 I/O 的平台适配。

---

### Q6：Observation 框架的 @Observable 相比传统的 ObservableObject + @Published 有什么优势？

**回答**：`@Observable`（iOS 17+ / macOS 14+）是 Apple 推出的下一代响应式状态管理方案，相比 `ObservableObject` + `@Published` 有三个核心优势：

**1. 属性级别的精细追踪**

```swift
// ObservableObject: 任何 @Published 变化 → 整个 View body 重新计算
class OldVM: ObservableObject {
    @Published var audioFiles: [AudioFileItem] = []
    @Published var denoiseStrength: Double = 0.8  // 拖动 Slider 时，FileListView 也会重绘
}

// @Observable: 只追踪 View body 中实际读取的属性
@Observable class NewVM {
    var audioFiles: [AudioFileItem] = []
    var denoiseStrength: Double = 0.8  // 拖动 Slider 时，只有读取了 denoiseStrength 的 View 重绘
}
```

在本项目中，`ContentView` 的 `controlsSection` 读取 `denoiseStrength`，`fileListSection` 读取 `audioFiles`。使用 `@Observable` 后，拖动降噪强度滑块**不会导致文件列表重绘**。

**2. 消除模板代码**

不再需要 `@Published`、`@StateObject`、`@ObservedObject`、`@EnvironmentObject` 的区分。一个 `@State private var viewModel = AudioViewModel()` 搞定一切。

**3. 与 SwiftUI 更深度集成**

`@Observable` 对象可以直接作为 `@State` 存储，自动管理生命周期。不再有 `@StateObject` vs `@ObservedObject` 的选择困惑和潜在的生命周期 bug。

本项目正是利用了这一点：`ContentView` 中 `@State private var viewModel = AudioViewModel()` 既拥有所有权，又能精细响应变化。

---

### Q7：降噪强度（denoiseStrength）在技术上是如何实现的？

**回答**：降噪强度控制的核心是对 Model 1 输出的频谱掩码进行**线性插值**：

```swift
func adjustMask(_ mask: Float) -> Float {
    if denoiseStrength >= 1.0 { return mask }
    return mask * denoiseStrength + (1.0 - denoiseStrength)
}
```

这个公式的含义：
- `denoiseStrength = 1.0`：完全使用模型预测的掩码（最强降噪）
- `denoiseStrength = 0.5`：掩码 = `0.5 × 模型掩码 + 0.5`，相当于将掩码值向 1.0 靠拢（降噪减半）
- `denoiseStrength = 0.0`：掩码恒为 1.0，即完全不做降噪（原始信号直通）

这种方式的优势是**不改变模型推理本身**，只在掩码应用阶段做插值，计算开销几乎为零。同时因为是从"全降噪"到"不降噪"的连续过渡，用户可以平滑地找到噪声抑制和语音保真之间的最佳平衡点。

需要注意的是，这个调节只作用于 Model 1 的频谱掩码。Model 2（特征域增强）的行为不受 `denoiseStrength` 直接控制，因为 Model 2 处理的是已经过 Model 1 掩码后的信号——Model 1 掩码越弱，送入 Model 2 的信号越接近原始信号，Model 2 的增强效果也相应降低。两级串联形成了自然的联动。

---

### Q8：在处理长音频（如 2 小时讲座）时，有哪些性能和内存方面的考量？

**回答**：以 2 小时、16kHz 的音频为例：

- 总采样点数：`16000 × 7200 = 1.152 亿`
- Float32 数组占用内存：`1.152 亿 × 4 字节 ≈ 440MB`
- 总帧数：`(115,200,000 - 512) / 128 + 1 ≈ 900,000 帧`

**当前存在的问题和优化方向**：

1. **内存占用**：当前 `loadAndResample()` 一次性加载全部音频数据到内存。对于超长音频，应改为**分段流式处理**：每次从 `AVAudioFile` 读取固定长度（如 30 秒）的缓冲区，处理完后写入输出文件，再读取下一段。DTLN 的 LSTM 状态天然支持跨段传递。

2. **Core ML 推理性能**：90 万帧意味着 90 万次 `model.prediction()` 调用。每次调用涉及 Objective-C bridge 和 MLMultiArray 的创建/销毁。优化方式：
   - 使用 `MLPredictionOptions` 设置 `usesCPUOnly = false` 确保 GPU/Neural Engine 被利用
   - 考虑使用 `MLModel.predictions(from:options:)` 的批量预测接口
   - 复用 `MLMultiArray` 缓冲区而非每帧重新创建

3. **进度回调频率**：当前每 50 帧回调一次（约 0.4 秒音频），对于 90 万帧足够流畅。但 `Task { @MainActor in ... }` 本身有一定开销，可改为 `DispatchQueue.main.async` 或降低回调频率。

4. **取消支持**：当前处理过程不支持中途取消。应检查 `Task.isCancelled` 实现协作式取消，允许用户中断长时间的处理任务。

---

### Q9：模型转换过程中遇到了什么挑战？如何解决的？

**回答**：模型转换是本项目技术难度最高的环节，主要遇到了三个挑战：

**挑战一：Keras 2 vs Keras 3 API 断裂**

DTLN 原始代码基于 TensorFlow 2.x + Keras 2 编写。但 TF 2.16+ 将默认 Keras 升级为 Keras 3，后者不允许对 `KerasTensor`（符号张量）直接调用 TF 操作如 `tf.expand_dims()`。

解决方案：将所有 TF 操作包裹在 Keras `Lambda` 层中：

```python
# 错误 (Keras 3)：
frame = tf.expand_dims(time_dat, axis=1)

# 正确：
mag, angle = Lambda(lambda x: fft_layer(x))(time_dat)
```

**挑战二：LSTM While Loop 与 coremltools 不兼容**

coremltools 的 TF2 前端在解析 Keras 3 LSTM 内部生成的 `while_loop` 控制流时崩溃（`AttributeError: 'NoneType' object has no attribute 'input'`）。

解决方案：在推理子模型中设置 `unroll=True`。由于推理时 `seq_len=1`，展开循环等价于单次执行，不增加任何开销：

```python
LSTM(NUM_UNITS, return_sequences=True, return_state=True, unroll=True)
```

**挑战三：权重在完整模型和子模型之间的正确迁移**

DTLN 的完整有状态模型使用 `stateful=True` 的 LSTM（内部管理状态），而推理子模型需要将状态暴露为输入/输出。两者的层结构不同，但权重矩阵的形状和语义是相同的。

解决方案：参照 DTLN 原始代码的 `create_tf_lite_model()` 方法，通过 `full_model.get_weights()` 获取平坦的权重列表，按已知的分割点（`NUM_LAYER * 3 + 2 = 8`）切片分配：

```python
model_1.set_weights(all_weights[:8])   # LSTM×2 + Dense
model_2.set_weights(all_weights[8:])   # Conv1D + LayerNorm + LSTM×2 + Dense + Conv1D
```
