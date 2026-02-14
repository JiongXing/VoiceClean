#!/usr/bin/env python3
"""
DTLN 预训练模型 -> Core ML 转换脚本

使用方法:
    1. pip install -r requirements.txt
    2. python convert_model.py

输出:
    DTLN_model_1.mlpackage  (分离网络 - 频谱掩码估计)
    DTLN_model_2.mlpackage  (增强网络 - 特征域信号增强)

将生成的两个 .mlpackage 文件拖入 Xcode 项目的 VoiceClean/ 目录即可。
"""

import os
import sys
import subprocess
import shutil
import numpy as np

# ── 常量 ──────────────────────────────────────────────
DTLN_REPO = "https://github.com/breizhn/DTLN.git"
DTLN_DIR = "DTLN"
BLOCK_LEN = 512
BLOCK_SHIFT = 128
NUM_UNITS = 128
NUM_LAYER = 2
ENCODER_SIZE = 256
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def clone_dtln():
    """克隆 DTLN 仓库（如果不存在）"""
    if os.path.isdir(DTLN_DIR):
        print(f"[INFO] DTLN 仓库已存在，跳过克隆。")
        return
    print("[INFO] 克隆 DTLN 仓库...")
    subprocess.run(["git", "clone", DTLN_REPO, DTLN_DIR], check=True)
    print("[INFO] 克隆完成。")


# ── InstantLayerNormalization ──
# 从 DTLN 原始代码移植，用于 Model 2 的特征归一化

import tensorflow as tf
from tensorflow.keras.layers import Layer

class InstantLayerNormalization(Layer):
    """Channel-wise layer normalization (Luo & Mesgarani, 2019)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:], initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=input_shape[-1:], initializer='zeros', trainable=True, name='beta')

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        return (inputs - mean) / std * self.gamma + self.beta


def build_and_load_models():
    """
    重建 DTLN 架构并加载预训练权重。

    严格参照 DTLN_model.py 中的:
      - build_DTLN_model_stateful() : 完整有状态模型
      - create_tf_lite_model() : 拆分为两个推理子模型

    DTLN 双子模型架构:
      - Model 1: 频谱域 — 输入幅度谱 → LSTM×2 → sigmoid 掩码
      - Model 2: 特征域 — 输入时域帧 → Conv1D 编码 → LayerNorm → LSTM×2
                           → sigmoid 掩码 → Multiply → Conv1D 解码
    """
    from tensorflow.keras.layers import (
        Input, Dense, LSTM, Activation, Lambda, Multiply, Conv1D
    )
    from tensorflow.keras.models import Model

    weights_path = os.path.join(DTLN_DIR, "pretrained_model", "model.h5")
    if not os.path.isfile(weights_path):
        print(f"[ERROR] 找不到预训练权重: {weights_path}")
        sys.exit(1)

    print(f"[INFO] 加载预训练权重: {weights_path}")

    # ================================================================
    # Step 1: 构建完整有状态模型（匹配 build_DTLN_model_stateful）
    # Keras 3 要求所有 TF 操作必须包裹在 Lambda 层中
    # ================================================================
    time_dat = Input(batch_shape=(1, BLOCK_LEN))

    # --- FFT (包裹在 Lambda 中以兼容 Keras 3) ---
    def fft_layer(x):
        frame = tf.expand_dims(x, axis=1)
        stft_dat = tf.signal.rfft(frame)
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        return mag, phase

    mag, angle = Lambda(lambda x: fft_layer(x))(time_dat)

    # --- 第一个分离核心 (频谱域) ---
    x = mag
    for _ in range(NUM_LAYER):
        x = LSTM(NUM_UNITS, return_sequences=True, stateful=True)(x)
    mask_1 = Dense(BLOCK_LEN // 2 + 1)(x)
    mask_1 = Activation('sigmoid')(mask_1)

    # 应用掩码 + IFFT
    estimated_mag = Multiply()([mag, mask_1])

    def ifft_layer(inputs):
        est_mag, phase = inputs
        s1_stft = tf.cast(est_mag, tf.complex64) * tf.exp(
            1j * tf.cast(phase, tf.complex64)
        )
        return tf.signal.irfft(s1_stft)

    estimated_frames_1 = Lambda(ifft_layer)([estimated_mag, angle])

    # --- 编码器 ---
    encoded_frames = Conv1D(ENCODER_SIZE, 1, strides=1, use_bias=False)(estimated_frames_1)
    encoded_frames_norm = InstantLayerNormalization()(encoded_frames)

    # --- 第二个分离核心 (特征域) ---
    x2 = encoded_frames_norm
    for _ in range(NUM_LAYER):
        x2 = LSTM(NUM_UNITS, return_sequences=True, stateful=True)(x2)
    mask_2 = Dense(ENCODER_SIZE)(x2)
    mask_2 = Activation('sigmoid')(mask_2)

    # 应用掩码 + 解码器
    estimated = Multiply()([encoded_frames, mask_2])
    decoded_frame = Conv1D(BLOCK_LEN, 1, padding='causal', use_bias=False)(estimated)

    full_model = Model(inputs=time_dat, outputs=decoded_frame)

    # 加载权重
    print("[INFO] 构建完整 DTLN 模型并加载权重...")
    try:
        full_model.load_weights(weights_path)
        all_weights = full_model.get_weights()
        print(f"[INFO] 权重加载成功，共 {len(all_weights)} 个张量。")
    except Exception as e:
        print(f"[ERROR] 权重加载失败: {e}")
        raise

    # ================================================================
    # Step 2: 构建推理子模型 1（频谱域分离网络）
    #
    # 输入: 幅度谱 (1,1,257) + 每层 LSTM 的 h,c 状态
    # 输出: 掩码 (1,1,257) + 更新后的 LSTM 状态
    # ================================================================
    num_elements_first_core = NUM_LAYER * 3 + 2  # 每层 LSTM 3 个权重 + Dense 2 个

    mag_input = Input(batch_shape=(1, 1, BLOCK_LEN // 2 + 1), name="mag_input")

    # LSTM 状态输入 (每层: h 和 c)
    m1_state_inputs = []
    for i in range(NUM_LAYER):
        m1_state_inputs.append(Input(batch_shape=(1, NUM_UNITS), name=f"m1_state_h_{i}"))
        m1_state_inputs.append(Input(batch_shape=(1, NUM_UNITS), name=f"m1_state_c_{i}"))

    m1_x = mag_input
    m1_state_outputs = []
    for i in range(NUM_LAYER):
        h_in = m1_state_inputs[i * 2]
        c_in = m1_state_inputs[i * 2 + 1]
        m1_x, h_out, c_out = LSTM(
            NUM_UNITS, return_sequences=True, return_state=True,
            unroll=True, name=f"m1_lstm_{i}"
        )(m1_x, initial_state=[h_in, c_in])
        # 用命名的 Activation('linear') 包裹状态输出，
        # 防止 coremltools 将 h_out 与 LSTM 序列输出合并优化掉
        h_out = Activation('linear', name=f"m1_out_h_{i}")(h_out)
        c_out = Activation('linear', name=f"m1_out_c_{i}")(c_out)
        m1_state_outputs.extend([h_out, c_out])

    m1_mask = Dense(BLOCK_LEN // 2 + 1, name="m1_dense")(m1_x)
    m1_mask = Activation('sigmoid', name="m1_sigmoid")(m1_mask)

    model_1 = Model(
        inputs=[mag_input] + m1_state_inputs,
        outputs=[m1_mask] + m1_state_outputs,
        name="DTLN_model_1",
    )

    # ================================================================
    # Step 3: 构建推理子模型 2（特征域增强网络）
    #
    # 输入: 时域帧 (1,1,512) + 每层 LSTM 的 h,c 状态
    # 输出: 增强帧 (1,1,512) + 更新后的 LSTM 状态
    # ================================================================
    frame_input = Input(batch_shape=(1, 1, BLOCK_LEN), name="frame_input")

    m2_state_inputs = []
    for i in range(NUM_LAYER):
        m2_state_inputs.append(Input(batch_shape=(1, NUM_UNITS), name=f"m2_state_h_{i}"))
        m2_state_inputs.append(Input(batch_shape=(1, NUM_UNITS), name=f"m2_state_c_{i}"))

    # Conv1D 编码器
    m2_encoded = Conv1D(ENCODER_SIZE, 1, strides=1, use_bias=False, name="m2_conv_encoder")(frame_input)
    # 层归一化
    m2_norm = InstantLayerNormalization(name="m2_layer_norm")(m2_encoded)

    m2_x = m2_norm
    m2_state_outputs = []
    for i in range(NUM_LAYER):
        h_in = m2_state_inputs[i * 2]
        c_in = m2_state_inputs[i * 2 + 1]
        m2_x, h_out, c_out = LSTM(
            NUM_UNITS, return_sequences=True, return_state=True,
            unroll=True, name=f"m2_lstm_{i}"
        )(m2_x, initial_state=[h_in, c_in])
        # 用命名的 Activation('linear') 包裹状态输出，
        # 防止 coremltools 将 h_out 与 LSTM 序列输出合并优化掉
        h_out = Activation('linear', name=f"m2_out_h_{i}")(h_out)
        c_out = Activation('linear', name=f"m2_out_c_{i}")(c_out)
        m2_state_outputs.extend([h_out, c_out])

    m2_mask = Dense(ENCODER_SIZE, name="m2_dense")(m2_x)
    m2_mask = Activation('sigmoid', name="m2_sigmoid")(m2_mask)

    # 掩码 × 编码帧
    m2_estimated = Multiply(name="m2_multiply")([m2_encoded, m2_mask])
    # Conv1D 解码器
    m2_decoded = Conv1D(BLOCK_LEN, 1, padding='causal', use_bias=False, name="m2_conv_decoder")(m2_estimated)

    model_2 = Model(
        inputs=[frame_input] + m2_state_inputs,
        outputs=[m2_decoded] + m2_state_outputs,
        name="DTLN_model_2",
    )

    # ================================================================
    # Step 4: 将权重从完整模型分配到子模型
    # 参照 DTLN_model.create_tf_lite_model 的权重分片方法
    # ================================================================
    print(f"[INFO] 分配权重: 前 {num_elements_first_core} 个 → Model 1, 其余 → Model 2")

    model_1.set_weights(all_weights[:num_elements_first_core])
    model_2.set_weights(all_weights[num_elements_first_core:])

    print(f"[INFO] Model 1: {len(model_1.get_weights())} 个权重张量")
    print(f"[INFO] Model 2: {len(model_2.get_weights())} 个权重张量")

    return model_1, model_2


def rename_model_outputs(mlmodel, state_prefix, main_output_name,
                         main_shape_last_dim, num_layer, num_units):
    """
    通过 shape 识别主输出(mask/frame)，通过扰动测试确定状态输出映射，然后重命名。
    """
    import coremltools as ct

    spec = mlmodel.get_spec()
    output_names = [o.name for o in spec.description.output]
    print(f"  原始输出名: {output_names}")

    # ── Step 1: 准备零输入 ──
    feed = {}
    for inp in spec.description.input:
        raw_shape = inp.type.multiArrayType.shape
        shape = tuple(int(d) for d in raw_shape)
        feed[inp.name] = np.zeros(shape, dtype=np.float32)

    baseline = mlmodel.predict(feed)

    # ── Step 2: 通过 shape 识别主输出 (mask / frame) ──
    main_key = None
    state_output_keys = []
    for k, v in baseline.items():
        if len(v.shape) == 3 and v.shape[-1] == main_shape_last_dim:
            main_key = k
        else:
            state_output_keys.append(k)

    assert main_key is not None, f"找不到主输出 (shape 末维={main_shape_last_dim})"
    print(f"  主输出: {main_key} -> {main_output_name}")

    # ── Step 3: 通过扰动测试确定状态映射 ──
    state_input_names = []
    for i in range(num_layer):
        state_input_names.append(f"{state_prefix}_state_h_{i}")
        state_input_names.append(f"{state_prefix}_state_c_{i}")

    # 对每个状态输入进行扰动，找出受影响最大的输出
    output_to_desired = {}
    claimed_outputs = set()

    for state_input in state_input_names:
        perturbed = dict(feed)
        perturbed[state_input] = np.full((1, num_units), 5.0, dtype=np.float32)
        result = mlmodel.predict(perturbed)

        # 计算每个状态输出与 baseline 的差异
        diffs = {}
        for out_key in state_output_keys:
            if out_key in claimed_outputs:
                continue
            diff = np.abs(result[out_key] - baseline[out_key]).sum()
            diffs[out_key] = diff

        if diffs:
            best_match = max(diffs, key=diffs.get)
            desired_name = state_input.replace("_state_", "_out_")
            output_to_desired[best_match] = desired_name
            claimed_outputs.add(best_match)
            print(f"  {best_match} -> {desired_name} (diff={diffs[best_match]:.4f})")

    # ── Step 4: 执行重命名 ──
    ct.utils.rename_feature(spec, main_key, main_output_name)
    for old_name, new_name in output_to_desired.items():
        ct.utils.rename_feature(spec, old_name, new_name)

    renamed = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)
    final_names = [o.name for o in renamed.get_spec().description.output]
    print(f"  重命名后: {final_names}")
    return renamed


def convert_to_coreml(model_1, model_2):
    """将两个子模型转换为 Core ML 格式"""
    import coremltools as ct

    # ── Model 1 转换 ──
    print("\n[INFO] 转换 Model 1 (分离网络) 到 Core ML...")

    m1_inputs = [
        ct.TensorType(name="mag_input", shape=(1, 1, BLOCK_LEN // 2 + 1)),
    ]
    for i in range(NUM_LAYER):
        m1_inputs.append(ct.TensorType(name=f"m1_state_h_{i}", shape=(1, NUM_UNITS)))
        m1_inputs.append(ct.TensorType(name=f"m1_state_c_{i}", shape=(1, NUM_UNITS)))

    try:
        mlmodel_1 = ct.convert(
            model_1,
            source="tensorflow",
            convert_to="mlprogram",
            inputs=m1_inputs,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as e:
        print(f"[ERROR] Model 1 转换失败: {e}")
        raise

    print("[INFO] 重命名 Model 1 输出...")
    mlmodel_1 = rename_model_outputs(
        mlmodel_1, state_prefix="m1",
        main_output_name="mask_output",
        main_shape_last_dim=BLOCK_LEN // 2 + 1,
        num_layer=NUM_LAYER, num_units=NUM_UNITS
    )

    mlmodel_1.author = "VoiceClean"
    mlmodel_1.short_description = "DTLN 分离网络 - 频谱掩码估计"
    mlmodel_1.version = "1.0"

    out_path_1 = os.path.join(OUTPUT_DIR, "DTLN_model_1.mlpackage")
    if os.path.exists(out_path_1):
        shutil.rmtree(out_path_1)
    mlmodel_1.save(out_path_1)
    print(f"[OK] Model 1 已保存: {out_path_1}")

    # ── Model 2 转换 ──
    print("\n[INFO] 转换 Model 2 (增强网络) 到 Core ML...")

    m2_inputs = [
        ct.TensorType(name="frame_input", shape=(1, 1, BLOCK_LEN)),
    ]
    for i in range(NUM_LAYER):
        m2_inputs.append(ct.TensorType(name=f"m2_state_h_{i}", shape=(1, NUM_UNITS)))
        m2_inputs.append(ct.TensorType(name=f"m2_state_c_{i}", shape=(1, NUM_UNITS)))

    try:
        mlmodel_2 = ct.convert(
            model_2,
            source="tensorflow",
            convert_to="mlprogram",
            inputs=m2_inputs,
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS13,
        )
    except Exception as e:
        print(f"[ERROR] Model 2 转换失败: {e}")
        raise

    print("[INFO] 重命名 Model 2 输出...")
    mlmodel_2 = rename_model_outputs(
        mlmodel_2, state_prefix="m2",
        main_output_name="frame_output",
        main_shape_last_dim=BLOCK_LEN,
        num_layer=NUM_LAYER, num_units=NUM_UNITS
    )

    mlmodel_2.author = "VoiceClean"
    mlmodel_2.short_description = "DTLN 增强网络 - 特征域信号增强"
    mlmodel_2.version = "1.0"

    out_path_2 = os.path.join(OUTPUT_DIR, "DTLN_model_2.mlpackage")
    if os.path.exists(out_path_2):
        shutil.rmtree(out_path_2)
    mlmodel_2.save(out_path_2)
    print(f"[OK] Model 2 已保存: {out_path_2}")

    return out_path_1, out_path_2


def verify_models(path_1, path_2):
    """验证模型可以加载并推理"""
    import coremltools as ct

    print("\n[INFO] 验证模型...")

    m1 = ct.models.MLModel(path_1)
    m2 = ct.models.MLModel(path_2)

    # 测试 Model 1
    m1_feed = {"mag_input": np.random.randn(1, 1, BLOCK_LEN // 2 + 1).astype(np.float32)}
    for i in range(NUM_LAYER):
        m1_feed[f"m1_state_h_{i}"] = np.zeros((1, NUM_UNITS), dtype=np.float32)
        m1_feed[f"m1_state_c_{i}"] = np.zeros((1, NUM_UNITS), dtype=np.float32)

    out1 = m1.predict(m1_feed)
    print(f"[OK] Model 1 推理成功，输出 keys: {sorted(out1.keys())}")
    for k, v in sorted(out1.items()):
        print(f"     {k}: shape={v.shape}")

    # 测试 Model 2
    m2_feed = {"frame_input": np.random.randn(1, 1, BLOCK_LEN).astype(np.float32)}
    for i in range(NUM_LAYER):
        m2_feed[f"m2_state_h_{i}"] = np.zeros((1, NUM_UNITS), dtype=np.float32)
        m2_feed[f"m2_state_c_{i}"] = np.zeros((1, NUM_UNITS), dtype=np.float32)

    out2 = m2.predict(m2_feed)
    print(f"[OK] Model 2 推理成功，输出 keys: {sorted(out2.keys())}")
    for k, v in sorted(out2.items()):
        print(f"     {k}: shape={v.shape}")

    print("\n✅ 模型转换和验证全部完成！")
    print(f"   {path_1}")
    print(f"   {path_2}")
    print("\n请将上述两个 .mlpackage 文件拖入 Xcode 项目的 VoiceClean/ 目录。")


def main():
    print("=" * 60)
    print("  DTLN -> Core ML 模型转换工具")
    print("=" * 60)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    clone_dtln()
    model_1, model_2 = build_and_load_models()
    path_1, path_2 = convert_to_coreml(model_1, model_2)
    verify_models(path_1, path_2)


if __name__ == "__main__":
    main()
