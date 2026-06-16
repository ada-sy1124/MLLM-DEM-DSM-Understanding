### 🏗️ VRT-GRPO 核心工程蓝图

这套工程架构由四大核心模块（Phases）组成，它们在物理层面和代码层面是完全解耦的。

#### Phase 1: 协议定义与预处理层 (Protocol & Pre-processing)

这是最高效的“减负”模块。通过定义人机交互协议和离线提取，把训练时的算力开销降到最低。

* **1.1 钩子语法协议 (Hook Syntax Protocol):**
在 Prompt 中严格规定模型的输出格式。定义三种必备的 `<tag>`：
* `<step_anchor>`: 用于输出参照物坐标。
* `<step_mask>`: 用于输出基于参照物推导出的搜索区域坐标。
* `<target_box>`: 用于输出最终目标坐标。


* **1.2 离线逻辑树提取 (Offline AST Compilation):**
训练前，写一个脚本调用本地或云端的高配大模型（如 Qwen-Max / GPT-4o）。把训练集（如 DIOR-RSVG）里的所有 Prompt（例如“房子左边的车”）**提前解析**成 JSON 格式的标准逻辑树（AST）。
* *作用：* 训练时，**逻辑法官**不需要再调大模型，直接用正则提取当前步骤的文本，与预存的 JSON 树进行极速字符串/正则匹配打分。



#### Phase 2: 裁判引擎矩阵 (The Decoupled Reward Engine)

这是整篇论文的“灵魂”，负责在强化学习过程中提供稠密的步级奖励（Step-wise Reward）。编写三个独立的 Python Class：

* **2.1 `LogicJudge` (逻辑法官):**
* *输入：* 模型生成的单步纯文本。
* *逻辑：* 与 Phase 1.2 提取的 JSON 树对比。是否有遗漏实体？方位词是否反转？
* *输出：* $R_{logic} \in \{-1, +1\}$。


* **2.2 `MathJudge` (代数法官):**
* *输入：* 模型生成的 `<step_mask>` 和紧随其后的子节点坐标。
* *逻辑：* 纯 CPU 数学运算。计算子节点是否 100% 满足 IoC（包裹率）约束，以及中心点是否符合上下左右的几何中轴线约束。
* *输出：* $R_{math} \in \{-1, +1\}$。


* **2.3 `VisionJudge` (视觉法官):**
* *输入：* 原始图像、模型生成的单步 `<anchor>` 坐标、对应的实体名词。
* *逻辑：* 将 Grounding DINO 挂载在主显存的边缘（占用极小）。不裁剪图像，而是将坐标外的区域用 NumPy 矩阵涂灰（Grayscale）。扔给 DINO 验货。
* *输出：* 若置信度 $> threshold$，得 $R_{vision} = +1$；否则为 $-1$。



#### Phase 3: GRPO 训练循环 (The RL Training Loop)

使用 Hugging Face 的 `trl` 库配合 `vLLM` 构建高并发强化学习流。

* **3.1 环境初始化:**
加载基座模型（如 Qwen2.5-VL-3B，兼顾速度与智商），加载裁判矩阵。
* **3.2 多轨迹采样 (Rollout):**
针对同一张图和同一个 Prompt，让模型并行生成 $G=8$ 条包含 `<think>` 过程的轨迹。
* **3.3 步级打分与优势计算 (Step-Reward & Advantage):**
把轨迹拆解成步骤，依次穿过三个法官。算出每条轨迹的累积奖励 $R_{total}$。利用 GRPO 的组内相对优势计算公式：$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$，更新模型策略。

#### Phase 4: 硬件编排与显存管理 (Hardware Orchestration)

这是工程落地的护城河，决定了你的代码能不能跑起来。

* **主 GPU (如 A100 / 4090):** 运行 Policy Model (策略模型) 的权重更新和 vLLM 高速推理生成。
* **副 GPU 或 CPU 内存:** 运行 `LogicJudge`（字符串匹配占用极低）、`MathJudge`（纯 CPU 运算）。
* **常驻显存角落:** 预分配 2GB 显存给 `VisionJudge` (Grounding DINO Swin-T)，使用 FP16 半精度加速验证推理。

---

### 🧭 下一步：编码起点抉择

这个蓝图已经清晰到了函数级别。如果要开始写第一行代码，我建议我们从“验证机制的闭环”开始写起。

在这三个裁判引擎中：
**A. 视觉法官 (Vision Judge):** 跑通 Grounding DINO 的掩码涂灰与验证接口。
**B. 代数法官 (Math Judge):** 写出 IoC 包含关系和空间拓扑的数学判定代码。
**C. 逻辑法官 (Logic Judge):** 写离线大模型抽取 AST 逻辑树的 Prompt 和解析脚本。

你想先从哪一个模块的代码骨架开始构建？
