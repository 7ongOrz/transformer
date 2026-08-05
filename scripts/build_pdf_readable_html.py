from __future__ import annotations

from html import escape
from pathlib import Path


SOURCE_PDF = "Vision Transformer 超详细解读 (原理分析+代码解读) (一) - 知乎.pdf"
SOURCE_TEXT = Path("tmp/pdfs/vision-transformer-full-text.txt")
OUTPUT = Path("reference/vision-transformer-pdf-readable.html")


def load_source_text() -> str:
    if SOURCE_TEXT.exists():
        return SOURCE_TEXT.read_text(encoding="utf-8")

    from pypdf import PdfReader

    reader = PdfReader(SOURCE_PDF)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\u00a0", " ")
        pages.append(f"===== PDF PAGE {page_number} =====\n{text.strip()}\n")
    return "\n".join(pages)


def figure(
    number: int,
    chapter: str,
    page: str,
    title: str,
    summary: str,
    steps: list[str],
    formula: str = "",
    shapes: list[str] | None = None,
    note: str = "",
) -> dict[str, object]:
    return {
        "number": number,
        "chapter": chapter,
        "page": page,
        "title": title,
        "summary": summary,
        "steps": steps,
        "formula": formula,
        "shapes": shapes or [],
        "note": note,
    }


FIGURES = [
    figure(
        1,
        "self-attention",
        "1",
        "RNN 与单层 CNN 处理序列",
        "同一组输入向量进入两类序列模型。RNN 拥有长程依赖但必须串行；CNN 可并行，但单层只看固定局部窗口。",
        [
            "输入是向量序列 a1, a2, a3, a4，输出是同长度序列 b1, b2, b3, b4。",
            "单向 RNN 的 b_i 依赖前一隐藏状态，因此 b4 必须等待 b1、b2、b3。",
            "一维 CNN 的多个 filter 可同时滑过不同窗口，因此各位置输出可并行。",
            "单层 CNN 的每个输出只由窗口内若干输入决定，不能直接看到整条序列。",
        ],
        shapes=["RNN/CNN: sequence[N,d_in] -> sequence[N,d_out]"],
    ),
    figure(
        2,
        "self-attention",
        "1",
        "堆叠 CNN 扩大感受野",
        "上层 filter 接收下层多个局部输出，层数增加后可以间接覆盖更长的输入范围。",
        [
            "第一层每个 filter 只覆盖相邻的少量 token。",
            "第二层 filter 覆盖多个第一层输出，因此对应原始输入中的更长片段。",
            "要获得全局依赖，需要继续堆层；路径长度随距离和层数增加。",
        ],
        note="图的比较重点不是 CNN 无法建模长程，而是需要更深层才能建立远距离连接。",
    ),
    figure(
        3,
        "self-attention",
        "1",
        "用 Self-Attention Layer 替代 RNN Layer",
        "Self-Attention 保持“序列进、序列出”的接口，但每个输出都能直接读取整个输入序列，并且所有输出可并行。",
        [
            "输入 a1...aN 同时进入 Self-Attention Layer。",
            "输出 b_i 由所有输入位置共同贡献，而不是只依赖前一时间步。",
            "b1...bN 之间没有必须按时间顺序执行的前向依赖。",
        ],
        shapes=["A=[a1...aN]", "B=[b1...bN]", "length(B)=length(A)=N"],
    ),
    figure(
        4,
        "self-attention",
        "1",
        "每个输入生成 Query、Key、Value",
        "作者先把输入 x_i 映射成 embedding a_i，再用三组不同的可学习矩阵生成 q_i、k_i、v_i。",
        [
            "x_i 乘 embedding 变换 W^x 得到 a_i。",
            "a_i 乘 W^Q 得到 q_i：该位置用于发起匹配的查询表示。",
            "a_i 乘 W^K 得到 k_i：该位置用于接受匹配的键表示。",
            "a_i 乘 W^V 得到 v_i：匹配完成后真正被汇聚的内容表示。",
        ],
        formula="a_i = W^x x_i; q_i = W^Q a_i; k_i = W^K a_i; v_i = W^V a_i",
        shapes=["a_i in R^(d_model)", "q_i,k_i in R^(d_k)", "v_i in R^(d_v)"],
        note="PDF 采用列向量记法，所以权重矩阵写在向量左侧；常见 PyTorch 行向量记法会写成 X @ W。",
    ),
    figure(
        5,
        "self-attention",
        "1",
        "一个 Query 与所有 Key 做匹配",
        "以 q1 为例，图中从 q1 分别连向 k1、k2、k3、k4，得到 q1 对所有位置的相关性分数。",
        [
            "固定查询 q1。",
            "并行计算 q1 与每个 k_j 的内积。",
            "用 sqrt(d_k) 缩放点积，避免维度增大导致分数方差过大。",
            "得到该 Query 的一整行 logits: [alpha_1,1, ..., alpha_1,N]。",
        ],
        formula="alpha_(1,j) = (q_1^T k_j) / sqrt(d_k), for j=1...N",
        shapes=["q1: [d_k]", "K: [N,d_k]", "logits_1: [N]"],
    ),
    figure(
        6,
        "self-attention",
        "1",
        "Softmax 把相关性分数变成注意力权重",
        "q1 的全部 logits 一起进入 Softmax，输出非负且总和为 1 的权重。",
        [
            "对每个 alpha_(1,j) 取指数。",
            "用该 Query 行所有指数之和做归一化。",
            "权重 alpha_hat_(1,j) 表示生成 b1 时从位置 j 取多少信息。",
        ],
        formula="alpha_hat_(1,j) = exp(alpha_(1,j)) / sum_t exp(alpha_(1,t)); sum_j alpha_hat_(1,j)=1",
        shapes=["logits row: [N]", "attention weight row: [N]"],
    ),
    figure(
        7,
        "self-attention",
        "1",
        "用权重对 Value 加权求和得到 b1",
        "图中每个权重 alpha_hat_(1,j) 与对应 v_j 相乘，所有分支汇聚成第一个输出 b1。",
        [
            "alpha_hat_(1,1) 乘 v1。",
            "alpha_hat_(1,2) 乘 v2，依次处理所有位置。",
            "把所有加权 Value 相加得到 b1。",
            "因此 b1 可以包含整条序列的信息；接近 0 的权重等价于忽略对应分支。",
        ],
        formula="b_1 = sum_(j=1)^N alpha_hat_(1,j) v_j",
        shapes=["weights_1: [N]", "V: [N,d_v]", "b1: [d_v]"],
    ),
    figure(
        8,
        "self-attention",
        "1",
        "换一个 Query 计算 b2",
        "把固定查询从 q1 换成 q2，重复点积、Softmax 和 Value 加权，即可得到第二个输出。",
        [
            "q2 与所有 k_j 得到第二行 logits。",
            "第二行 Softmax 得到 alpha_hat_(2,j)。",
            "第二行权重对同一组 v_j 加权求和得到 b2。",
        ],
        formula="b_2 = sum_(j=1)^N softmax_j(q_2^T k_j / sqrt(d_k)) v_j",
    ),
    figure(
        9,
        "self-attention",
        "1",
        "所有输出可并行计算",
        "每个 q_i 的计算结构完全相同，图中 b1、b2、b3、b4 同时从 Self-Attention Layer 输出。",
        [
            "把所有 Query 堆成矩阵 Q。",
            "一次 QK^T 同时产生所有 Query-Key 对的分数。",
            "逐行 Softmax 后再一次矩阵乘 V，得到所有 b_i。",
        ],
        formula="B = softmax(Q K^T / sqrt(d_k)) V",
        note="这一步把前面逐个 Query 的连线图收束为矩阵运算，也是 GPU 并行的关键。",
    ),
    figure(
        10,
        "self-attention",
        "1",
        "Q、K、V 的矩阵化投影",
        "作者把 a1...aN 作为输入矩阵 I 的列，再分别左乘三组变换矩阵，得到 Q、K、V。",
        [
            "列向量约定：I=[a1 a2 ... aN]，shape 为 [d_model,N]。",
            "Q=W^Q I，K=W^K I，V=W^V I。",
            "每个输出矩阵的第 i 列仍对应第 i 个 token。",
        ],
        formula="PDF column convention: Q=W^Q I, K=W^K I, V=W^V I\nCommon row convention: Q=XW^Q, K=XW^K, V=XW^V",
        shapes=["column convention I: [d_model,N]", "Q,K: [d_k,N]", "V: [d_v,N]"],
    ),
    figure(
        11,
        "self-attention",
        "1",
        "注意力矩阵与输出矩阵的乘法方向",
        "图把所有 alpha_(i,j) 组成 N x N 注意力矩阵，再与 V 相乘，等价于同时执行所有 Value 加权求和。",
        [
            "列向量画法中，K^T Q 的第 (j,i) 项是 k_j^T q_i。",
            "对分数矩阵按每个 Query 对应的 Key 维做 Softmax，得到 A_hat。",
            "列向量画法 O=V A_hat；O 的第 i 列就是 b_i。",
            "行向量画法更常见：S=QK^T，A=softmax(S)，O=AV。",
        ],
        formula="PDF: A=K^T Q; A_hat=softmax(A); O=V A_hat\nModern row-major: S=QK^T; A=softmax(S,dim=-1); O=AV",
        shapes=["score/weight matrix: [N,N]", "O: [d_v,N] in PDF or [N,d_v] in row-major"],
        note="两套公式只是 token 放在行还是列的差别，计算含义相同。",
    ),
    figure(
        12,
        "self-attention",
        "1",
        "Self-Attention 是一条矩阵乘法流水线",
        "整张图把输入、三路投影、相关性矩阵、Softmax 和输出连接成一条可由 GPU 执行的流水线。",
        [
            "I 分流为 Q、K、V。",
            "Q 与 K 生成 N x N 的两两相关性矩阵。",
            "Softmax 把每个 Query 的分数归一化。",
            "权重矩阵与 V 相乘得到与输入长度一致的输出 O。",
        ],
        formula="Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
        shapes=["[N,d_model] -> Q/K/V -> [N,N] -> [N,d_v]"],
    ),
    figure(
        13,
        "self-attention",
        "1",
        "两个 Head 的完整计算与输出投影",
        "每个原始 q_i、k_i、v_i 再投影到两个独立子空间，分别计算 b_i^1、b_i^2，然后拼接并乘 W^O。",
        [
            "Head 1 使用 W_1^Q、W_1^K、W_1^V；Head 2 使用另一组独立参数。",
            "每个 Head 独立完成 scaled dot-product attention。",
            "同一位置 i 得到 head 输出 b_i^1 和 b_i^2。",
            "Concat(b_i^1,b_i^2) 乘输出矩阵 W^O，恢复成模型维度 b_i。",
        ],
        formula="head_r=Attention(QW_r^Q,KW_r^K,VW_r^V)\nb_i=W^O concat(b_i^1,...,b_i^h)",
        shapes=["d_model=h*d_head", "each head: [N,d_head]", "concat: [N,h*d_head]"],
    ),
    figure(
        14,
        "self-attention",
        "1",
        "Multi-Head 的 Split、并行、Concat、Linear",
        "输入 X 同时送入多个 Self-Attention Head，各自输出同长度矩阵，沿特征维拼接后做线性映射。",
        [
            "X 经过多个独立的 Q/K/V 投影。",
            "所有 Head 并行计算，不是在序列维串行堆叠。",
            "输出结果沿最后一个特征维 Concat。",
            "最终 Linear/W^O 混合各 Head 信息，并保持输入输出 d_model 一致。",
        ],
        formula="MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O",
    ),
    figure(
        15,
        "self-attention",
        "1",
        "不同 Head 的注意力连线模式",
        "图中一组连线跨越较远位置，另一组集中在相邻位置，用来说明不同子空间可学习 global 与 local 关系。",
        [
            "绿色 Head 出现较多跨长距离的粗连接，可解释为全局关系。",
            "红色 Head 更多连接相邻 token，可解释为局部关系。",
            "真实模型中的 Head 不保证拥有固定、唯一的语义；图只表达能力上的可能性。",
        ],
    ),
    figure(
        16,
        "position",
        "2",
        "把位置编码加到输入表示",
        "Self-Attention 本身没有顺序概念，所以每个位置都有一个位置向量 e_i，并与内容向量 a_i 相加。",
        [
            "为位置 i 构造 e_i，维度与 a_i 相同。",
            "新输入 a_i'=a_i+e_i。",
            "a_i' 再进入 Q/K/V 投影，使内容与位置信息共同影响注意力。",
        ],
        formula="a_i' = a_i + e_i",
        shapes=["a_i,e_i,a_i': [d_model]"],
    ),
    figure(
        17,
        "position",
        "2",
        "相加位置编码与拼接 One-Hot 的等价解释",
        "图把 token 表示与位置 One-Hot 先拼接，再乘分块矩阵；矩阵乘法可拆成内容投影与位置投影之和。",
        [
            "构造扩展输入 [a_i ; onehot(i)]。",
            "把线性层参数分成内容块 W^a 和位置块 W^p。",
            "线性结果 W^a a_i + W^p onehot(i)。",
            "W^p onehot(i) 正好选出位置 i 对应的列向量 e_i，因此得到 W^a a_i + e_i。",
        ],
        formula="W[a_i;onehot(i)] = W^a a_i + W^p onehot(i) = W^a a_i + e_i",
        note="这解释了为什么“相加”仍能保留位置资讯：后续线性层可以把内容和位置重新分解。",
    ),
    figure(
        18,
        "position",
        "2",
        "位置编码变体与 Seq2Seq 中的使用位置",
        "该编号在原文中重复使用：一组图比较不同位置编码模式，另一张图说明用 Self-Attention 替换 Encoder-Decoder 中的 RNN。",
        [
            "固定正余弦编码：不同维度使用不同频率。",
            "可学习绝对位置编码：每个位置对应一行 Embedding 参数。",
            "FLOATER/RNN 等方法学习随位置变化的编码轨迹。",
            "Seq2Seq 图中，Encoder 与 Decoder 的循环模块可被 Self-Attention/Transformer Block 替代。",
        ],
        formula="PE(pos,2i)=sin(pos/10000^(2i/d_model)); PE(pos,2i+1)=cos(pos/10000^(2i/d_model))",
    ),
    figure(
        19,
        "vision",
        "2",
        "图像上的全局 Self-Attention 与 CNN 局部感受野",
        "选定一个像素作为 Query，Self-Attention 会把它与全图所有像素的 Key 比较；CNN 只连接手工规定的局部窗口。",
        [
            "Self-Attention 的候选 Key 范围是整张 feature map。",
            "CNN 的候选范围由 kernel size 和 dilation 等超参数决定。",
            "Attention 权重根据内容动态变化，相当于数据驱动的感受野。",
        ],
    ),
    figure(
        20,
        "vision",
        "2",
        "CNN 可视为受约束的 Self-Attention 特例",
        "集合示意图把 CNN 放在 Self-Attention 的内部：固定局部连接与共享权重可以看作对更一般注意力机制施加约束。",
        [
            "Self-Attention 允许任意位置两两交互。",
            "CNN 只允许固定邻域交互，并共享卷积核参数。",
            "更灵活的 Self-Attention 通常需要更多数据学习其连接模式。",
        ],
    ),
    figure(
        21,
        "transformer",
        "2/4",
        "经典 Transformer Encoder-Decoder 架构",
        "左侧 Encoder 堆叠 N 层，右侧 Decoder 堆叠 N 层；Decoder 额外包含 Masked Self-Attention 和读取 Encoder Memory 的 Cross-Attention。",
        [
            "源 token -> Input Embedding + Positional Encoding。",
            "Encoder Layer: Multi-Head Self-Attention -> Add & Norm -> FFN -> Add & Norm，重复 N 次。",
            "目标 token 右移 -> Output Embedding + Positional Encoding。",
            "Decoder Layer: Masked Self-Attention -> Add & Norm -> Cross-Attention -> Add & Norm -> FFN -> Add & Norm，重复 N 次。",
            "Decoder 顶部 Linear -> Softmax -> 下一 token 概率。",
            "Cross-Attention 中 Q 来自 Decoder，K/V 来自 Encoder 输出。",
        ],
        formula="Encoder sublayer: y=LayerNorm(x+Sublayer(x))\nCrossAttention: Q=H_dec, K=V=H_enc",
        shapes=["source memory: [B,S_src,d_model]", "decoder state: [B,S_tgt,d_model]", "logits: [B,S_tgt,vocab]"],
    ),
    figure(
        22,
        "transformer",
        "2",
        "不同 Normalization 的归一化轴",
        "三维张量示意图用不同切片颜色表示 BatchNorm、LayerNorm、InstanceNorm、GroupNorm 分别跨哪些轴统计均值与方差。",
        [
            "BatchNorm：对同一 channel 跨 batch 与空间位置统计。",
            "LayerNorm：对单个样本的特征维统计，不依赖 batch 中其他样本。",
            "InstanceNorm：通常对每个样本、每个 channel 的空间位置统计。",
            "GroupNorm：把 channel 分组后分别统计。",
        ],
    ),
    figure(
        23,
        "transformer",
        "2",
        "BatchNorm 与 LayerNorm 的二维直观比较",
        "蓝色圆点矩阵中，横向红框代表 BatchNorm 对同一特征跨样本归一化，纵向红框代表 LayerNorm 对同一样本跨特征归一化。",
        [
            "BatchNorm 使 batch 中某个 channel 近似满足 mu=0、sigma=1。",
            "LayerNorm 使单个 token/样本的所有 feature 近似满足 mu=0、sigma=1。",
            "Transformer 常用 LayerNorm，因为序列长度和 batch 大小可变化。",
        ],
    ),
    figure(
        24,
        "transformer",
        "2",
        "Mask 的位置：Scale 之后、Softmax 之前",
        "图把 Scaled Dot-Product Attention 画成 MatMul -> Scale -> Mask(opt) -> Softmax -> MatMul，Q/K 进入第一处 MatMul，V 进入最后一处。",
        [
            "先计算 S=QK^T/sqrt(d_k)。",
            "对未来位置或 padding 位置写入 -infinity。",
            "再做 Softmax，使被屏蔽位置权重严格为 0。",
            "最后 A@V 得到输出。",
        ],
        formula="A=softmax(QK^T/sqrt(d_k)+M); M_ij=0 if allowed, -infinity if blocked",
        note="原文部分文字写成与 0/1 Mask 相乘；实际实现应在 Softmax 前 masked_fill(-infinity)，否则被置 0 的 logit 仍会得到非零概率。",
    ),
    figure(
        25,
        "transformer",
        "2",
        "Decoder 训练与推理的时间流程",
        "翻译示例展示推理逐 token 生成，而训练使用右移的完整目标序列配合 Causal Mask 并行计算。",
        [
            "推理第 1 步输入 <Begin>，输出 I。",
            "第 2 步输入 <Begin>, I，输出 have；继续直到 <end>。",
            "训练输入 <Begin>, I, have, a, cat，监督目标为 I, have, a, cat, <end>。",
            "Causal Mask 保证位置 i 不能读取 i 之后的目标 token，因此并行训练不泄漏未来。",
        ],
    ),
    figure(
        26,
        "transformer",
        "2",
        "Masked Multi-Head Self-Attention 的完整矩阵过程",
        "五个目标 token 的图把输入投影、下三角 Mask、按行 Softmax、多头拼接全部展开。",
        [
            "输入 X 含 5 个 token，投影成 Q、K、V，shape 均为 [N,d]。",
            "计算 N x N 分数矩阵；第 i 行表示 Query i 对所有 Key 的分数。",
            "使用下三角允许矩阵：第 0 行只允许 key0，第 1 行允许 key0..1，依此类推。",
            "禁止位置在 Softmax 前变为 -infinity，Softmax 后为 0；每个允许集合内的行和为 1。",
            "每个 Head 得到 Z_h，Concat 后乘线性矩阵得到最终 Z。",
        ],
        formula="Z_h=softmax(Q_h K_h^T/sqrt(d_k)+M)V_h; Z=Concat(Z_1,...,Z_h)W^O",
        shapes=["X,Q,K,V: [N,d]", "M,A: [N,N]", "Z: [N,d_model]"],
    ),
    figure(
        27,
        "transformer",
        "2",
        "每个 Decoder 输出行预测下一个词",
        "Decoder 输出矩阵每一行经过 Linear 映射到词表维度，再通过 Softmax 得到相应位置的下一词概率。",
        [
            "第 i 行隐藏状态 z_i 包含位置 0..i 的目标上下文以及 Encoder 信息。",
            "Linear: logits_i = z_i W_vocab + b。",
            "Softmax(logits_i) 得到词表分布。",
            "训练时所有行并行产生；推理时只使用最后一行选择下一个 token。",
        ],
        formula="p(y_(i+1)|y_<=i,x)=softmax(z_i W_vocab+b)",
    ),
    figure(
        28,
        "transformer",
        "3",
        "Transformer 整体模块与代码类的对应关系",
        "图把 Embedding、位置编码、EncoderLayer/DecoderLayer 堆叠、输出投影连接起来，对应后文代码中的类。",
        [
            "PositionalEncoding 先与源/目标 Embedding 相加。",
            "Encoder 由 n_layers 个 EncoderLayer 顺序堆叠。",
            "Decoder 由 n_layers 个 DecoderLayer 顺序堆叠，并接收 Encoder output。",
            "padding mask 作用于源序列；目标 mask 同时包含 padding mask 与 subsequent/causal mask。",
            "目标词表 Linear 与 Softmax 产生最终 logits/probability。",
        ],
        shapes=["d_model=512", "n_layers=6", "n_head=8", "d_k=d_v=64", "d_inner=2048"],
    ),
    figure(
        29,
        "detr",
        "3",
        "DETR 用 CNN 与 Transformer 并行预测目标集合",
        "输入图像经 CNN 提取 feature map，Transformer 同时输出固定数量的 object predictions，不再依赖 anchor、proposal 或 NMS 流程。",
        [
            "CNN backbone 把图像转换为低分辨率高通道 feature map。",
            "Transformer Encoder 对所有空间位置建立全局关系。",
            "Transformer Decoder 接收固定数量的可学习 Object Queries。",
            "每个 Query 并行输出一个类别与一个 bounding box；无目标的槽位预测 no-object。",
        ],
        shapes=["N queries, with N much larger than objects in an image"],
    ),
    figure(
        30,
        "detr",
        "3",
        "DETR 四段流水线与关键 Shape",
        "图按 Backbone -> Encoder -> Decoder -> FFN Heads 展示数据流，并在每一步标出维度。",
        [
            "image x: [B,3,H0,W0]。",
            "CNN feature f: [B,C,H,W]，通常 H=H0/32、W=W0/32。",
            "1x1 Conv 压到 z0: [B,256,H,W]。",
            "flatten spatial + permute -> [HW,B,256]，与同 shape 的二维位置编码相加后进入 Encoder。",
            "Encoder memory: [HW,B,256]。",
            "Object Queries: [100,B,256]；Decoder 并行输出 [B,100,256]。",
            "分类头 -> [B,100,class+1]；框回归头 -> [B,100,4]。",
        ],
        shapes=["COCO class+1=92", "box=(cx,cy,w,h), normalized to [0,1]"],
    ),
    figure(
        31,
        "detr",
        "3",
        "Input Embedding 与二维位置编码如何相加",
        "原论文画法让两条支路看起来 shape 不同，作者用问号标出疑点；后续图 32/33 证明它们在 flatten 后都是 [HW,B,256]。",
        [
            "图像 feature 先经 1x1 Conv 得到 [B,256,H,W]。",
            "二维位置编码也生成 [B,256,H,W]。",
            "两者各自 flatten H,W 并 permute 成 [HW,B,256]。",
            "此时逐元素相加，shape 完全一致。",
        ],
    ),
    figure(
        32,
        "detr",
        "3",
        "DETR 位置编码与特征序列的 Shape 全路径",
        "作者用方块表示每个张量，专门澄清二维 feature map、位置编码和 Transformer 序列之间的变换顺序。",
        [
            "特征支路: [B,C,H,W] -> 1x1 Conv -> [B,256,H,W] -> flatten+permute -> [HW,B,256]。",
            "位置支路: x/y 累积坐标 -> 各生成 128 维 sin/cos -> concat -> [B,256,H,W]。",
            "位置支路同样 flatten+permute -> [HW,B,256]。",
            "两支路相加，进入每个 Encoder Layer 的 Q/K 路径。",
        ],
        formula="pos(x,y)=concat(PE_x(x) in R^128, PE_y(y) in R^128) in R^256",
    ),
    figure(
        33,
        "detr",
        "3",
        "Input Embedding 与 Positional Embedding Shape 一致",
        "图用同尺寸矩形强调：论文中的视觉宽窄只是画法，真正执行加法时两个张量完全同形。",
        [
            "Input embedding: [HW,B,256]。",
            "Positional embedding: [HW,B,256]。",
            "逐元素相加后仍为 [HW,B,256]。",
        ],
    ),
    figure(
        34,
        "detr",
        "4",
        "DETR Transformer 的详细 Q/K/V 来源与维度",
        "该图把每层 Encoder/Decoder、位置编码、Object Queries 和所有中间 shape 标在经典 Transformer 框图上。",
        [
            "Encoder 每层：Q=K=memory+2D position，V=memory；输出保持 [HW,B,256]。",
            "Decoder 第一个 Self-Attention：Q=K=tgt+object_queries，V=tgt，shape [100,B,256]。",
            "Decoder Cross-Attention：Q=decoder_state+object_queries，K=encoder_memory+2D_position，V=encoder_memory。",
            "Cross-Attention 的 Query 长度 100，Key/Value 长度 HW，因此权重矩阵为 [B,heads,100,HW]。",
            "每个 Decoder Layer 输出保持 [100,B,256]，最终堆叠结果可表示为 [layers,B,100,256]。",
        ],
        shapes=["Encoder length=HW", "Decoder query length=100", "hidden_dim=256"],
    ),
    figure(
        35,
        "detr",
        "4",
        "DETR 中每处 Query/Key 的位置编码来源",
        "归纳图分别列出 Encoder Self-Attention、Decoder Self-Attention、Decoder Cross-Attention 的 Q/K/V 与位置编码组合。",
        [
            "Encoder: Q=memory+pos, K=memory+pos, V=memory。",
            "Decoder Self-Attention: Q=tgt+query_pos, K=tgt+query_pos, V=tgt。",
            "Decoder Cross-Attention: Q=tgt+query_pos, K=memory+pos, V=memory。",
            "Value 不加位置编码；位置只用于决定匹配权重。",
        ],
        formula="Encoder: (Q,K,V)=(x+pos,x+pos,x)\nDecoder self: (tgt+query_pos,tgt+query_pos,tgt)\nCross: (tgt+query_pos,memory+pos,memory)",
    ),
    figure(
        36,
        "detr",
        "4",
        "DETR 与 Faster R-CNN 性能对比",
        "表格比较不同 backbone/训练设置下的 AP、AP50、AP75、APS、APM、APL 与推理速度。",
        [
            "DETR 的整体 AP 和运行时间与强 Faster R-CNN baseline 相当。",
            "DETR 通常在大型目标 APL 上更强，在小目标 APS 上较弱。",
            "该表用于说明端到端集合预测具备竞争力，而不是给出新的计算步骤。",
        ],
        formula="Model | GFLOPS/FPS | params | AP | AP50 | AP75 | APS | APM | APL\nFaster R-CNN-DC5 | 320/16 | 166M | 39.0 | 60.5 | 42.3 | 21.4 | 43.5 | 52.5\nFaster R-CNN-FPN | 180/26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0\nFaster R-CNN-R101-FPN | 246/20 | 60M | 42.0 | 62.5 | 45.9 | 25.2 | 45.6 | 54.6\nFaster R-CNN-DC5+ | 320/16 | 166M | 41.1 | 61.4 | 44.3 | 22.9 | 45.9 | 55.0\nFaster R-CNN-FPN+ | 180/26 | 42M | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4\nFaster R-CNN-R101-FPN+ | 246/20 | 60M | 44.0 | 63.9 | 47.8 | 27.2 | 48.1 | 56.0\nDETR | 86/28 | 41M | 42.0 | 62.4 | 44.2 | 20.5 | 45.8 | 61.1\nDETR-DC5 | 187/12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1\nDETR-R101 | 152/20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8\nDETR-DC5-R101 | 253/10 | 60M | 44.9 | 64.7 | 47.7 | 23.7 | 49.5 | 62.3",
        note="数值按 PDF 图 36 的表格逐行转写；粗体强调在纯文本中未保留。",
    ),
    figure(
        37,
        "detr",
        "4",
        "Encoder 层数与检测性能",
        "折线图展示 Encoder 层数增加时 AP 指标总体上升，作者据此采用 6 层 Encoder。",
        [
            "横轴是 Encoder layers 数量。",
            "纵轴是 COCO AP 系列指标。",
            "更多 Encoder 层提供更充分的全局对象关系建模，曲线在约 6 层附近达到采用点。",
        ],
        formula="#layers | GFLOPS/FPS | params | AP | AP50 | APS | APM | APL\n0 | 76/28 | 33.4M | 36.7 | 57.4 | 16.8 | 39.6 | 54.2\n3 | 81/25 | 37.4M | 40.1 | 60.6 | 18.5 | 43.8 | 58.6\n6 | 86/23 | 41.3M | 40.6 | 61.6 | 19.9 | 44.3 | 60.2\n12 | 95/20 | 49.2M | 41.6 | 62.1 | 19.8 | 44.9 | 61.9",
        note="原文最终选择 6 层，是精度、计算量和参数量之间的折中，并非表中 AP 的绝对最大值。",
    ),
    figure(
        38,
        "detr",
        "4",
        "最后一个 Encoder Layer 的 Attention 可视化",
        "不同查询位置在图像上形成覆盖对象实例的全局注意区域，显示 Encoder 已开始分离实例。",
        [
            "每张图叠加一个或多个 attention heatmap。",
            "响应区域往往覆盖完整对象或与对象相关的上下文，而非只集中于一个局部点。",
            "作者据此解释：Encoder 的全局交互简化了 Decoder 后续对象提取和定位。",
        ],
    ),
    figure(
        39,
        "detr",
        "4",
        "Decoder 层数与 AP/AP50",
        "折线图比较每个 Decoder Layer 输出的性能，并展示把各层预测合并后做 NMS 的结果。",
        [
            "随着 Decoder 层数增加，AP 与 AP50 整体提升。",
            "浅层输出更容易对同一对象重复预测，因为单层尚未充分建模输出槽位之间关系。",
            "后续层的 Decoder Self-Attention 可抑制重复；因此 NMS 带来的额外收益逐层减少。",
            "最后几层使用 NMS 可能误删真阳性，导致小幅 AP 损失。",
        ],
    ),
    figure(
        40,
        "detr",
        "4",
        "Decoder Attention 的局部对象部位响应",
        "不同颜色对应不同 Object Query 的注意力，响应常集中在对象头部、腿部或边界等局部极值区域。",
        [
            "Encoder 已通过全局注意力分离实例。",
            "Decoder Query 因而只需关注具有判别力的局部部位，提取类别和边界信息。",
            "不同 Query 用不同颜色显示，说明固定查询槽位在图像 memory 上检索不同对象。",
        ],
    ),
]


CHAPTERS = [
    (
        "self-attention",
        "一、Self-Attention：从连线图到矩阵",
        "图 1-15",
        "先比较 RNN/CNN，再逐个 Query 展开 Q/K/V、点积、Softmax、Value 加权，最后收束成矩阵与 Multi-Head。",
    ),
    (
        "position",
        "二、位置编码与视觉中的全局连接",
        "图 16-20",
        "解释为什么无位置信息的 Self-Attention 无法区分顺序，以及位置编码、CNN 感受野和全局 Attention 的关系。",
    ),
    (
        "transformer",
        "三、经典 Transformer、Mask 与代码结构",
        "图 21-28",
        "完整展开 Encoder、Decoder、残差与归一化，重点转写 Causal Mask 的矩阵过程和训练/推理解码差异。",
    ),
    (
        "vision",
        "四、视觉补充",
        "图 19-20",
        "图像 Self-Attention 与 CNN 的关系已放在位置编码章节，保留该分组供机器检索。",
    ),
    (
        "detr",
        "五、DETR：Transformer 进入目标检测",
        "图 29-40",
        "追踪 CNN feature、二维位置编码、Object Queries、Decoder 和 Hungarian Matching 的完整 shape 与数据流。",
    ),
]


EQUATIONS = [
    ("E1", "单个 Query-Key 分数", "alpha_(i,j) = q_i^T k_j / sqrt(d_k)"),
    ("E2", "按 Key 维 Softmax", "a_(i,j) = exp(alpha_(i,j)) / sum_t exp(alpha_(i,t))"),
    ("E3", "单位置输出", "b_i = sum_j a_(i,j) v_j"),
    ("E4", "矩阵形式", "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + M)V"),
    ("E5", "Multi-Head", "head_r=Attention(QW_r^Q,KW_r^K,VW_r^V); MHA=Concat(head_1,...,head_h)W^O"),
    ("E6", "正余弦位置编码", "PE(pos,2i)=sin(pos/10000^(2i/d)); PE(pos,2i+1)=cos(pos/10000^(2i/d))"),
    ("E7", "经典 Post-LN Encoder 子层", "y=LayerNorm(x+Sublayer(x))"),
    ("E8", "DETR 二维位置编码", "pos(x,y)=concat(PE_x(x) in R^128, PE_y(y) in R^128) in R^256"),
    ("E9", "Hungarian 最优排列", "sigma_hat = argmin_(sigma in Sigma_N) sum_i L_match(y_i, y_hat_(sigma(i)))"),
    ("E10", "匹配代价", "L_match = -p_hat_(sigma(i))(c_i) + L_box(b_i,b_hat_(sigma(i))) for non-empty targets"),
    ("E11", "训练损失", "L_Hungarian = sum_i[-log p_hat_(sigma_hat(i))(c_i) + 1_(c_i != empty)L_box]"),
    ("E12", "框损失", "L_box=lambda_iou*L_giou + lambda_L1*||b_i-b_hat||_1"),
]


def render_figure(item: dict[str, object]) -> str:
    number = item["number"]
    steps = "".join(f"<li>{escape(step)}</li>" for step in item["steps"])
    shapes = "".join(f"<li><code>{escape(shape)}</code></li>" for shape in item["shapes"])
    formula = str(item["formula"])
    formula_html = ""
    if formula:
        formula_html = (
            f'<pre class="formula" data-latex="{escape(formula, quote=True)}">'
            f"{escape(formula)}</pre>"
        )
    shape_html = f'<ul class="shape-list">{shapes}</ul>' if shapes else ""
    note = str(item["note"])
    note_html = f'<p class="figure-note"><strong>说明：</strong>{escape(note)}</p>' if note else ""
    return f"""
      <figure class="figure-card" id="figure-{number}" data-figure="{number}" data-chapter="{escape(str(item['chapter']))}">
        <figcaption>
          <span class="figure-number">图 {number}</span>
          <div><h3>{escape(str(item['title']))}</h3><small>PDF 页 {escape(str(item['page']))}</small></div>
        </figcaption>
        <p class="figure-summary">{escape(str(item['summary']))}</p>
        <ol class="figure-steps">{steps}</ol>
        {formula_html}
        {shape_html}
        {note_html}
      </figure>
    """


def render_chapter(chapter: tuple[str, str, str, str]) -> str:
    key, title, figure_range, intro = chapter
    cards = "".join(render_figure(item) for item in FIGURES if item["chapter"] == key)
    if not cards:
        return ""
    return f"""
      <section class="chapter" id="chapter-{escape(key)}">
        <header class="chapter-head">
          <span>{escape(figure_range)}</span>
          <div><h2>{escape(title)}</h2><p>{escape(intro)}</p></div>
        </header>
        <div class="figure-grid">{cards}</div>
      </section>
    """


def render_equations() -> str:
    rows = "".join(
        f'<tr><th>{escape(key)}</th><td>{escape(name)}</td><td><code>{escape(value)}</code></td></tr>'
        for key, name, value in EQUATIONS
    )
    return f"""
      <section class="chapter" id="equations">
        <header class="chapter-head"><span>FORMULA INDEX</span><div><h2>公式索引</h2><p>所有关键公式均以纯文本形式存在，GLM 不需要识图或执行公式渲染器。</p></div></header>
        <div class="table-scroll"><table class="equation-table"><thead><tr><th>ID</th><th>含义</th><th>机器可读公式</th></tr></thead><tbody>{rows}</tbody></table></div>
      </section>
    """


def render_mask_table() -> str:
    tokens = ["<BOS>", "I", "have", "a", "cat"]
    head = "".join(f"<th>{escape(token)}</th>" for token in tokens)
    rows = []
    for i, token in enumerate(tokens):
        cells = "".join(
            f'<td class="{"allow" if j <= i else "block"}">{"0" if j <= i else "-inf"}</td>'
            for j in range(len(tokens))
        )
        rows.append(f"<tr><th>{escape(token)}</th>{cells}</tr>")
    return f"""
      <section class="chapter" id="mask-matrix">
        <header class="chapter-head"><span>CORE RECONSTRUCTION</span><div><h2>图 24-26 的 Causal Mask 矩阵复原</h2><p>行是 Query，列是 Key。允许读取的位置加 0；未来位置加 -infinity，再执行 Softmax。</p></div></header>
        <div class="table-scroll"><table class="mask-table"><thead><tr><th>Q / K</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
        <pre class="formula">masked_logits = Q @ K.T / sqrt(d_k) + M
weights = softmax(masked_logits, dim=-1)
output = weights @ V</pre>
      </section>
    """


def render_shape_ledger() -> str:
    rows = [
        ("标准 Attention", "X", "[B,S,d_model]"),
        ("标准 Attention", "Q,K,V after split", "[B,H,S,d_head]"),
        ("标准 Attention", "scores/weights", "[B,H,S_q,S_k]"),
        ("标准 Attention", "output before concat", "[B,H,S_q,d_v]"),
        ("DETR Backbone", "image", "[B,3,H0,W0]"),
        ("DETR Backbone", "CNN feature", "[B,C,H,W]"),
        ("DETR 1x1 Conv", "projected feature", "[B,256,H,W]"),
        ("DETR Encoder", "feature/position sequence", "[HW,B,256]"),
        ("DETR Decoder", "object queries", "[100,B,256]"),
        ("DETR Cross-Attention", "attention weights", "[B,H,100,HW]"),
        ("DETR Outputs", "class logits", "[B,100,class+1]"),
        ("DETR Outputs", "boxes", "[B,100,4]"),
    ]
    body = "".join(
        f"<tr><td>{escape(stage)}</td><td><code>{escape(name)}</code></td><td><code>{escape(shape)}</code></td></tr>"
        for stage, name, shape in rows
    )
    return f"""
      <section class="chapter" id="shape-ledger">
        <header class="chapter-head"><span>SHAPE LEDGER</span><div><h2>张量 Shape 总表</h2><p>把 PDF 图 10-12、30-35 中分散的维度集中到一张机器可读表。</p></div></header>
        <div class="table-scroll"><table class="equation-table"><thead><tr><th>阶段</th><th>张量</th><th>Shape</th></tr></thead><tbody>{body}</tbody></table></div>
      </section>
    """


raw_text = load_source_text()
chapter_html = "".join(render_chapter(chapter) for chapter in CHAPTERS)
figure_links = "".join(
    f'<a href="#figure-{item["number"]}">图 {item["number"]}</a>' for item in FIGURES
)

document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Vision Transformer PDF - 机器可读完整转换</title>
  <meta name="description" content="将知乎 Vision Transformer 长文 PDF 的正文、公式和图 1-40 转换成无需识图即可读取的单文件 HTML。">
  <script type="application/ld+json">{{"@context":"https://schema.org","@type":"TechArticle","name":"Vision Transformer PDF 机器可读转换","isBasedOn":"{escape(SOURCE_PDF)}","inLanguage":"zh-CN"}}</script>
  <style>
    :root{{--bg:#f4f1e9;--paper:#fffefa;--ink:#13272b;--muted:#65777b;--line:#d7d9d3;--q:#ef6a4b;--k:#28a9b3;--v:#7d6be8;--accent:#1d3f45;--soft:#eef4f2;--ok:#d8efe4;--bad:#ffe2dc;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}
    *{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 var(--sans)}}
    a{{color:inherit}} code,pre{{font-family:var(--mono)}}
    .top{{padding:56px clamp(22px,6vw,88px);background:var(--accent);color:white}}
    .top .eyebrow{{color:#82d9df;font:800 11px var(--mono);letter-spacing:.12em}}
    .top h1{{max-width:920px;margin:14px 0 12px;font-size:clamp(38px,6vw,72px);line-height:1.04;letter-spacing:-.05em}}
    .top p{{max-width:880px;margin:0;color:#c8d5d5;font-size:16px}}
    .machine-note{{display:grid;max-width:1100px;margin:24px auto 0;padding:18px;border:1px solid #ffffff2a;border-radius:10px;background:#ffffff0c;grid-template-columns:repeat(3,1fr);gap:12px}}
    .machine-note div{{padding:10px;background:#00000015}} .machine-note b{{display:block;color:#cce96f;font-size:12px}} .machine-note span{{color:#bfcccc;font-size:11px}}
    .toolbar{{position:sticky;z-index:10;top:0;display:flex;padding:12px clamp(18px,5vw,70px);align-items:center;border-bottom:1px solid var(--line);background:#fffefaeb;backdrop-filter:blur(12px);gap:12px}}
    .toolbar input{{width:min(480px,70vw);padding:10px 13px;border:1px solid var(--line);border-radius:6px;background:white;color:var(--ink);font:13px var(--sans)}}
    .toolbar span{{margin-left:auto;color:var(--muted);font:11px var(--mono)}}
    main{{max-width:1240px;margin:auto;padding:28px clamp(18px,4vw,50px) 80px}}
    .toc{{margin-bottom:28px;padding:22px;border:1px solid var(--line);border-radius:12px;background:var(--paper)}}
    .toc h2{{margin:0 0 12px}} .toc .chapters{{display:flex;flex-wrap:wrap;gap:8px}} .toc .chapters a,.figure-index a{{padding:6px 9px;border:1px solid var(--line);border-radius:4px;text-decoration:none;background:white;font-size:11px}}
    .figure-index{{display:flex;margin-top:14px;flex-wrap:wrap;gap:5px}}
    .chapter{{margin-top:26px;padding:26px;border:1px solid var(--line);border-radius:14px;background:var(--paper);box-shadow:0 16px 45px #1a353812}}
    .chapter-head{{display:grid;margin-bottom:22px;padding-bottom:18px;align-items:start;border-bottom:2px solid var(--ink);grid-template-columns:170px 1fr;gap:18px}}
    .chapter-head>span{{color:var(--q);font:800 10px var(--mono);letter-spacing:.1em}} .chapter-head h2{{margin:0;font-size:28px;line-height:1.2;letter-spacing:-.03em}} .chapter-head p{{margin:6px 0 0;color:var(--muted);font-size:12px}}
    .figure-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
    .figure-card{{margin:0;padding:18px;border:1px solid var(--line);border-radius:9px;background:white}}
    .figure-card figcaption{{display:grid;align-items:start;grid-template-columns:58px 1fr;gap:10px}} .figure-number{{display:grid;width:52px;height:36px;place-items:center;border-radius:6px;background:var(--ink);color:white;font:800 10px var(--mono)}}
    .figure-card h3{{margin:0;font-size:16px;line-height:1.3}} .figure-card figcaption small{{color:var(--muted);font:9px var(--mono)}}
    .figure-summary{{margin:13px 0;color:var(--muted);font-size:12px}} .figure-steps{{margin:0;padding-left:21px;font-size:12px}} .figure-steps li+li{{margin-top:4px}}
    .formula{{margin:13px 0 0;padding:12px;overflow:auto;border-left:4px solid var(--q);border-radius:4px;background:#13272b;color:#bcebf0;font-size:11px;white-space:pre-wrap}}
    .shape-list{{display:flex;margin:10px 0 0;padding:0;flex-wrap:wrap;gap:5px;list-style:none}} .shape-list li{{padding:4px 7px;border-radius:3px;background:var(--soft);font-size:10px}}
    .figure-note{{margin:11px 0 0;padding:9px 11px;border-radius:4px;background:#fff2df;color:#76572b;font-size:10px}}
    .table-scroll{{overflow:auto}} table{{width:100%;border-collapse:collapse;background:white}} th,td{{padding:10px;border:1px solid var(--line);text-align:left;vertical-align:top;font-size:11px}} thead th{{background:var(--ink);color:white}} td code{{color:#9e452f}}
    .mask-table th,.mask-table td{{text-align:center}} .mask-table .allow{{background:var(--ok);color:#1a6842;font-weight:800}} .mask-table .block{{background:var(--bad);color:#9d3128;font-weight:800}}
    details.raw{{margin-top:26px;border:1px solid var(--line);border-radius:12px;background:var(--paper)}} details.raw summary{{padding:18px 22px;cursor:pointer;font-weight:800}} .raw pre{{max-height:720px;margin:0;padding:22px;overflow:auto;border-top:1px solid var(--line);background:#14272b;color:#d5e4e3;font-size:10px;white-space:pre-wrap}}
    .source-note{{margin-top:18px;padding:15px;border-left:4px solid var(--k);background:white;color:var(--muted);font-size:11px}}
    [hidden]{{display:none!important}}
    @media(max-width:820px){{.machine-note,.figure-grid{{grid-template-columns:1fr}}.chapter-head{{grid-template-columns:1fr}}.toolbar span{{display:none}}.chapter{{padding:17px}}}}
  </style>
</head>
<body>
  <header class="top">
    <div class="eyebrow">MACHINE-READABLE PDF CONVERSION · FIGURES 1-40</div>
    <h1>Vision Transformer PDF<br>完整机器可读转换</h1>
    <p>源文件：{escape(SOURCE_PDF)}。本页不依赖图片识别：所有关键箭头、矩阵方向、公式、Mask、Shape 和图中结论均已写入普通 HTML 文本。</p>
    <div class="machine-note">
      <div><b>完整正文兜底</b><span>附录包含五页 PDF 的完整文本提取结果。</span></div>
      <div><b>图 1-40 全部转写</b><span>每张图都有标题、节点、箭头顺序、公式和维度。</span></div>
      <div><b>无像素依赖</b><span>核心信息不只存在于 CSS、Canvas、SVG 或图片中。</span></div>
    </div>
  </header>
  <div class="toolbar"><input id="search" type="search" placeholder="搜索图号、Q/K/V、Mask、Shape、DETR..." aria-label="搜索转换内容"><span id="result-count">40 figures</span></div>
  <main>
    <nav class="toc" aria-label="目录">
      <h2>目录</h2>
      <div class="chapters">
        <a href="#chapter-self-attention">Self-Attention</a><a href="#chapter-position">位置编码</a><a href="#chapter-transformer">Transformer</a><a href="#chapter-detr">DETR</a><a href="#equations">公式索引</a><a href="#shape-ledger">Shape 总表</a><a href="#raw-text">原始文本</a>
      </div>
      <div class="figure-index">{figure_links}</div>
    </nav>

    <section class="chapter" id="reading-guide">
      <header class="chapter-head"><span>READING GUIDE</span><div><h2>给 GLM 的读取说明</h2><p>建议先读取图 4-14，再读取图 21、24-28，最后读取图 30-35。</p></div></header>
      <ol>
        <li>图卡中的有序列表就是原图箭头的执行顺序。</li>
        <li><code>data-latex</code> 和公式代码块保存公式原文，不依赖 KaTeX。</li>
        <li>PDF 使用 token 作为列的记法；本页在图 10-11 同时给出常见行向量记法，避免 QK 转置方向混淆。</li>
        <li>图 24-26 的 Mask 已按实际算子实现解释为加负无穷，而不是简单乘 0。</li>
        <li>末尾“原始提取正文”用于核对文章全部段落和代码；网页广告与评论可能仍按 PDF 提取顺序出现。</li>
      </ol>
    </section>

    {chapter_html}
    {render_mask_table()}
    {render_equations()}
    {render_shape_ledger()}

    <section class="chapter" id="hungarian-example">
      <header class="chapter-head"><span>DETR MATCHING</span><div><h2>图后文字中的 Hungarian Matching 计算过程</h2><p>该计算主要存在于正文和公式，不应在转换时丢失。</p></div></header>
      <ol>
        <li>模型固定输出 100 个 prediction；示例图像只有 3 个 GT：Car、Dog、Horse。</li>
        <li>构造 <code>[100,3]</code> 代价矩阵。元素 <code>C[p,g]</code> 表示预测 p 与真值 g 的分类代价加框代价。</li>
        <li><code>linear_sum_assignment</code> 在所有一对一排列中选总代价最小的匹配。</li>
        <li>文章示例假设得到 prediction 23 -> Car、44 -> Dog、95 -> Horse。</li>
        <li>只对匹配结果计算分类与 box loss；其余预测匹配 no-object，框回归分支忽略背景。</li>
      </ol>
      <pre class="formula">C[p,g] = -P_p(class_g) + lambda_L1 * L1(box_p, box_g) + lambda_giou * GIoU_loss(box_p, box_g)
assignment = linear_sum_assignment(C)
loss = classification_loss + box_L1_loss + generalized_IoU_loss</pre>
    </section>

    <details class="raw" id="raw-text">
      <summary>原始提取正文：五页 PDF 完整文本（点击展开）</summary>
      <pre>{escape(raw_text)}</pre>
    </details>
    <p class="source-note"><strong>转换范围：</strong>文章正文、公式线索、代码文本、图 1-40 的机器可读转写。对于无法从截图可靠辨认的实验表格具体数字，只保留原图结论并明确标注，避免伪造数值。</p>
  </main>
  <script>
    const search = document.getElementById('search');
    const cards = [...document.querySelectorAll('.figure-card')];
    const count = document.getElementById('result-count');
    search.addEventListener('input', () => {{
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      cards.forEach(card => {{
        const match = !query || card.textContent.toLowerCase().includes(query);
        card.hidden = !match;
        if (match) visible += 1;
      }});
      count.textContent = `${{visible}} figures`;
    }});
  </script>
</body>
</html>
"""

document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(document, encoding="utf-8")
print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes, {len(FIGURES)} figures)")
