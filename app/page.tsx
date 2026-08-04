"use client";

import { useEffect, useMemo, useState } from "react";

const matrixA = [
  [1, 2, 0],
  [0, 1, 3],
];

const matrixB = [
  [2, 1],
  [1, 0],
  [0, 2],
];

const matrixC = matrixA.map((row) =>
  matrixB[0].map((_, columnIndex) =>
    row.reduce(
      (sum, value, sharedIndex) =>
        sum + value * matrixB[sharedIndex][columnIndex],
      0,
    ),
  ),
);

const tokens = ["小猫", "追着", "自己的", "尾巴"];
const queries = [
  [1.2, 0.1],
  [0.6, 1.0],
  [1.1, 0.5],
  [0.4, 1.2],
];
const keys = [
  [1.1, 0.2],
  [0.5, 1.1],
  [1.0, 0.4],
  [0.3, 1.3],
];
const values = [
  [1.0, 0.2],
  [0.2, 0.9],
  [0.8, 0.5],
  [0.1, 1.0],
];

const heads = [
  {
    name: "Head 01",
    title: "指代关系",
    color: "coral",
    note: "更关注“自己的”指向谁。不同头并非被人工指定，而是在训练中可能自发形成不同偏好。",
    matrix: [
      [0.58, 0.12, 0.22, 0.08],
      [0.18, 0.16, 0.12, 0.54],
      [0.66, 0.08, 0.2, 0.06],
      [0.14, 0.46, 0.09, 0.31],
    ],
  },
  {
    name: "Head 02",
    title: "动作对象",
    color: "cyan",
    note: "把动作“追着”和对象“尾巴”联系起来，像是在补全“谁做了什么”。",
    matrix: [
      [0.36, 0.38, 0.12, 0.14],
      [0.12, 0.18, 0.08, 0.62],
      [0.3, 0.22, 0.34, 0.14],
      [0.08, 0.58, 0.08, 0.26],
    ],
  },
  {
    name: "Head 03",
    title: "邻近信息",
    color: "lime",
    note: "偏向当前位置附近的词，保留局部短语与词序线索。",
    matrix: [
      [0.58, 0.3, 0.08, 0.04],
      [0.22, 0.46, 0.24, 0.08],
      [0.08, 0.25, 0.46, 0.21],
      [0.04, 0.08, 0.31, 0.57],
    ],
  },
  {
    name: "Head 04",
    title: "全局语义",
    color: "violet",
    note: "权重更平均，让每个词都带上一点整句语境。",
    matrix: [
      [0.28, 0.24, 0.25, 0.23],
      [0.23, 0.29, 0.22, 0.26],
      [0.27, 0.2, 0.3, 0.23],
      [0.22, 0.27, 0.21, 0.3],
    ],
  },
];

const codeSnippets = {
  attention: {
    label: "01 · 单头 Attention",
    title: "公式落地：只有 8 行核心计算",
    note: "matmul 负责相似度与加权汇总；scale 防止 Softmax 过饱和；mask 必须在 Softmax 前加入。",
    code: `def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights`,
  },
  multihead: {
    label: "02 · 多头 Attention",
    title: "拆头、并行、拼接、再投影",
    note: "reshape 并没有复制数据；transpose 把 head 维移到前面，使每个头独立计算 [S, S] 注意力矩阵。",
    code: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, seq, self.h, self.d_head) \
                    .transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        x, weights = scaled_dot_product_attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, seq, self.h * self.d_head)
        return self.out(x), weights`,
  },
  block: {
    label: "03 · Transformer Block",
    title: "Attention 不是整座 Transformer",
    note: "一个经典块还包含残差连接、LayerNorm 和逐位置前馈网络。现代模型常把 LayerNorm 放到子层之前。",
    code: `class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, heads=8, d_ff=2048):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, weights = self.attn(self.norm1(x), mask)
        x = x + attn_out          # residual connection
        x = x + self.ffn(self.norm2(x))
        return x, weights`,
  },
  test: {
    label: "04 · 算子测试",
    title: "用参考实现锁住正确性",
    note: "小尺寸、固定随机种子、float64 参考值最容易定位问题；性能测试与正确性测试要分开。",
    code: `def test_attention_matches_reference():
    torch.manual_seed(7)
    q = torch.randn(2, 4, 5, 8, dtype=torch.float64)
    k = torch.randn(2, 4, 5, 8, dtype=torch.float64)
    v = torch.randn(2, 4, 5, 8, dtype=torch.float64)

    actual, weights = attention_op(q, k, v)
    expected, _ = scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights[..., 0]),
    )`,
  },
};

const quiz = [
  {
    question: "QKᵀ 得到的矩阵，两个序列维度分别代表什么？",
    answers: ["词和特征", "查询位置和被查询位置", "批次和注意力头"],
    correct: 1,
    explain: "每一行属于一个 Query 位置，每一列属于一个 Key 位置，所以形状是 [S_query, S_key]。",
  },
  {
    question: "为什么要除以 √dₖ？",
    answers: ["让矩阵变小以节省显存", "让大维度点积不过大，避免 Softmax 过饱和", "保证权重严格相等"],
    correct: 1,
    explain: "维度越大，点积的幅度通常越大；缩放可让 Softmax 仍处在有梯度的区间。",
  },
  {
    question: "因果 Mask 应该放在哪一步？",
    answers: ["Softmax 之前", "乘 V 之后", "输出投影之后"],
    correct: 0,
    explain: "先把未来位置的分数变成 −∞，Softmax 后这些位置的概率才会成为 0。",
  },
];

const testChecklist = [
  ["形状契约", "覆盖 2D / 3D / 4D、batch、head、不同 S_q/S_k", "P0"],
  ["参考值比对", "与朴素 float64 实现逐元素比较", "P0"],
  ["Mask 语义", "causal、padding、bool/float mask；被遮挡权重为 0", "P0"],
  ["数值稳定", "极大输入、全 mask 行、FP16/BF16、无 NaN/Inf", "P0"],
  ["梯度正确", "gradcheck + Q/K/V/Wq/Wk/Wv 梯度形状", "P1"],
  ["性质验证", "Softmax 每行和为 1；输出是 V 的加权组合", "P1"],
  ["布局与连续性", "transpose 后非连续 Tensor、stride、不同 batch_first", "P1"],
  ["性能回归", "预热后统计延迟、吞吐、峰值显存；与相同 shape 基线比较", "P2"],
];

function softmax(input: number[]) {
  const finite = input.filter(Number.isFinite);
  const max = finite.length ? Math.max(...finite) : 0;
  const exps = input.map((value) =>
    Number.isFinite(value) ? Math.exp(value - max) : 0,
  );
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function formatNumber(value: number, digits = 2) {
  if (!Number.isFinite(value)) return "−∞";
  return value.toFixed(digits);
}

function SectionTitle({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-heading">
      <div className="eyebrow"><span />{eyebrow}</div>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  );
}

function Matrix({
  values: matrix,
  label,
  highlightRow,
  highlightColumn,
  onCellClick,
}: {
  values: number[][];
  label: string;
  highlightRow?: number;
  highlightColumn?: number;
  onCellClick?: (row: number, column: number) => void;
}) {
  return (
    <div className="matrix-wrap">
      <div className="matrix-label">{label}</div>
      <div className="matrix-bracket">
        {matrix.map((row, rowIndex) => (
          <div className="matrix-row" key={`${label}-${rowIndex}`}>
            {row.map((value, columnIndex) => {
              const active =
                rowIndex === highlightRow || columnIndex === highlightColumn;
              const exact =
                rowIndex === highlightRow && columnIndex === highlightColumn;
              return (
                <button
                  type="button"
                  className={`matrix-cell ${active ? "is-active" : ""} ${exact ? "is-exact" : ""}`}
                  onClick={() => onCellClick?.(rowIndex, columnIndex)}
                  tabIndex={onCellClick ? 0 : -1}
                  key={`${rowIndex}-${columnIndex}`}
                  aria-label={`${label} 第 ${rowIndex + 1} 行第 ${columnIndex + 1} 列，值 ${value}`}
                >
                  {value}
                </button>
              );
            })}
          </div>
        ))}
      </div>
      <div className="matrix-shape">
        {matrix.length} × {matrix[0].length}
      </div>
    </div>
  );
}

export default function Home() {
  const [selectedCell, setSelectedCell] = useState<[number, number]>([0, 0]);
  const [selectedToken, setSelectedToken] = useState(2);
  const [causal, setCausal] = useState(false);
  const [selectedHead, setSelectedHead] = useState(0);
  const [architecture, setArchitecture] = useState<"original" | "encoder" | "decoder">("original");
  const [activeCode, setActiveCode] = useState<keyof typeof codeSnippets>("attention");
  const [copied, setCopied] = useState(false);
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [presenterMode, setPresenterMode] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const updateProgress = () => {
      const total = document.documentElement.scrollHeight - window.innerHeight;
      setScrollProgress(total > 0 ? (window.scrollY / total) * 100 : 0);
    };
    updateProgress();
    window.addEventListener("scroll", updateProgress, { passive: true });
    return () => window.removeEventListener("scroll", updateProgress);
  }, []);

  const attention = useMemo(() => {
    const q = queries[selectedToken];
    const rawScores = keys.map((key) =>
      q.reduce((sum, value, index) => sum + value * key[index], 0),
    );
    const scaledScores = rawScores.map((score, index) =>
      causal && index > selectedToken ? Number.NEGATIVE_INFINITY : score / Math.sqrt(q.length),
    );
    const weights = softmax(scaledScores);
    const output = values[0].map((_, dimension) =>
      weights.reduce(
        (sum, weight, index) => sum + weight * values[index][dimension],
        0,
      ),
    );
    return { q, rawScores, scaledScores, weights, output };
  }, [selectedToken, causal]);

  const [rowIndex, columnIndex] = selectedCell;
  const selectedRow = matrixA[rowIndex];
  const selectedColumn = matrixB.map((row) => row[columnIndex]);
  const code = codeSnippets[activeCode];

  async function copyCode() {
    await navigator.clipboard.writeText(code.code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <main className={presenterMode ? "presenter-mode" : ""}>
      <div className="scroll-progress" style={{ width: `${scrollProgress}%` }} />

      <nav className="topbar" aria-label="页面导航">
        <a className="brand" href="#top" aria-label="Attention Lab 首页">
          <span className="brand-mark">A</span>
          <span>
            <strong>ATTENTION LAB</strong>
            <small>从 0 到算子测试</small>
          </span>
        </a>
        <div className="nav-links">
          <a href="#matrix">矩阵热身</a>
          <a href="#attention">Attention</a>
          <a href="#transformer">Transformer</a>
          <a href="#testing">算子测试</a>
        </div>
        <button
          type="button"
          className={`presenter-toggle ${presenterMode ? "is-on" : ""}`}
          onClick={() => setPresenterMode((value) => !value)}
          aria-pressed={presenterMode}
        >
          <span className="toggle-dot" />
          讲解模式
        </button>
      </nav>

      <header className="hero" id="top">
        <div className="hero-copy">
          <div className="hero-kicker">TRANSFORMER FOUNDATION · 互动讲义 01</div>
          <h1>
            一次讲透
            <span>Attention</span>
          </h1>
          <p className="hero-lead">
            它不是“模型在思考”的魔法，而是一套清晰的<strong>查找、打分、加权汇总</strong>流程。
            从一个乘法开始，我们亲手把它算出来。
          </p>
          <div className="hero-actions">
            <a className="primary-button" href="#matrix">从第 1 步开始 <span>↓</span></a>
            <a className="text-button" href="#testing">直接看测试清单 →</a>
          </div>
          <div className="hero-meta">
            <div><strong>≈ 45</strong><span>分钟完整讲解</span></div>
            <div><strong>7</strong><span>个渐进章节</span></div>
            <div><strong>4</strong><span>个互动实验</span></div>
          </div>
        </div>

        <div className="hero-visual" aria-label="Attention 直觉示意图">
          <div className="visual-topline">
            <span>LIVE INTUITION</span>
            <span className="live-dot">演示中</span>
          </div>
          <div className="sentence-stage">
            <span>小猫</span><span>追着</span><span className="query-token">自己的<i>Q</i></span><span>尾巴</span>
          </div>
          <div className="attention-readout">
            <div className="readout-row hot"><span>小猫 <small>K₁</small></span><div><i style={{ width: "86%" }} /></div><b>0.86</b></div>
            <div className="readout-row"><span>追着 <small>K₂</small></span><div><i style={{ width: "28%" }} /></div><b>0.28</b></div>
            <div className="readout-row"><span>尾巴 <small>K₄</small></span><div><i style={{ width: "18%" }} /></div><b>0.18</b></div>
          </div>
          <div className="visual-result">
            <span>CONTEXT</span>
            “自己的”融合了更多<strong>小猫</strong>的信息
          </div>
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
        </div>
      </header>

      <div className="chapter-strip" aria-label="学习路径">
        {[
          ["01", "矩阵语言"], ["02", "Q · K · V"], ["03", "完整计算"],
          ["04", "多头机制"], ["05", "Transformer"], ["06", "经典代码"], ["07", "算子测试"],
        ].map(([number, title]) => (
          <div key={number}><span>{number}</span><strong>{title}</strong></div>
        ))}
      </div>

      <section className="section motivation-section" id="motivation">
        <div className="motivation-heading">
          <div className="eyebrow"><span />00 · WHY ATTENTION</div>
          <h2>先回答最重要的问题：<br />为什么要发明 Attention？</h2>
          <p>处理序列时，我们既想看得远，又想算得快。过去的方案通常只能先满足其中一个。</p>
        </div>
        <div className="model-race">
          <article className="race-card rnn-card">
            <div className="race-title"><span>RNN</span><strong>接力阅读</strong></div>
            <div className="rnn-visual">
              {["我", "喜欢", "研究", "注意力"].map((token, index) => (
                <div key={token}><b>{token}</b><i>{index < 3 ? "→" : "✓"}</i></div>
              ))}
            </div>
            <h3>看得到很远，但必须按顺序等</h3>
            <p>第 4 步依赖第 3 步的状态，像接力棒一样传递，难以把所有位置同时算完。</p>
            <div className="race-meters"><span><i style={{ width: "92%" }} />上下文范围</span><span><i style={{ width: "28%" }} />并行能力</span></div>
          </article>
          <article className="race-card cnn-card">
            <div className="race-title"><span>CNN</span><strong>滑窗阅读</strong></div>
            <div className="cnn-visual">
              <div className="filter-window" />
              {["我", "喜欢", "研究", "注意力"].map((token) => <b key={token}>{token}</b>)}
            </div>
            <h3>能并行，但一层只看局部</h3>
            <p>卷积核同时滑动很快；想连接很远的词，通常需要堆叠更多层来扩大感受野。</p>
            <div className="race-meters"><span><i style={{ width: "46%" }} />单层范围</span><span><i style={{ width: "88%" }} />并行能力</span></div>
          </article>
          <article className="race-card attention-race-card">
            <div className="race-title"><span>SELF-ATTENTION</span><strong>全局检索</strong></div>
            <div className="global-visual">
              <b className="global-query">研究</b>
              <div><span>我</span><span>喜欢</span><span>研究</span><span>注意力</span></div>
              <i className="beam b1" /><i className="beam b2" /><i className="beam b3" /><i className="beam b4" />
            </div>
            <h3>每个位置直接看全局，并且一起算</h3>
            <p>所有 Query 与所有 Key 一次矩阵乘法完成匹配，GPU 很擅长这种大规模并行计算。</p>
            <div className="race-meters"><span><i style={{ width: "100%" }} />上下文范围</span><span><i style={{ width: "96%" }} />并行能力</span></div>
          </article>
        </div>
        <div className="motivation-conclusion">
          <span>核心交换</span>
          <p>Self-Attention 用 <strong>S × S 的计算与显存</strong>，换来<strong>全局信息连接 + 高度并行</strong>。这也是它强大、同时又在长序列上昂贵的根本原因。</p>
        </div>
      </section>

      <section className="section warm-section" id="matrix">
        <SectionTitle
          eyebrow="01 · MATRIX WARM-UP"
          title="先别怕矩阵：它只是把很多次“行 × 列”一起算"
          description="Attention 最核心的操作就是矩阵乘法。看懂一个格子如何得到，就看懂了整张矩阵。"
        />

        <div className="concept-grid three">
          <article className="concept-card">
            <span className="card-index">A</span>
            <h3>标量 Scalar</h3>
            <div className="type-demo scalar-demo">7</div>
            <p>一个数。比如温度、分数、某个权重。</p>
          </article>
          <article className="concept-card">
            <span className="card-index">B</span>
            <h3>向量 Vector</h3>
            <div className="type-demo vector-demo"><span>2</span><span>1</span><span>0</span></div>
            <p>一排数。可以把一个词的多种特征装在一起。</p>
          </article>
          <article className="concept-card dark-card">
            <span className="card-index">C</span>
            <h3>矩阵 Matrix</h3>
            <div className="type-demo mini-matrix"><span>1</span><span>2</span><span>0</span><span>0</span><span>1</span><span>3</span></div>
            <p>很多向量整齐排成表。句子中每个词一行。</p>
          </article>
        </div>

        <div className="matrix-lab">
          <div className="lab-header">
            <div>
              <span className="lab-tag">INTERACTIVE · 点击结果矩阵任意格</span>
              <h3>矩阵乘法拆解台</h3>
            </div>
            <div className="shape-rule"><span>内维相同</span> (2 × <b>3</b>) · (<b>3</b> × 2) → (2 × 2)</div>
          </div>

          <div className="matrix-equation">
            <Matrix values={matrixA} label="A" highlightRow={rowIndex} />
            <span className="operator">×</span>
            <Matrix values={matrixB} label="B" highlightColumn={columnIndex} />
            <span className="operator">=</span>
            <Matrix
              values={matrixC}
              label="C"
              highlightRow={rowIndex}
              highlightColumn={columnIndex}
              onCellClick={(row, column) => setSelectedCell([row, column])}
            />
          </div>

          <div className="calculation-trace" aria-live="polite">
            <div className="trace-coordinate">C<sub>{rowIndex + 1},{columnIndex + 1}</sub></div>
            <div className="trace-copy">
              <span>取 A 的第 {rowIndex + 1} 行</span>
              <strong>[ {selectedRow.join(" , ")} ]</strong>
            </div>
            <span className="trace-symbol">·</span>
            <div className="trace-copy">
              <span>取 B 的第 {columnIndex + 1} 列</span>
              <strong>[ {selectedColumn.join(" , ")} ]</strong>
            </div>
            <span className="trace-symbol">=</span>
            <div className="trace-result">
              {selectedRow.map((value, index) => `${value}×${selectedColumn[index]}`).join(" + ")}
              <strong>= {matrixC[rowIndex][columnIndex]}</strong>
            </div>
          </div>
        </div>

        <aside className="speaker-note">
          <span>讲师提示 · 5 MIN</span>
          <p>先让听众猜 C 的形状，再点击四个结果格。强调“行找问题、列找对象”——后面 QKᵀ 的每个格子同样是一行 Query 与一列 Key 做点积。</p>
        </aside>
      </section>

      <section className="section ink-section" id="qkv">
        <SectionTitle
          eyebrow="02 · THE LIBRARY ANALOGY"
          title="Q、K、V 到底是什么？把它想成一次图书检索"
          description="三个字母不是三份神秘数据，而是同一批词经过三个可学习的线性投影后，扮演的三种角色。"
        />

        <div className="qkv-stage">
          <div className="qkv-card q-card">
            <div className="qkv-letter">Q</div>
            <span>QUERY · 查询</span>
            <h3>“我现在想找什么？”</h3>
            <p>当前词带着一个问题去扫描全句。像你输入检索框的关键词。</p>
            <code>Q = X · Wq</code>
          </div>
          <div className="qkv-connector"><span>相似度</span><b>Q · Kᵀ</b></div>
          <div className="qkv-card k-card">
            <div className="qkv-letter">K</div>
            <span>KEY · 索引</span>
            <h3>“我可以被怎样找到？”</h3>
            <p>每个词给自己贴上可检索标签。像书脊上的题名、作者和分类号。</p>
            <code>K = X · Wk</code>
          </div>
          <div className="qkv-connector"><span>拿到权重后</span><b>Σ α · V</b></div>
          <div className="qkv-card v-card">
            <div className="qkv-letter">V</div>
            <span>VALUE · 内容</span>
            <h3>“找到我后，取走什么？”</h3>
            <p>真正被汇总的信息。像翻开书后读到的正文，而不是书脊标签。</p>
            <code>V = X · Wv</code>
          </div>
        </div>

        <div className="truth-callout">
          <span className="truth-icon">!</span>
          <div>
            <strong>最容易讲错的一点</strong>
            <p>Q、K、V 不是人为写死的“问题、关键词、答案”。Wq、Wk、Wv 从训练数据中学习，模型自己找到有用的检索方式；同一个词在不同上下文里也会得到不同的表示。</p>
          </div>
        </div>
      </section>

      <section className="section light-section" id="attention">
        <SectionTitle
          eyebrow="03 · SCALED DOT-PRODUCT ATTENTION"
          title="把 Attention 完整跑一遍"
          description="请选择一个词作为 Query，再观察它怎样给所有 Key 打分、归一化，最后混合 Value。数字是教学用的微型示例。"
        />

        <div className="formula-banner">
          <div className="formula-main">Attention(Q, K, V) = softmax( <span>QKᵀ</span> / √d<sub>k</sub> + Mask ) V</div>
          <div className="formula-legend"><i /> 相似度打分 <i /> 稳定数值 <i /> 变成权重 <i /> 汇总信息</div>
        </div>

        <div className="attention-lab">
          <div className="attention-controls">
            <div>
              <span className="lab-tag">STEP 0 · 选择当前 Query</span>
              <div className="token-picker">
                {tokens.map((token, index) => (
                  <button
                    type="button"
                    key={token}
                    className={selectedToken === index ? "is-selected" : ""}
                    onClick={() => setSelectedToken(index)}
                  >
                    <small>{index + 1}</small>{token}
                  </button>
                ))}
              </div>
            </div>
            <label className="causal-toggle">
              <input type="checkbox" checked={causal} onChange={(event) => setCausal(event.target.checked)} />
              <span className="switch"><i /></span>
              <span><strong>因果 Mask</strong><small>开启后不能看未来</small></span>
            </label>
          </div>

          <div className="attention-steps">
            <article className="step-card score-card">
              <div className="step-number">01</div>
              <span>DOT PRODUCT</span>
              <h3>Q 与每个 K 做点积</h3>
              <code>q = [{attention.q.map((value) => value.toFixed(1)).join(", ")}]</code>
              <div className="score-list">
                {attention.rawScores.map((score, index) => (
                  <div key={tokens[index]}>
                    <span>{tokens[index]}</span><b>{formatNumber(score)}</b>
                  </div>
                ))}
              </div>
              <p>方向越相似，点积通常越大。</p>
            </article>

            <article className="step-card scale-card">
              <div className="step-number">02</div>
              <span>SCALE + MASK</span>
              <h3>除以 √2，再遮住未来</h3>
              <div className="scale-value">÷ √d<sub>k</sub><strong>÷ 1.414</strong></div>
              <div className="score-list">
                {attention.scaledScores.map((score, index) => (
                  <div className={!Number.isFinite(score) ? "masked" : ""} key={tokens[index]}>
                    <span>{tokens[index]}</span><b>{formatNumber(score)}</b>
                  </div>
                ))}
              </div>
              <p>缩放避免大点积把 Softmax 推到饱和区。</p>
            </article>

            <article className="step-card softmax-card">
              <div className="step-number">03</div>
              <span>SOFTMAX</span>
              <h3>把分数变成概率权重</h3>
              <div className="weight-bars">
                {attention.weights.map((weight, index) => (
                  <div key={tokens[index]}>
                    <span>{tokens[index]}</span>
                    <div><i style={{ width: `${weight * 100}%` }} /></div>
                    <b>{(weight * 100).toFixed(1)}%</b>
                  </div>
                ))}
              </div>
              <div className="sum-check">Σ 权重 = <strong>{attention.weights.reduce((a, b) => a + b, 0).toFixed(2)}</strong></div>
            </article>

            <article className="step-card output-card">
              <div className="step-number">04</div>
              <span>WEIGHTED SUM</span>
              <h3>按权重混合所有 V</h3>
              <div className="output-vector">
                {attention.output.map((value, index) => <strong key={index}>{value.toFixed(3)}</strong>)}
              </div>
              <p className="output-caption">这是“{tokens[selectedToken]}”的新上下文表示</p>
              <div className="output-insight">它仍是一个向量，但已经带上全句中相关词的信息。</div>
            </article>
          </div>
        </div>

        <div className="why-scale">
          <div><span>不缩放</span><strong>Softmax(2, 8, 14)</strong><div className="distribution sharp"><i /><i /><i /></div><small>几乎 one-hot，梯度很小</small></div>
          <div className="versus">VS</div>
          <div><span>除以 √dₖ</span><strong>Softmax(0.5, 2, 3.5)</strong><div className="distribution smooth"><i /><i /><i /></div><small>分布仍有区分，也保留梯度</small></div>
          <p><b>一句话：</b>维度越大，点积的方差越大；缩放把数值拉回 Softmax 舒适区。</p>
        </div>
      </section>

      <section className="section multi-section" id="multihead">
        <SectionTitle
          eyebrow="04 · MULTI-HEAD ATTENTION"
          title="一个视角不够，就让多个“检索专家”并行工作"
          description="每个头使用不同的 Wq/Wk/Wv 投影，在更小的子空间里独立做 Attention；最后拼接并通过 Wo 融合。"
        />

        <div className="head-selector" role="tablist" aria-label="选择注意力头">
          {heads.map((head, index) => (
            <button
              type="button"
              role="tab"
              aria-selected={selectedHead === index}
              className={`${selectedHead === index ? "is-selected" : ""} ${head.color}`}
              key={head.name}
              onClick={() => setSelectedHead(index)}
            >
              <span>{head.name}</span><strong>{head.title}</strong><i />
            </button>
          ))}
        </div>

        <div className={`head-lab ${heads[selectedHead].color}`}>
          <div className="heatmap-panel">
            <div className="heatmap-axis top-axis"><span />{tokens.map((token) => <b key={token}>{token}</b>)}</div>
            <div className="heatmap-body">
              <div className="heatmap-axis side-axis">{tokens.map((token) => <b key={token}>{token}</b>)}</div>
              <div className="heatmap-grid">
                {heads[selectedHead].matrix.flatMap((row, rowIdx) =>
                  row.map((weight, colIdx) => (
                    <div
                      key={`${rowIdx}-${colIdx}`}
                      className="heat-cell"
                      style={{ "--heat": `${Math.round(weight * 100)}%` } as React.CSSProperties}
                      title={`${tokens[rowIdx]} → ${tokens[colIdx]}: ${(weight * 100).toFixed(0)}%`}
                    >
                      {(weight * 100).toFixed(0)}
                    </div>
                  )),
                )}
              </div>
            </div>
            <div className="heatmap-caption"><span>Query 行</span><span>Key 列 · 颜色越深，注意越多</span></div>
          </div>
          <div className="head-explanation">
            <span className="lab-tag">当前观察 · {heads[selectedHead].name}</span>
            <h3>{heads[selectedHead].title}</h3>
            <p>{heads[selectedHead].note}</p>
            <div className="head-pipeline">
              <div><small>输入</small><strong>X</strong><span>[B,S,512]</span></div>
              <i>→</i>
              <div><small>拆成 8 头</small><strong>8 × 64</strong><span>[B,8,S,64]</span></div>
              <i>→</i>
              <div><small>拼接 + Wo</small><strong>Concat</strong><span>[B,S,512]</span></div>
            </div>
            <div className="head-equation">headᵢ = Attention(QWᵢ<sup>Q</sup>, KWᵢ<sup>K</sup>, VWᵢ<sup>V</sup>)</div>
          </div>
        </div>

        <div className="myth-grid">
          <div><span>误区</span><strong>“头越多，模型一定越好”</strong><p>头数增加会缩小每个 head 的维度；过多可能冗余，还会改变性能与显存行为。</p></div>
          <div><span>关键</span><strong>d_model 必须能被 num_heads 整除</strong><p>经典实现靠 reshape 拆分特征维，d_head = d_model / h。</p></div>
          <div><span>算子视角</span><strong>真正的大头是 S × S</strong><p>注意力分数矩阵随序列长度平方增长，这是长上下文成本的来源。</p></div>
        </div>
      </section>

      <section className="section architecture-section" id="transformer">
        <SectionTitle
          eyebrow="05 · THE WHOLE TRANSFORMER"
          title="Attention 是发动机，Transformer 才是整辆车"
          description="原始 Transformer 是编码器—解码器架构。今天常见的大模型通常选择其中一侧，再堆叠许多层。"
        />

        <div className="architecture-tabs" role="tablist">
          <button type="button" className={architecture === "original" ? "is-selected" : ""} onClick={() => setArchitecture("original")}>原始 Transformer <small>翻译</small></button>
          <button type="button" className={architecture === "encoder" ? "is-selected" : ""} onClick={() => setArchitecture("encoder")}>Encoder-only <small>BERT 类</small></button>
          <button type="button" className={architecture === "decoder" ? "is-selected" : ""} onClick={() => setArchitecture("decoder")}>Decoder-only <small>GPT 类</small></button>
        </div>

        <div className={`architecture-board mode-${architecture}`}>
          {(architecture === "original" || architecture === "encoder") && (
            <div className="tower encoder-tower">
              <div className="tower-title"><span>ENCODER</span><strong>理解输入</strong></div>
              <div className="embedding-block">词向量 + 位置编码</div>
              <div className="layer-stack">
                <span className="repeat-badge">× N 层</span>
                <div className="layer-block attention-block"><small>01</small><strong>Multi-Head Self-Attention</strong><span>任意词可以看见全部输入词</span></div>
                <div className="add-norm">残差 Add + LayerNorm</div>
                <div className="layer-block ffn-block"><small>02</small><strong>Feed Forward Network</strong><span>每个位置独立做非线性变换</span></div>
                <div className="add-norm">残差 Add + LayerNorm</div>
              </div>
              <div className="tower-output">上下文化的输入表示</div>
            </div>
          )}

          {architecture === "original" && (
            <div className="cross-bridge"><span>Encoder 的 K、V</span><i>→</i><b>Cross-Attention</b></div>
          )}

          {(architecture === "original" || architecture === "decoder") && (
            <div className="tower decoder-tower">
              <div className="tower-title"><span>DECODER</span><strong>逐词生成</strong></div>
              <div className="embedding-block">已生成词 + 位置编码</div>
              <div className="layer-stack">
                <span className="repeat-badge">× N 层</span>
                <div className="layer-block masked-block"><small>01</small><strong>Masked Self-Attention</strong><span>只能看当前位置与过去</span></div>
                <div className="add-norm">残差 Add + LayerNorm</div>
                {architecture === "original" && <><div className="layer-block cross-block"><small>02</small><strong>Cross-Attention</strong><span>Q 来自 Decoder，K/V 来自 Encoder</span></div><div className="add-norm">残差 Add + LayerNorm</div></>}
                <div className="layer-block ffn-block"><small>{architecture === "original" ? "03" : "02"}</small><strong>Feed Forward Network</strong><span>扩维 → 激活 → 降维</span></div>
                <div className="add-norm">残差 Add + LayerNorm</div>
              </div>
              <div className="tower-output">Linear + Softmax → 下一个词</div>
            </div>
          )}
        </div>

        <div className="architecture-explainer" aria-live="polite">
          {architecture === "original" && <><strong>原始论文版本</strong><p>Encoder 读取完整源句；Decoder 一边看已生成内容，一边通过 Cross-Attention 读取 Encoder 输出，适合机器翻译等序列到序列任务。</p></>}
          {architecture === "encoder" && <><strong>Encoder-only</strong><p>每个位置能双向看到整段输入，擅长理解、分类、检索与信息抽取。没有因果 Mask，也不天然用于逐词续写。</p></>}
          {architecture === "decoder" && <><strong>Decoder-only</strong><p>使用因果 Mask 进行下一个 Token 预测。不断把新 Token 接到序列末尾，就能持续生成文本；现代大语言模型多采用这一路线。</p></>}
        </div>

        <div className="position-panel">
          <div>
            <span className="lab-tag">为什么还需要位置编码？</span>
            <h3>Attention 本身不认识“先后顺序”</h3>
            <p>若只打乱输入行，纯 Attention 会跟着一起打乱输出，并不知道“狗咬人”和“人咬狗”的差别来自顺序。位置编码把第 1、2、3… 个位置的信息注入词向量。</p>
          </div>
          <div className="position-demo">
            {[0, 1, 2, 3, 4, 5].map((value) => (
              <div key={value}><span>{value + 1}</span><i style={{ height: `${24 + Math.sin(value) * 13}px` }} /><i style={{ height: `${24 + Math.cos(value) * 13}px` }} /></div>
            ))}
          </div>
        </div>
      </section>

      <section className="section code-section" id="code">
        <SectionTitle
          eyebrow="06 · CLASSIC CODE WALKTHROUGH"
          title="从公式对照到经典 PyTorch 代码"
          description="现场讲解时按 01 → 04 切换：先认出公式中的每一步，再看 shape 如何在多头之间变化，最后落到测试。"
        />

        <div className="code-explorer">
          <div className="code-tabs" role="tablist">
            {(Object.keys(codeSnippets) as Array<keyof typeof codeSnippets>).map((key) => (
              <button
                type="button"
                role="tab"
                aria-selected={activeCode === key}
                className={activeCode === key ? "is-selected" : ""}
                key={key}
                onClick={() => setActiveCode(key)}
              >
                <span>{codeSnippets[key].label.split(" · ")[0]}</span>
                {codeSnippets[key].label.split(" · ")[1]}
              </button>
            ))}
          </div>
          <div className="code-window">
            <div className="code-toolbar">
              <div><i /><i /><i /><span>attention_walkthrough.py</span></div>
              <button type="button" onClick={copyCode}>{copied ? "已复制 ✓" : "复制代码"}</button>
            </div>
            <div className="code-content">
              <pre aria-label={code.title}>
                {code.code.split("\n").map((line, index) => (
                  <code key={index}><span>{String(index + 1).padStart(2, "0")}</span>{line || " "}</code>
                ))}
              </pre>
              <aside className="code-note">
                <span>WALKTHROUGH</span>
                <h3>{code.title}</h3>
                <p>{code.note}</p>
                <div className="shape-flow">
                  {activeCode === "attention" && <><b>[B,H,S,D]</b><i>QKᵀ</i><b>[B,H,S,S]</b><i>× V</i><b>[B,H,S,D]</b></>}
                  {activeCode === "multihead" && <><b>[B,S,512]</b><i>split</i><b>[B,8,S,64]</b><i>concat</i><b>[B,S,512]</b></>}
                  {activeCode === "block" && <><b>x</b><i>+ Attn</i><b>x′</b><i>+ FFN</i><b>output</b></>}
                  {activeCode === "test" && <><b>actual</b><i>≈</i><b>reference</b><i>+</i><b>invariants</b></>}
                </div>
              </aside>
            </div>
          </div>
        </div>

        <div className="code-map">
          <div><code>q @ k.transpose(-2, -1)</code><span>QKᵀ · 两两相似度</span></div>
          <div><code>/ math.sqrt(d_k)</code><span>缩放 · 数值稳定</span></div>
          <div><code>softmax(dim=-1)</code><span>沿 Key 维归一化</span></div>
          <div><code>weights @ v</code><span>加权汇总 Value</span></div>
        </div>
      </section>

      <section className="section testing-section" id="testing">
        <SectionTitle
          eyebrow="07 · OPERATOR TESTING"
          title="从“会算”到“敢上线”：Attention 算子怎么测"
          description="算子测试不只验证一个输出数值。我们要同时锁住 shape、语义、精度、梯度、布局与性能。"
        />

        <div className="testing-hero">
          <div className="contract-card">
            <div className="contract-title"><span>INPUT CONTRACT</span><b>经典 4D 布局</b></div>
            <div className="tensor-contract">
              <div><strong>B</strong><span>Batch</span><small>样本数</small></div>
              <div><strong>H</strong><span>Heads</span><small>头数</small></div>
              <div><strong>S</strong><span>Sequence</span><small>序列长度</small></div>
              <div><strong>D</strong><span>Head dim</span><small>每头维度</small></div>
            </div>
            <code>Q, K, V: [B, H, S, D]</code>
          </div>
          <div className="complexity-card">
            <span>COMPLEXITY</span>
            <div className="big-o">O(S² · D)</div>
            <p>序列长度 S 翻倍，注意力分数矩阵大约变成 <strong>4 倍</strong>。测试长序列时要特别观察显存峰值。</p>
            <div className="complexity-bars"><i /><i /><i /><i /></div>
          </div>
        </div>

        <div className="test-table" role="table" aria-label="Attention 算子测试清单">
          <div className="test-row test-header" role="row"><span>检查面</span><span>最小测试设计</span><span>级别</span></div>
          {testChecklist.map(([name, design, priority], index) => (
            <div className="test-row" role="row" key={name}>
              <span><i>{String(index + 1).padStart(2, "0")}</i><strong>{name}</strong></span>
              <span>{design}</span>
              <span className={`priority priority-${priority.toLowerCase()}`}>{priority}</span>
            </div>
          ))}
        </div>

        <div className="bug-cards">
          <article><span>BUG 01</span><h3>Softmax 维度写错</h3><code>dim=-2 ✕　dim=-1 ✓</code><p>应该让同一 Query 对所有 Key 的权重和为 1。</p></article>
          <article><span>BUG 02</span><h3>Mask 放在 Softmax 后</h3><code>softmax → mask ✕</code><p>直接清零会破坏概率和；必须先把 score 设为 −∞。</p></article>
          <article><span>BUG 03</span><h3>漏掉缩放</h3><code>QKᵀ / √dₖ</code><p>小 shape 可能看不出，维度增大后误差与梯度问题更明显。</p></article>
          <article><span>BUG 04</span><h3>transpose 后直接 view</h3><code>.contiguous().view(...)</code><p>非连续内存可能报错，或在某些实现里得到错误布局。</p></article>
        </div>

        <div className="test-strategy">
          <div className="strategy-number">3 × 3</div>
          <div><span className="lab-tag">推荐最小参数矩阵</span><h3>三类 shape × 三种 dtype</h3><p>先用小矩阵把逻辑覆盖完整，再单独扩到真实规模做性能回归。</p></div>
          <div className="strategy-grid">
            <span>S=1 · 边界</span><span>FP32 · 基准</span><span>S=7 · 非对齐</span>
            <span>FP16 · 容差</span><span>S_q≠S_k · Cross</span><span>BF16 · 范围</span>
          </div>
        </div>

        <aside className="speaker-note dark-note">
          <span>讲师提示 · 8 MIN</span>
          <p>现场最好先展示一个“Softmax 行和不为 1”的失败用例，让大家从性质反推轴写错；这比只说 dim=-1 更容易形成记忆。</p>
        </aside>
      </section>

      <section className="section quiz-section" id="quiz">
        <SectionTitle
          eyebrow="KNOWLEDGE CHECK"
          title="三道题，确认真的听懂了"
          description="点击选项立即查看解释。讲解时可以让听众先举手，再公布答案。"
        />
        <div className="quiz-grid">
          {quiz.map((item, questionIndex) => (
            <article className="quiz-card" key={item.question}>
              <span>0{questionIndex + 1}</span>
              <h3>{item.question}</h3>
              <div className="quiz-options">
                {item.answers.map((answer, answerIndex) => {
                  const answered = quizAnswers[questionIndex] !== undefined;
                  const chosen = quizAnswers[questionIndex] === answerIndex;
                  const correct = answerIndex === item.correct;
                  return (
                    <button
                      type="button"
                      key={answer}
                      className={`${chosen ? "is-chosen" : ""} ${answered && correct ? "is-correct" : ""} ${answered && chosen && !correct ? "is-wrong" : ""}`}
                      onClick={() => setQuizAnswers((current) => ({ ...current, [questionIndex]: answerIndex }))}
                    >
                      <i>{String.fromCharCode(65 + answerIndex)}</i>{answer}
                    </button>
                  );
                })}
              </div>
              {quizAnswers[questionIndex] !== undefined && (
                <p className="quiz-explain"><strong>{quizAnswers[questionIndex] === item.correct ? "答对了" : "再想一步"}</strong>{item.explain}</p>
              )}
            </article>
          ))}
        </div>
      </section>

      <section className="section summary-section" id="summary">
        <div className="summary-heading">
          <span>ONE-PAGE RECAP</span>
          <h2>最后只记住这 6 句话</h2>
        </div>
        <div className="summary-grid">
          {[
            ["01", "词先变向量", "Embedding 把离散 Token 变成可计算的连续表示。"],
            ["02", "Q 去找 K", "点积衡量每个 Query 与所有 Key 的匹配程度。"],
            ["03", "Softmax 变权重", "缩放、Mask 后归一化，每行权重和为 1。"],
            ["04", "权重混合 V", "新表示是相关 Value 的加权汇总。"],
            ["05", "多头看多面", "多个子空间并行关注不同关系，再拼接融合。"],
            ["06", "Attention 构成 Transformer", "再配合位置、残差、归一化和 FFN，层层堆叠。"],
          ].map(([number, title, copy]) => (
            <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>
          ))}
        </div>

        <div className="importance-card">
          <div className="importance-kicker">WHY IT MATTERS</div>
          <h2>为什么 Attention 如此重要？</h2>
          <p>它让每个位置能够<strong>按内容直接读取任意位置的信息</strong>，长距离关系不必沿着循环网络一步步传递；所有位置的计算又能矩阵化并行。正是这种“动态路由信息”的能力，让同一套骨架可以处理语言、视觉、语音、代码与多模态数据。</p>
          <div className="importance-tags"><span>全局依赖</span><span>高度并行</span><span>上下文表示</span><span>可扩展骨架</span></div>
        </div>
      </section>

      <footer>
        <div className="footer-brand"><span className="brand-mark">A</span><div><strong>ATTENTION LAB</strong><small>从一个乘法，到 Transformer 的核心算子。</small></div></div>
        <div className="references">
          <span>延伸阅读</span>
          <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">原论文 · Attention Is All You Need ↗</a>
          <a href="https://nlp.seas.harvard.edu/annotated-transformer/" target="_blank" rel="noreferrer">The Annotated Transformer ↗</a>
          <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html" target="_blank" rel="noreferrer">PyTorch MultiheadAttention ↗</a>
          <a href="https://zhuanlan.zhihu.com/p/340149804" target="_blank" rel="noreferrer">讲解思路参考文章 ↗</a>
        </div>
        <div className="footer-note">适合零基础讲解 · 建议配合代码逐步演示</div>
      </footer>
    </main>
  );
}
