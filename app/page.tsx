"use client";

import { useEffect, useMemo, useState } from "react";
import katex from "katex";

type FormulaProps = {
  tex: string;
  block?: boolean;
  className?: string;
};

function Formula({ tex, block = false, className = "" }: FormulaProps) {
  let html: string;
  let errored = false;
  try {
    html = katex.renderToString(tex, {
      displayMode: block,
      throwOnError: false,
      output: "html",
      strict: false,
    });
    // 渲染产出含错误标记时降级
    if (html.includes("katex-error") || html.includes("Parsing error")) {
      errored = true;
    }
  } catch {
    errored = true;
    html = "";
  }

  const Tag = block ? "div" : "span";
  if (errored) {
    return (
      <Tag
        className={`math ${block ? "math-block" : "math-inline"} math-error ${className}`}
      >
        {tex}
      </Tag>
    );
  }
  return (
    <Tag
      className={`math ${block ? "math-block" : "math-inline"} ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

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
  matrixB[0].map((_, column) =>
    row.reduce((sum, value, inner) => sum + value * matrixB[inner][column], 0),
  ),
);

const positions = ["x₁", "x₂", "x₃", "x₄"];
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

const attentionHeads = [
  {
    name: "Head 1",
    role: "长程关联",
    note: "非相邻位置之间出现高响应。",
    color: "coral",
    matrix: [
      [0.58, 0.12, 0.22, 0.08],
      [0.18, 0.16, 0.12, 0.54],
      [0.66, 0.08, 0.2, 0.06],
      [0.14, 0.46, 0.09, 0.31],
    ],
  },
  {
    name: "Head 2",
    role: "局部结构",
    note: "权重集中在主对角线附近。",
    color: "cyan",
    matrix: [
      [0.62, 0.27, 0.07, 0.04],
      [0.24, 0.48, 0.22, 0.06],
      [0.07, 0.24, 0.48, 0.21],
      [0.03, 0.08, 0.3, 0.59],
    ],
  },
  {
    name: "Head 3",
    role: "全局汇聚",
    note: "分布较平滑，保留全局统计信息。",
    color: "violet",
    matrix: [
      [0.28, 0.24, 0.25, 0.23],
      [0.23, 0.29, 0.22, 0.26],
      [0.27, 0.2, 0.3, 0.23],
      [0.22, 0.27, 0.21, 0.3],
    ],
  },
];

const codeSamples = {
  attention: {
    label: "核心算子",
    title: "Scaled Dot-Product Attention",
    code: `def attention(q, k, v, mask=None):
    d_k = q.size(-1)
    logits = q @ k.transpose(-2, -1)
    logits = logits / math.sqrt(d_k)

    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(logits, dim=-1)
    output = weights @ v
    return output, weights`,
  },
  multihead: {
    label: "多头封装",
    title: "Multi-Head Attention",
    code: `def forward(self, x, mask=None):
    B, S, _ = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)

    def split(t):
        return t.view(B, S, self.h, self.d_head) \
                .transpose(1, 2)

    q, k, v = map(split, (q, k, v))
    z, weights = attention(q, k, v, mask)

    z = z.transpose(1, 2).contiguous() \
         .view(B, S, self.h * self.d_head)
    return self.out(z), weights`,
  },
  encoder: {
    label: "Encoder 层",
    title: "Transformer Encoder Layer · Post-LN",
    code: `class EncoderLayer(nn.Module):
    def forward(self, x, mask=None):
        attn_out, weights = self.self_attn(
            x, x, x, attn_mask=mask
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.linear2(
            self.dropout(F.relu(self.linear1(x)))
        )
        x = self.norm2(x + self.dropout(ffn_out))
        return x, weights`,
  },
};

function softmax(values: number[]) {
  const finite = values.filter(Number.isFinite);
  const maximum = Math.max(...finite);
  const exponents = values.map((value) =>
    Number.isFinite(value) ? Math.exp(value - maximum) : 0,
  );
  const denominator = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / denominator);
}

function SectionHeader({
  number,
  title,
  description,
}: {
  number: string;
  title: string;
  description: string;
}) {
  return (
    <header className="section-header">
      <div className="section-meta">
        <span>{number}</span>
      </div>
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
    </header>
  );
}

/* 经典 Transformer 论文 Figure 1 的精致 SVG 重绘 */
function TransformerFigure() {
  return (
    <div className="svg-figure">
      <svg viewBox="0 0 940 600" width="940" role="img"
        aria-label="经典 Transformer Encoder-Decoder 结构图">
        <defs>
          <marker id="fah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
            <path d="M0,0 L8,4.5 L0,9 z" fill="#66767a" />
          </marker>
        </defs>
        {/* 标题 */}
        <text x="250" y="26" textAnchor="middle" className="fig-title" fill="#56d6dd">Encoder × N（左）</text>
        <text x="710" y="26" textAnchor="middle" className="fig-title" fill="#ff6544">Decoder × N（右）</text>

        {/* ===== ENCODER ===== */}
        <rect x="60" y="60" width="120" height="40" rx="8" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
        <text x="120" y="85" textAnchor="middle" className="fig-lbl">Input Embedding</text>
        <circle cx="200" cy="80" r="15" fill="#fff" stroke="#a18aff" />
        <text x="200" y="85" textAnchor="middle" fill="#a18aff" fontSize="15">+</text>
        <text x="200" y="50" textAnchor="middle" className="fig-mute">Positional</text>
        <text x="200" y="62" textAnchor="middle" className="fig-mute">Encoding</text>
        <path d="M180,80 H186" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />

        <rect x="80" y="125" width="290" height="350" rx="14" fill="none" stroke="rgba(17,35,38,.2)" strokeDasharray="5 5" />
        <text x="225" y="120" textAnchor="middle" className="fig-mute">Encoder Layer (堆叠 N 次)</text>

        <rect x="120" y="150" width="210" height="54" rx="9" fill="rgba(86,214,221,.12)" stroke="#56d6dd" />
        <text x="225" y="174" textAnchor="middle" fill="#2a8b91" style={{ fontWeight: 700, fontSize: "13px" }}>Multi-Head Self-Attention</text>
        <text x="225" y="192" textAnchor="middle" className="fig-mute">本节讲的核心算子</text>

        <rect x="150" y="222" width="150" height="36" rx="8" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
        <text x="225" y="245" textAnchor="middle" className="fig-lbl">Add &amp; Norm（残差+归一化）</text>

        <rect x="120" y="278" width="210" height="50" rx="9" fill="rgba(161,138,255,.12)" stroke="#a18aff" />
        <text x="225" y="300" textAnchor="middle" fill="#6d57d6" style={{ fontWeight: 700, fontSize: "13px" }}>Feed-Forward Network</text>
        <text x="225" y="318" textAnchor="middle" className="fig-mute">两层 MLP（逐位置作用）</text>

        <rect x="150" y="346" width="150" height="36" rx="8" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
        <text x="225" y="369" textAnchor="middle" className="fig-lbl">Add &amp; Norm</text>

        <path d="M225,204 V218" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        <path d="M225,258 V274" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        <path d="M225,328 V342" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        {/* 残差旁路 */}
        <path d="M120,170 H100 V410 H225" stroke="#ff6544" strokeWidth="1.3" fill="none" strokeDasharray="4 3" markerEnd="url(#fah)" />
        <text x="92" y="395" className="fig-mute" fill="#ff6544">残差</text>

        <rect x="150" y="405" width="150" height="34" rx="8" fill="rgba(255,101,68,.12)" stroke="#ff6544" />
        <text x="225" y="427" textAnchor="middle" fill="#c44a2f">编码器输出（K, V）</text>

        <path d="M215,100 V150" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />

        {/* ===== DECODER ===== */}
        <rect x="560" y="60" width="120" height="40" rx="8" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
        <text x="620" y="85" textAnchor="middle" className="fig-lbl">Output Embedding</text>
        <circle cx="700" cy="80" r="15" fill="#fff" stroke="#a18aff" />
        <text x="700" y="85" textAnchor="middle" fill="#a18aff" fontSize="15">+</text>
        <text x="700" y="50" textAnchor="middle" className="fig-mute">Positional</text>
        <text x="700" y="62" textAnchor="middle" className="fig-mute">Encoding</text>
        <path d="M680,80 H686" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />

        <rect x="540" y="125" width="310" height="350" rx="14" fill="none" stroke="rgba(17,35,38,.2)" strokeDasharray="5 5" />
        <text x="695" y="120" textAnchor="middle" className="fig-mute">Decoder Layer (堆叠 N 次)</text>

        <rect x="575" y="150" width="240" height="50" rx="9" fill="rgba(255,101,68,.12)" stroke="#ff6544" />
        <text x="695" y="172" textAnchor="middle" fill="#c44a2f" style={{ fontWeight: 700, fontSize: "13px" }}>Masked Multi-Head Attention</text>
        <text x="695" y="189" textAnchor="middle" className="fig-mute">只能看过去（屏蔽未来位）</text>

        <rect x="620" y="216" width="150" height="34" rx="8" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
        <text x="695" y="238" textAnchor="middle" className="fig-lbl">Add &amp; Norm</text>

        <rect x="575" y="268" width="240" height="50" rx="9" fill="rgba(86,214,221,.12)" stroke="#56d6dd" />
        <text x="695" y="290" textAnchor="middle" fill="#2a8b91" style={{ fontWeight: 700, fontSize: "13px" }}>Cross Attention（编码-解码交互）</text>
        <text x="695" y="307" textAnchor="middle" className="fig-mute">Q 来自解码器，K,V 来自编码器</text>

        <rect x="620" y="334" width="150" height="34" rx="8" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
        <text x="695" y="356" textAnchor="middle" className="fig-lbl">Add &amp; Norm</text>

        <rect x="575" y="386" width="240" height="44" rx="9" fill="rgba(161,138,255,.12)" stroke="#a18aff" />
        <text x="695" y="413" textAnchor="middle" fill="#6d57d6" style={{ fontWeight: 700, fontSize: "13px" }}>Feed-Forward Network</text>

        <rect x="620" y="442" width="150" height="32" rx="8" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
        <text x="695" y="463" textAnchor="middle" className="fig-lbl">Add &amp; Norm</text>

        <path d="M695,200 V212" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        <path d="M695,250 V264" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        <path d="M695,318 V330" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        <path d="M695,368 V382" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        <path d="M695,430 V438" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />

        {/* 编码器 K,V 跨到 cross attention */}
        <path d="M300,422 C440,422 460,293 573,293" stroke="#a18aff" strokeWidth="1.4" fill="none" strokeDasharray="4 3" markerEnd="url(#fah)" />
        <text x="430" y="360" className="fig-mute" fill="#a18aff">编码器 K, V 传过来</text>

        <path d="M695,474 V492" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />
        <rect x="600" y="495" width="190" height="34" rx="8" fill="rgba(161,138,255,.12)" stroke="#a18aff" />
        <text x="695" y="517" textAnchor="middle" fill="#6d57d6">Linear → Softmax → 词概率</text>

        <path d="M685,100 V150" stroke="#66767a" strokeWidth="1.4" fill="none" markerEnd="url(#fah)" />

        {/* 图例 */}
        <g transform="translate(80,555)">
          <rect x="0" y="0" width="14" height="14" rx="3" fill="rgba(86,214,221,.12)" stroke="#56d6dd" />
          <text x="20" y="12" className="fig-mute">Attention</text>
          <rect x="110" y="0" width="14" height="14" rx="3" fill="rgba(161,138,255,.12)" stroke="#a18aff" />
          <text x="130" y="12" className="fig-mute">FFN</text>
          <rect x="190" y="0" width="14" height="14" rx="3" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
          <text x="210" y="12" className="fig-mute">Add&amp;Norm</text>
          <rect x="310" y="0" width="14" height="14" rx="3" fill="#fff" stroke="#a18aff" />
          <text x="330" y="12" className="fig-mute">位置编码</text>
        </g>
      </svg>
      <div className="svg-caption">图 · 论文 Figure 1 重绘 — Attention 在 Encoder/Decoder 中共出现三次，是同一算子</div>
    </div>
  );
}

/* 向量级 self-attention 四步推导 SVG（对应 PDF 图4-9） */
function AttentionStepsFigure() {
  return (
    <div className="svg-figure">
      <svg viewBox="0 0 920 430" width="920" role="img"
        aria-label="self-attention 向量级四步推导">
        <defs>
          <marker id="sah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L7,4 L0,8 z" fill="#66767a" />
          </marker>
        </defs>

        <text x="10" y="24" className="fig-title">步骤 ① 输入 → 线性变换得到 q / k / v</text>
        {/* 输入列 */}
        <g fontFamily="SFMono-Regular, Consolas, monospace" fontSize="12">
          {[70, 120, 170, 220].map((y, i) => (
            <g key={`x${i}`}>
              <rect x="20" y={y} width="40" height="30" rx="6" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
              <text x="40" y={y + 19} textAnchor="middle" fill="#18373b">x{i + 1}</text>
            </g>
          ))}
          {/* q */}
          <rect x="150" y="70" width="44" height="30" rx="6" fill="rgba(255,101,68,.14)" stroke="#ff6544" />
          <text x="172" y="89" textAnchor="middle" fill="#c44a2f">q₁</text>
          {/* k */}
          {[120, 170, 220].map((y, i) => (
            <g key={`k${i}`}>
              <rect x="150" y={y} width="44" height="30" rx="6" fill="rgba(161,138,255,.14)" stroke="#a18aff" />
              <text x="172" y={y + 19} textAnchor="middle" fill="#6d57d6">k{i + 1}</text>
            </g>
          ))}
          {/* v */}
          {[120, 170, 220].map((y, i) => (
            <g key={`v${i}`}>
              <rect x="240" y={y} width="44" height="30" rx="6" fill="rgba(86,214,221,.16)" stroke="#56d6dd" />
              <text x="262" y={y + 19} textAnchor="middle" fill="#2a8b91">v{i + 1}</text>
            </g>
          ))}
        </g>
        <text x="172" y="62" className="fig-mute" fill="#c44a2f">Wᵠ·x</text>
        <text x="172" y="278" className="fig-mute" fill="#6d57d6">Wᵏ·x</text>
        <text x="262" y="278" className="fig-mute" fill="#2a8b91">Wᵛ·x</text>

        <path d="M60,85 H148" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M60,135 H148" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M60,185 H148" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M60,235 H148" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M60,135 H238" stroke="#66767a" strokeWidth="1.3" fill="none" strokeDasharray="3 3" markerEnd="url(#sah)" />
        <text x="92" y="78" className="fig-mute">×Wᵠ</text>

        {/* 步骤② q·k 点积 */}
        <text x="360" y="24" className="fig-title">步骤 ② q₁ 与每个 k 点积 → 相关性 α</text>
        <g fontFamily="SFMono-Regular, Consolas, monospace" fontSize="12">
          {[120, 160, 200, 240].map((y, i) => (
            <g key={`a${i}`}>
              <rect x="370" y={y} width="56" height="30" rx="6" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
              <text x="398" y={y + 19} textAnchor="middle" fill="#2a8b91">α₁,{i + 1}</text>
            </g>
          ))}
        </g>
        <text x="398" y="290" className="fig-mute">q₁·kⱼ</text>
        <path d="M194,85 C300,85 330,135 368,135" stroke="#ff6544" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M194,185 C290,185 320,175 368,175" stroke="#a18aff" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M194,235 C300,235 330,215 368,215" stroke="#a18aff" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M194,135 C300,135 330,255 368,255" stroke="#a18aff" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />

        {/* 步骤③ softmax */}
        <text x="470" y="24" className="fig-title">步骤 ③ softmax → 归一化权重（和=1）</text>
        <g fontFamily="SFMono-Regular, Consolas, monospace" fontSize="12">
          {[120, 160, 200, 240].map((y, i) => (
            <g key={`ah${i}`}>
              <rect x="490" y={y} width="60" height="30" rx="6" fill="rgba(86,214,221,.12)" stroke="#56d6dd" />
              <text x="520" y={y + 19} textAnchor="middle" fill="#2a8b91">α̂₁,{i + 1}</text>
            </g>
          ))}
        </g>
        <text x="520" y="290" className="fig-mute">Σ=1.0</text>
        <path d="M426,135 H488" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M426,175 H488" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M426,215 H488" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M426,255 H488" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />

        {/* 步骤④ 加权求和 */}
        <text x="600" y="24" className="fig-title">步骤 ④ 权重 × 对应 v 求和 → b₁</text>
        <g fontFamily="SFMono-Regular, Consolas, monospace" fontSize="12">
          {[120, 160, 200, 240].map((y, i) => (
            <g key={`wv${i}`}>
              <rect x="610" y={y} width="120" height="30" rx="6" fill="#fffaf1" stroke="rgba(17,35,38,.14)" />
              <text x="670" y={y + 19} textAnchor="middle" fill="#2a8b91">α̂₁,{i + 1}·v{i + 1}</text>
            </g>
          ))}
        </g>
        <path d="M550,135 H608" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M550,175 H608" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M550,215 H608" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M550,255 H608" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M284,135 C450,135 500,135 608,135" stroke="#56d6dd" strokeWidth="1.3" fill="none" strokeDasharray="3 3" markerEnd="url(#sah)" />

        {/* b1 */}
        <rect x="780" y="175" width="70" height="40" rx="9" fill="rgba(200,237,105,.18)" stroke="#9bc23a" strokeWidth="1.6" />
        <text x="815" y="201" textAnchor="middle" fill="#6d7a1a" fontFamily="SFMono-Regular, Consolas, monospace" fontSize="16" fontWeight="700">b₁</text>
        <path d="M730,135 C760,135 770,180 778,188" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M730,175 H778" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M730,215 C760,215 770,200 778,198" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <path d="M730,255 C760,255 770,210 778,205" stroke="#66767a" strokeWidth="1.3" fill="none" markerEnd="url(#sah)" />
        <text x="815" y="235" className="fig-mute" fill="#6d7a1a">= Σ α̂·v</text>
      </svg>
      <div className="svg-caption">图 · 对应 PDF 图 4–9 — 用 q₁ 计算出 b₁ 的完整 4 步（b₂ b₃ b₄ 同理并行）</div>
    </div>
  );
}

function Matrix({
  values,
  label,
  activeRow,
  activeColumn,
  onSelect,
}: {
  values: number[][];
  label: string;
  activeRow?: number;
  activeColumn?: number;
  onSelect?: (row: number, column: number) => void;
}) {
  return (
    <div className="matrix">
      <div className="matrix-name">{label}</div>
      <div className="matrix-body">
        {values.map((row, rowIndex) => (
          <div className="matrix-row" key={`${label}-${rowIndex}`}>
            {row.map((value, columnIndex) => {
              const highlighted =
                rowIndex === activeRow || columnIndex === activeColumn;
              const selected =
                rowIndex === activeRow && columnIndex === activeColumn;
              return (
                <button
                  type="button"
                  key={`${rowIndex}-${columnIndex}`}
                  className={`${highlighted ? "highlighted" : ""} ${selected ? "selected" : ""}`}
                  onClick={() => onSelect?.(rowIndex, columnIndex)}
                  tabIndex={onSelect ? 0 : -1}
                  aria-label={`${label} 第 ${rowIndex + 1} 行第 ${columnIndex + 1} 列：${value}`}
                >
                  {value}
                </button>
              );
            })}
          </div>
        ))}
      </div>
      <small>{values.length} × {values[0].length}</small>
    </div>
  );
}

export default function Home() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [selectedCell, setSelectedCell] = useState<[number, number]>([0, 0]);
  const [queryIndex, setQueryIndex] = useState(2);
  const [causalMask, setCausalMask] = useState(false);
  const [headIndex, setHeadIndex] = useState(0);
  const [codeTab, setCodeTab] = useState<keyof typeof codeSamples>("attention");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const update = () => {
      const height = document.documentElement.scrollHeight - window.innerHeight;
      setScrollProgress(height > 0 ? (window.scrollY / height) * 100 : 0);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);

  const attention = useMemo(() => {
    const q = queries[queryIndex];
    const logits = keys.map((key) =>
      q.reduce((sum, component, index) => sum + component * key[index], 0),
    );
    const scaled = logits.map((value, index) =>
      causalMask && index > queryIndex
        ? Number.NEGATIVE_INFINITY
        : value / Math.sqrt(q.length),
    );
    const weights = softmax(scaled);
    const output = values[0].map((_, dimension) =>
      weights.reduce(
        (sum, weight, index) => sum + weight * values[index][dimension],
        0,
      ),
    );
    return { q, logits, scaled, weights, output };
  }, [causalMask, queryIndex]);

  const [matrixRow, matrixColumn] = selectedCell;
  const selectedA = matrixA[matrixRow];
  const selectedB = matrixB.map((row) => row[matrixColumn]);
  const activeCode = codeSamples[codeTab];

  async function copyCode() {
    await navigator.clipboard.writeText(activeCode.code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <main>
      <div className="progress" style={{ width: `${scrollProgress}%` }} />

      <nav className="nav" aria-label="演示章节">
        <a href="#top" className="logo"><i>A</i><span>ATTENTION<br /><small>TECHNICAL BRIEF</small></span></a>
        <div className="nav-sections">
          <a href="#why">01</a>
          <a href="#matrix">02</a>
          <a href="#attention">03</a>
          <a href="#multihead">04</a>
          <a href="#transformer">05</a>
          <a href="#code">06</a>
        </div>
        <span className="nav-mark">Q · K · V</span>
      </nav>

      <header className="hero" id="top">
        <div className="hero-copy">
          <span className="kicker">TRANSFORMER CORE OPERATOR</span>
          <h1>Attention</h1>
          <h2>从数学原理到经典实现与算子测试</h2>
          <p>从矩阵乘法出发，推导 Q/K/V、Scaled Dot-Product Attention、多头注意力与 Transformer Block。</p>
          <a href="#why" className="start-button">开始讲解 <span>↓</span></a>
        </div>
        <div className="hero-formula">
          <div className="formula-label">CORE EQUATION</div>
          <Formula
            block
            tex={String.raw`\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V`}
          />
          <div className="hero-shapes">
            <span><b>Q</b>[B,H,S<sub>q</sub>,D]</span>
            <span><b>K</b>[B,H,S<sub>k</sub>,D]</span>
            <span><b>A</b>[B,H,S<sub>q</sub>,S<sub>k</sub>]</span>
          </div>
        </div>
        <div className="timeline">
          {[
            "为什么需要",
            "矩阵乘法",
            "Attention",
            "多头",
            "Transformer",
            "代码与测试",
          ].map((label, index) => (
            <div key={label}><span>0{index + 1}</span><strong>{label}</strong></div>
          ))}
        </div>
      </header>

      <section className="section why-section" id="why">
        <SectionHeader
          number="01"
          title="为什么需要 Self-Attention"
          description="目标只有两个：全局依赖与并行计算。"
        />
        <div className="comparison-grid">
          <article>
            <span>RNN</span>
            <h3>全局，但串行</h3>
            <div className="sequence-line serial"><i>x₁</i><b>→</b><i>x₂</i><b>→</b><i>x₃</i><b>→</b><i>x₄</i></div>
            <p>第 t 步依赖前一步状态；训练难以在序列维并行。</p>
          </article>
          <article>
            <span>CNN</span>
            <h3>并行，但局部</h3>
            <div className="sequence-line local"><i>x₁</i><i>x₂</i><i>x₃</i><i>x₄</i><em /></div>
            <p>单层感受野有限；需要堆叠多层才能连接远距离位置。</p>
          </article>
          <article className="featured">
            <span>SELF-ATTENTION</span>
            <h3>全局且并行</h3>
            <div className="sequence-line global"><i>qᵢ</i><b>↘</b><i>k₁</i><i>k₂</i><i>k₃</i><i>k₄</i></div>
            <p>一次矩阵乘法得到所有位置对的相关性；代价是 S² 级分数矩阵。</p>
          </article>
        </div>
        <div className="takeaway">
          <Formula tex={String.raw`\text{全局连接}+\text{并行计算}\quad\Longleftrightarrow\quad O(S^2)\text{ 的时间与显存}`}/>
        </div>
      </section>

      <section className="section matrix-section" id="matrix">
        <SectionHeader
          number="02"
          title="只需要一个矩阵乘法规则"
          description="结果矩阵的第 (i,j) 项，等于左矩阵第 i 行与右矩阵第 j 列的点积。"
        />
        <div className="matrix-layout">
          <div className="matrix-equation">
            <Matrix values={matrixA} label="A" activeRow={matrixRow} />
            <b>×</b>
            <Matrix values={matrixB} label="B" activeColumn={matrixColumn} />
            <b>=</b>
            <Matrix
              values={matrixC}
              label="C"
              activeRow={matrixRow}
              activeColumn={matrixColumn}
              onSelect={(row, column) => setSelectedCell([row, column])}
            />
          </div>
          <div className="matrix-explain">
            <span>点击 C 中任意元素</span>
            <Formula block tex={String.raw`C_{ij}=\sum_{r=1}^{k}A_{ir}B_{rj}`} />
            <code>
              {selectedA.map((value, index) => `${value}×${selectedB[index]}`).join(" + ")} = {matrixC[matrixRow][matrixColumn]}
            </code>
            <p><b>形状：</b>[m,k] × [k,n] → [m,n]。QKᵀ 使用的就是同一规则。</p>
          </div>
        </div>
      </section>

      <section className="section attention-section" id="attention">
        <SectionHeader
          number="03"
          title="Scaled Dot-Product Attention"
          description="先看整体 shape，再跟踪一个 Query 位置的完整计算。"
        />

        <AttentionStepsFigure />

        <div className="step-grid">
          <article>
            <b>① 生成 Q/K/V</b>
            <p>每个词 x 乘三个可学习矩阵，得到"身份三件套"：去问、被问、内容。</p>
          </article>
          <article>
            <b>② 点积打分</b>
            <p>q₁ 与所有人的 k 点积，得到相关度分数 α——越像分数越高。</p>
          </article>
          <article>
            <b>③ softmax 变权重</b>
            <p>分数过 softmax，变成加起来=1 的权重，即"注意力分配"。</p>
          </article>
          <article>
            <b>④ 加权求和</b>
            <p>权重去加权所有人的 v，求和得到融合全局信息的 b₁。</p>
          </article>
        </div>

        <figure className="attention-overview" aria-labelledby="attention-flow-title">
          <figcaption>
            <span>FIG 01 · DATA FLOW</span>
            <div>
              <b id="attention-flow-title">一条完整的数据流：X 如何变成 Z</b>
              <small>Q、K 决定“看谁”，V 携带“取回什么”；所有 Query 行在 GPU 上并行计算。</small>
            </div>
          </figcaption>

          <div className="operator-pipeline">
            <article className="pipeline-node input-node">
              <span>INPUT</span>
              <Formula block tex="X" />
              <code>[B, S, d_model]</code>
              <small>Token Embedding + Position</small>
            </article>

            <div className="pipeline-arrow">
              <small>3 组可学习参数</small>
              <b>→</b>
            </div>

            <div className="qkv-bank">
              {[
                ["Q", "查询", String.raw`Q=XW^Q`, "我要找什么"],
                ["K", "键", String.raw`K=XW^K`, "我能被怎样匹配"],
                ["V", "值", String.raw`V=XW^V`, "真正被汇聚的内容"],
              ].map(([symbol, name, formula, note]) => (
                <article className={`qkv-card ${symbol.toLowerCase()}`} key={symbol}>
                  <i>{symbol}</i>
                  <div>
                    <span>{name}</span>
                    <Formula tex={formula} />
                    <small>{note}</small>
                  </div>
                </article>
              ))}
            </div>

            <div className="pipeline-arrow score-arrow">
              <small>Q、K 进入打分支路</small>
              <b>→</b>
            </div>

            <div className="score-engine">
              <header><span>ATTENTION SCORE</span><code>[B,H,Sq,Sk]</code></header>
              <article>
                <i>01</i>
                <div><b>点积相关性</b><Formula tex={String.raw`S=QK^{\mathsf T}`} /></div>
              </article>
              <em>↓</em>
              <article>
                <i>02</i>
                <div><b>缩放并加 Mask</b><Formula tex={String.raw`L=S/\sqrt{d_k}+M`} /></div>
              </article>
              <em>↓</em>
              <article>
                <i>03</i>
                <div><b>沿 Key 维归一化</b><Formula tex={String.raw`A=\operatorname{softmax}(L)`} /></div>
              </article>
            </div>

            <div className="pipeline-arrow output-arrow">
              <small>A 与 V 汇合</small>
              <b>→</b>
            </div>

            <article className="pipeline-node output-node">
              <span>WEIGHTED SUM</span>
              <div className="combine-badges"><i>A</i><b>×</b><i>V</i></div>
              <Formula block tex={String.raw`Z=AV`} />
              <code>[B,H,Sq,dv]</code>
              <small>每行都是 Value 的加权和</small>
            </article>
          </div>

          <div className="pipeline-ledger">
            <div><b>投影</b><code>[B,S,d_model] → [B,H,S,d_head]</code></div>
            <div><b>分数矩阵</b><code>Sq × Sk</code><small>每个 Query 对所有 Key 的权重</small></div>
            <div><b>Softmax 轴</b><code>dim = −1</code><small>每个 Query 行的权重和为 1</small></div>
            <div><b>V 的路径</b><code>不参与打分</code><small>只在最后一步被 A 加权汇聚</small></div>
          </div>
        </figure>

        <div className="attention-lab">
          <div className="lab-controls">
            <div>
              <span>选择 Query 行</span>
              <div className="position-buttons">
                {positions.map((position, index) => (
                  <button
                    type="button"
                    key={position}
                    className={queryIndex === index ? "active" : ""}
                    onClick={() => setQueryIndex(index)}
                  >
                    {position}
                  </button>
                ))}
              </div>
            </div>
            <label className="mask-control">
              <input type="checkbox" checked={causalMask} onChange={(event) => setCausalMask(event.target.checked)} />
              <i />
              <span><b>Causal Mask</b><small>屏蔽 j &gt; i</small></span>
            </label>
          </div>

          <div className="query-trace" aria-label="当前 Query 对所有 Key 的打分与加权路径">
            <div className="trace-source">
              <small>当前 Query</small>
              <b>q<sub>{queryIndex + 1}</sub></b>
              <code>[{attention.q.map((value) => value.toFixed(1)).join(", ")}]</code>
            </div>
            <div className="trace-arrow"><span>同时点积</span><b>→</b></div>
            <div className="trace-branches">
              {positions.map((position, index) => (
                <article className={!Number.isFinite(attention.scaled[index]) ? "masked" : ""} key={position}>
                  <header><b>k<sub>{index + 1}</sub></b><small>{position}</small></header>
                  <code>{Number.isFinite(attention.scaled[index]) ? attention.scaled[index].toFixed(2) : "−∞"}</code>
                  <i><em style={{ width: `${attention.weights[index] * 100}%` }} /></i>
                  <strong>{(attention.weights[index] * 100).toFixed(1)}%</strong>
                  <span>× v<sub>{index + 1}</sub></span>
                </article>
              ))}
            </div>
            <div className="trace-arrow"><span>加权求和</span><b>→</b></div>
            <div className="trace-result">
              <small>输出</small>
              <b>z<sub>{queryIndex + 1}</sub></b>
              <code>[{attention.output.map((value) => value.toFixed(3)).join(", ")}]</code>
            </div>
          </div>

          <div className="lab-steps">
            <article>
              <span>1 · LOGITS</span>
              <Formula block tex={String.raw`s_{ij}=q_i k_j^{\mathsf T}`} />
              <div className="number-list">
                {attention.logits.map((value, index) => <div key={positions[index]}><b>{positions[index]}</b><code>{value.toFixed(2)}</code></div>)}
              </div>
            </article>
            <article>
              <span>2 · SCALE + MASK</span>
              <Formula block tex={String.raw`\ell_{ij}=\frac{s_{ij}}{\sqrt{d_k}}+M_{ij}`} />
              <div className="number-list">
                {attention.scaled.map((value, index) => <div className={!Number.isFinite(value) ? "masked" : ""} key={positions[index]}><b>{positions[index]}</b><code>{Number.isFinite(value) ? value.toFixed(2) : "−∞"}</code></div>)}
              </div>
            </article>
            <article>
              <span>3 · SOFTMAX</span>
              <Formula block tex={String.raw`\alpha_{ij}=\frac{e^{\ell_{ij}}}{\sum_t e^{\ell_{it}}}`} />
              <div className="weight-list">
                {attention.weights.map((value, index) => <div key={positions[index]}><b>{positions[index]}</b><i><em style={{ width: `${value * 100}%` }} /></i><code>{(value * 100).toFixed(1)}%</code></div>)}
              </div>
            </article>
            <article className="output-step">
              <span>4 · OUTPUT</span>
              <Formula block tex={String.raw`z_i=\sum_j\alpha_{ij}v_j`} />
              <div className="output-vector">
                {attention.output.map((value, index) => <b key={index}>{value.toFixed(3)}</b>)}
              </div>
              <small>输出维度保持 dᵥ</small>
            </article>
          </div>
        </div>

        <div className="scale-note">
          <Formula tex={String.raw`\operatorname{Var}(q_i k_j^{\mathsf T})\approx d_k\quad\Rightarrow\quad \operatorname{Var}\!\left(\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}\right)\approx1`} />
          <p>除以 √dₖ 防止维度增大时 logits 过大、Softmax 过饱和。Mask 必须在 Softmax 前加入。</p>
        </div>
      </section>

      <section className="section multihead-section" id="multihead">
        <SectionHeader
          number="04"
          title="Multi-Head Attention"
          description="多个头在不同子空间独立计算，再沿特征维拼接。"
        />

        <figure className="multihead-pipeline" aria-labelledby="multihead-flow-title">
          <figcaption>
            <span>FIG 02 · SPLIT / PARALLEL / CONCAT</span>
            <div>
              <b id="multihead-flow-title">多头不是重复计算：它把特征维拆成多个子空间</b>
              <small>以 d_model=512、h=8 为例，每个 Head 只处理 d_head=64，最后再恢复到 512 维。</small>
            </div>
          </figcaption>
          <div className="multihead-flow">
            <article className="mh-node mh-input">
              <span>输入 X</span>
              <b>d_model = 512</b>
              <code>[B,S,512]</code>
            </article>
            <div className="mh-arrow"><small>WQ / WK / WV</small><b>→</b></div>
            <div className="mh-head-bank">
              <header>
                <span>reshape + transpose</span>
                <code>[B,S,512] → [B,8,S,64]</code>
              </header>
              <div>
                {[
                  ["HEAD 1", "句法/长程"],
                  ["HEAD 2", "局部邻域"],
                  ["···", "并行子空间"],
                  ["HEAD 8", "全局汇聚"],
                ].map(([name, role], index) => (
                  <article key={name}>
                    <i>{index === 2 ? "···" : index + 1}</i>
                    <b>{name}</b>
                    <Formula tex={String.raw`A_rV_r`} />
                    <small>{role}</small>
                    <code>[B,S,64]</code>
                  </article>
                ))}
              </div>
            </div>
            <div className="mh-arrow"><small>Concat heads</small><b>→</b></div>
            <article className="mh-node mh-concat">
              <span>拼接</span>
              <b>8 × 64 = 512</b>
              <code>[B,S,512]</code>
            </article>
            <div className="mh-arrow"><small>输出投影 WO</small><b>→</b></div>
            <article className="mh-node mh-output">
              <span>MultiHead 输出</span>
              <b>混合各头信息</b>
              <code>[B,S,512]</code>
            </article>
          </div>
          <div className="multihead-rules">
            <span><b>拆头发生在特征维：</b>S 不变，512 → 8 × 64</span>
            <span><b>各头参数独立：</b>每个 Head 有自己的 WQ、WK、WV</span>
            <span><b>WO 负责融合：</b>Concat 只拼接，WO 才学习跨 Head 组合</span>
          </div>
        </figure>

        <div className="multihead-formula">
          <Formula block tex={String.raw`\operatorname{head}_r=\operatorname{Attention}(QW_r^Q,KW_r^K,VW_r^V)`} />
          <Formula block tex={String.raw`\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O`} />
        </div>
        <div className="head-layout">
          <div className={`heatmap ${attentionHeads[headIndex].color}`}>
            <div className="heatmap-top"><span />{positions.map((position) => <b key={position}>{position}</b>)}</div>
            <div className="heatmap-main">
              <div className="heatmap-side">{positions.map((position) => <b key={position}>{position}</b>)}</div>
              <div className="heatmap-cells">
                {attentionHeads[headIndex].matrix.flatMap((row, rowIndex) =>
                  row.map((value, columnIndex) => (
                    <div
                      key={`${rowIndex}-${columnIndex}`}
                      style={{ "--heat": `${Math.round(value * 100)}%` } as React.CSSProperties}
                      title={`${positions[rowIndex]} → ${positions[columnIndex]}: ${(value * 100).toFixed(0)}%`}
                    >
                      {(value * 100).toFixed(0)}
                    </div>
                  )),
                )}
              </div>
            </div>
            <small>Query 行 × Key 列</small>
          </div>
          <div className="head-copy">
            <div className="head-tabs">
              {attentionHeads.map((head, index) => (
                <button type="button" className={headIndex === index ? "active" : ""} onClick={() => setHeadIndex(index)} key={head.name}>
                  <span>{head.name}</span><b>{head.role}</b>
                </button>
              ))}
            </div>
            <p>{attentionHeads[headIndex].note}</p>
            <div className="head-shapes">
              <span><b>输入</b>[B,S,d_model]</span><i>→</i>
              <span><b>拆头</b>[B,h,S,d_head]</span><i>→</i>
              <span><b>拼接</b>[B,S,d_model]</span>
            </div>
            <aside>这些模式仅用于说明“不同子空间可以学习不同关系”，不能假设每个真实 Head 都有唯一、稳定的语言学含义。</aside>
          </div>
        </div>
      </section>

      <section className="section transformer-section" id="transformer">
        <SectionHeader
          number="05"
          title="Attention 如何组成 Transformer"
          description="原始模型是 Encoder–Decoder；每层还包含残差、归一化与 FFN。"
        />
        <TransformerFigure />
        <figure className="classic-transformer" aria-labelledby="classic-transformer-title">
          <figcaption>
            <span>FIG 03 · ORIGINAL TRANSFORMER</span>
            <div>
              <b id="classic-transformer-title">经典 Transformer Encoder–Decoder 全结构</b>
              <small>按 2017 原论文 Figure 1 重绘。箭头从下向上：左侧编码整句，右侧逐 Token 解码。</small>
            </div>
          </figcaption>

          <div className="architecture-map">
            <div className="arch-column encoder-column">
              <header className="arch-column-title"><span>ENCODER</span><b>理解输入序列</b></header>
              <div className="arch-output memory-output">
                <span>ENCODER MEMORY</span>
                <b>上下文化表示</b>
                <code>[B,S_src,d_model]</code>
              </div>
              <div className="flow-up">↑</div>
              <div className="arch-repeat">
                <span className="repeat-badge">N×</span>
                <div className="residual-unit">
                  <div className="norm-box"><b>Add &amp; Norm</b><small>LayerNorm(x + sublayer(x))</small></div>
                  <div className="unit-arrow">↑ <span>残差旁路</span></div>
                  <div className="arch-op ffn-op"><b>Position-wise FFN</b><small>d_model → d_ff → d_model</small></div>
                </div>
                <div className="flow-up compact">↑</div>
                <div className="residual-unit">
                  <div className="norm-box"><b>Add &amp; Norm</b><small>保持 shape 不变</small></div>
                  <div className="unit-arrow">↑ <span>残差旁路</span></div>
                  <div className="arch-op self-op">
                    <b>Multi-Head Self-Attention</b>
                    <small>每个位置可看见全部源序列</small>
                    <div className="source-badges"><i>Q = X</i><i>K = X</i><i>V = X</i></div>
                  </div>
                </div>
              </div>
              <div className="flow-up">↑</div>
              <div className="position-mix"><span>Input Embedding</span><i>＋</i><span>Positional Encoding</span></div>
              <div className="flow-up compact">↑</div>
              <div className="token-strip"><span>源序列</span><b>机器</b><b>学习</b><b>很</b><b>有趣</b></div>
            </div>

            <div className="memory-bus">
              <span>跨注意力数据源</span>
              <div><b>K、V</b><i>→</i></div>
              <code>Encoder Memory</code>
              <small>Decoder 的 Query 去检索源序列</small>
            </div>

            <div className="arch-column decoder-column">
              <header className="arch-column-title"><span>DECODER</span><b>自回归生成</b></header>
              <div className="arch-output probability-output">
                <span>OUTPUT PROBABILITIES</span>
                <b>下一个 Token 的概率</b>
                <code>[B,S_tgt,vocab]</code>
              </div>
              <div className="flow-up compact">↑</div>
              <div className="head-box softmax-box">Softmax <small>沿 vocab 维归一化</small></div>
              <div className="flow-up compact">↑</div>
              <div className="head-box linear-box">Linear <small>d_model → vocab_size</small></div>
              <div className="flow-up">↑</div>
              <div className="arch-repeat decoder-repeat">
                <span className="repeat-badge">N×</span>
                <div className="residual-unit">
                  <div className="norm-box"><b>Add &amp; Norm</b><small>第三条残差支路</small></div>
                  <div className="unit-arrow">↑ <span>残差旁路</span></div>
                  <div className="arch-op ffn-op"><b>Position-wise FFN</b><small>逐位置、共享参数</small></div>
                </div>
                <div className="flow-up compact">↑</div>
                <div className="residual-unit cross-unit">
                  <div className="norm-box"><b>Add &amp; Norm</b><small>第二条残差支路</small></div>
                  <div className="unit-arrow">↑ <span>残差旁路</span></div>
                  <div className="arch-op cross-op">
                    <b>Encoder–Decoder Attention</b>
                    <small>用目标侧状态检索源序列 Memory</small>
                    <div className="source-badges"><i>Q = Decoder</i><i>K,V = Encoder</i></div>
                  </div>
                </div>
                <div className="flow-up compact">↑</div>
                <div className="residual-unit">
                  <div className="norm-box"><b>Add &amp; Norm</b><small>第一条残差支路</small></div>
                  <div className="unit-arrow">↑ <span>残差旁路</span></div>
                  <div className="arch-op masked-op">
                    <b>Masked Multi-Head Self-Attention</b>
                    <small>上三角 Mask：位置 i 不能读取 j &gt; i</small>
                    <div className="source-badges"><i>Q = K = V = Decoder</i><i>causal mask</i></div>
                  </div>
                </div>
              </div>
              <div className="flow-up">↑</div>
              <div className="position-mix"><span>Output Embedding</span><i>＋</i><span>Positional Encoding</span></div>
              <div className="flow-up compact">↑</div>
              <div className="token-strip"><span>右移目标序列</span><b>&lt;BOS&gt;</b><b>machine</b><b>learning</b></div>
            </div>
          </div>

          <div className="architecture-equations">
            <div><span>Encoder 子层</span><Formula tex={String.raw`y=\operatorname{LN}(x+\operatorname{Sublayer}(x))`} /></div>
            <div><span>位置注入</span><Formula tex={String.raw`x_0=E_{token}+E_{pos}`} /></div>
            <div><span>Cross-Attention</span><Formula tex={String.raw`Q=H_{dec},\quad K=V=H_{enc}`} /></div>
          </div>

          <div className="architecture-notes">
            <article><b>为什么 Encoder 能并行</b><p>源序列已知，所有位置的 Q/K/V 可以一次生成；每层内部没有时间步依赖。</p></article>
            <article><b>为什么推理解码仍然串行</b><p>第 t 个 Token 要依赖已经生成的 1…t−1；训练时可用右移目标序列和 Causal Mask 并行。</p></article>
            <article><b>经典图的 Norm 位置</b><p>原论文是 Post-LN；很多现代大模型改用 Pre-LN，但 Attention 数据流不变。</p></article>
            <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">结构来源：Attention Is All You Need · Figure 1 ↗</a>
          </div>
        </figure>
      </section>

      <section className="section code-section" id="code">
        <SectionHeader
          number="06"
          title="经典代码与算子测试"
          description="代码逐行对应公式；测试优先锁住数值、Mask 与 shape。"
        />
        <div className="code-test-layout">
          <div className="code-panel">
            <div className="code-tabs">
              {(Object.keys(codeSamples) as Array<keyof typeof codeSamples>).map((key) => (
                <button type="button" className={codeTab === key ? "active" : ""} onClick={() => setCodeTab(key)} key={key}>{codeSamples[key].label}</button>
              ))}
              <button type="button" className="copy-button" onClick={copyCode}>{copied ? "已复制 ✓" : "复制"}</button>
            </div>
            <div className="code-title"><span>attention.py</span><b>{activeCode.title}</b></div>
            <pre>{activeCode.code.split("\n").map((line, index) => <code key={index}><span>{String(index + 1).padStart(2, "0")}</span>{line || " "}</code>)}</pre>
          </div>
          <div className="test-panel">
            <header><span>OPERATOR CHECKLIST</span><b>P0 → P1</b></header>
            {[
              ["01", "参考值", "与朴素 float64 实现逐元素对齐"],
              ["02", "Shape", "覆盖 B/H/S/D、S_q ≠ S_k、S = 1"],
              ["03", "Mask", "被屏蔽权重为 0，因果位置无泄漏"],
              ["04", "Softmax", "每个 Query 行的权重和为 1"],
              ["05", "数值", "FP16/BF16、极值输入、无 NaN/Inf"],
              ["06", "梯度", "gradcheck 与 Q/K/V 参数梯度"],
            ].map(([number, title, detail]) => (
              <div className="test-item" key={number}><i>{number}</i><span><b>{title}</b><small>{detail}</small></span></div>
            ))}
            <aside><b>高频错误</b>Softmax 轴写成 −2；Mask 放在 Softmax 后；漏除 √dₖ；transpose 后直接 view。</aside>
          </div>
        </div>
        <div className="closing-line">
          <Formula tex={String.raw`X\xrightarrow{W^Q,W^K,W^V}Q,K,V\xrightarrow{\,\operatorname{softmax}(QK^{\mathsf T}/\sqrt{d_k})V\,}Z\xrightarrow{\text{Multi-Head + FFN}}\text{Transformer}`} />
        </div>
      </section>

      <footer>
        <div><i>A</i><span><b>ATTENTION · TECHNICAL BRIEF</b><small>从矩阵到算子测试</small></span></div>
        <nav>
          <a href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">原论文</a>
          <a href="https://nlp.seas.harvard.edu/annotated-transformer/" target="_blank" rel="noreferrer">Annotated Transformer</a>
          <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html" target="_blank" rel="noreferrer">PyTorch 文档</a>
        </nav>
        <span>CORE OPERATOR</span>
      </footer>
    </main>
  );
}
