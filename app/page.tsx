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
      <Tag className={`math ${block ? "math-block" : "math-inline"} math-error ${className}`}>
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

/* ---------- 矩阵乘法演示数据 ---------- */
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
  matrixB[0].map((_, c) =>
    row.reduce((s, v, i) => s + v * matrixB[i][c], 0),
  ),
);

/* ---------- Attention 演示数据 ---------- */
const words = ["The", "cat", "sat", "here"];
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

function softmax(values: number[]) {
  const finite = values.filter(Number.isFinite);
  const mx = Math.max(...finite);
  const ex = values.map((v) => (Number.isFinite(v) ? Math.exp(v - mx) : 0));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / sum);
}

/* ---------- 多头热力图数据 ---------- */
const heads = [
  {
    name: "Head 1 · 长程",
    note: "权重大量出现在非对角线位置 → 擅长捕捉长距离依赖。",
    matrix: [
      [0.58, 0.12, 0.22, 0.08],
      [0.18, 0.16, 0.12, 0.54],
      [0.66, 0.08, 0.20, 0.06],
      [0.14, 0.46, 0.09, 0.31],
    ],
  },
  {
    name: "Head 2 · 局部",
    note: "权重集中在主对角线附近 → 只关注相邻词。",
    matrix: [
      [0.62, 0.27, 0.07, 0.04],
      [0.24, 0.48, 0.22, 0.06],
      [0.07, 0.24, 0.48, 0.21],
      [0.03, 0.08, 0.30, 0.59],
    ],
  },
  {
    name: "Head h · 全局",
    note: "权重分布很平均 → 保留全局统计，近似看整句。",
    matrix: [
      [0.28, 0.24, 0.25, 0.23],
      [0.23, 0.29, 0.22, 0.26],
      [0.27, 0.20, 0.30, 0.23],
      [0.22, 0.27, 0.21, 0.30],
    ],
  },
];

/* ============================================================
 * SVG 图：向量级 Self-Attention 四步推导（对应 PDF 图4-9）
 * ============================================================ */
function FigAttentionSteps() {
  const cell = (x: number, y: number, w: number, fill: string, stroke: string, label: string, fc: string) => (
    <g key={`c-${x}-${y}-${label}`}>
      <rect x={x} y={y} width={w} height={30} rx={6} fill={fill} stroke={stroke} />
      <text x={x + w / 2} y={y + 19} textAnchor="middle" fill={fc} fontFamily="var(--mono)" fontSize="12">{label}</text>
    </g>
  );
  return (
    <div className="fig">
      <svg viewBox="0 0 920 420" width="920" role="img" aria-label="self-attention 向量级四步推导">
        <defs>
          <marker id="ah-s" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L7,4 L0,8 z" fill="#6e7aab" />
          </marker>
        </defs>

        <text x="10" y="24" fill="#c8d4ff" fontSize="14" fontWeight="700">① 线性变换：每个 x 得到 q / k / v</text>
        {cell(20, 70, 40, "var(--panel-3)", "var(--hairline)", "x₁", "var(--t2)")}
        {cell(20, 120, 40, "var(--panel-3)", "var(--hairline)", "x₂", "var(--t2)")}
        {cell(20, 170, 40, "var(--panel-3)", "var(--hairline)", "x₃", "var(--t2)")}
        {cell(20, 220, 40, "var(--panel-3)", "var(--hairline)", "x₄", "var(--t2)")}
        {cell(150, 70, 44, "var(--q-soft)", "var(--q)", "q₁", "var(--q)")}
        {cell(150, 120, 44, "var(--k-soft)", "var(--k)", "k₁", "var(--k)")}
        {cell(150, 170, 44, "var(--k-soft)", "var(--k)", "k₂", "var(--k)")}
        {cell(150, 220, 44, "var(--k-soft)", "var(--k)", "k₃", "var(--k)")}
        {cell(240, 120, 44, "var(--v-soft)", "var(--v)", "v₁", "var(--v)")}
        {cell(240, 170, 44, "var(--v-soft)", "var(--v)", "v₂", "var(--v)")}
        {cell(240, 220, 44, "var(--v-soft)", "var(--v)", "v₃", "var(--v)")}
        <text x="172" y="62" fill="var(--q)" fontSize="10.5" textAnchor="middle">Wᵠ·x</text>
        <text x="172" y="278" fill="var(--k)" fontSize="10.5" textAnchor="middle">Wᵏ·x</text>
        <text x="262" y="278" fill="var(--v)" fontSize="10.5" textAnchor="middle">Wᵛ·x</text>
        <path d="M60,85 H148" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M60,135 H148" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M60,185 H148" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M60,235 H148" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M60,135 H238" stroke="var(--t3)" strokeWidth="1.2" fill="none" strokeDasharray="3 3" markerEnd="url(#ah-s)" />
        <text x="92" y="78" fill="var(--t3)" fontSize="10">×Wᵠ</text>

        <text x="360" y="24" fill="#c8d4ff" fontSize="14" fontWeight="700">② q₁ 与每个 k 点积 → 相关性 α</text>
        {cell(370, 120, 56, "var(--panel-3)", "var(--hairline)", "α₁,₁", "var(--att)")}
        {cell(370, 160, 56, "var(--panel-3)", "var(--hairline)", "α₁,₂", "var(--att)")}
        {cell(370, 200, 56, "var(--panel-3)", "var(--hairline)", "α₁,₃", "var(--att)")}
        {cell(370, 240, 56, "var(--panel-3)", "var(--hairline)", "α₁,₄", "var(--att)")}
        <text x="398" y="290" fill="var(--t3)" fontSize="10.5" textAnchor="middle">q₁·kⱼ</text>
        <path d="M194,85 C300,85 330,135 368,135" stroke="var(--q)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M194,185 C290,185 320,175 368,175" stroke="var(--k)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M194,235 C300,235 330,215 368,215" stroke="var(--k)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M194,135 C300,135 330,255 368,255" stroke="var(--k)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />

        <text x="470" y="24" fill="#c8d4ff" fontSize="14" fontWeight="700">③ softmax → 权重（和=1）</text>
        {cell(490, 120, 60, "var(--att-soft)", "var(--att)", "α̂₁,₁", "var(--att)")}
        {cell(490, 160, 60, "var(--att-soft)", "var(--att)", "α̂₁,₂", "var(--att)")}
        {cell(490, 200, 60, "var(--att-soft)", "var(--att)", "α̂₁,₃", "var(--att)")}
        {cell(490, 240, 60, "var(--att-soft)", "var(--att)", "α̂₁,₄", "var(--att)")}
        <text x="520" y="290" fill="var(--t3)" fontSize="10.5" textAnchor="middle">Σ=1.0</text>
        {[135, 175, 215, 255].map((y) => (
          <path key={`s3-${y}`} d={`M426,${y} H488`} stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        ))}

        <text x="600" y="24" fill="#c8d4ff" fontSize="14" fontWeight="700">④ 权重 × v 求和 → b₁</text>
        {cell(610, 120, 120, "var(--panel-3)", "var(--hairline)", "α̂₁,₁·v₁", "var(--v)")}
        {cell(610, 160, 120, "var(--panel-3)", "var(--hairline)", "α̂₁,₂·v₂", "var(--v)")}
        {cell(610, 200, 120, "var(--panel-3)", "var(--hairline)", "α̂₁,₃·v₃", "var(--v)")}
        {cell(610, 240, 120, "var(--panel-3)", "var(--hairline)", "α̂₁,₄·v₄", "var(--v)")}
        {[135, 175, 215, 255].map((y) => (
          <path key={`s4-${y}`} d={`M550,${y} H608`} stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        ))}
        <path d="M284,135 C450,135 500,135 608,135" stroke="var(--v)" strokeWidth="1.2" fill="none" strokeDasharray="3 3" markerEnd="url(#ah-s)" />

        <rect x="780" y="175" width="70" height="40" rx="9" fill="var(--out-soft)" stroke="var(--out)" strokeWidth="1.6" />
        <text x="815" y="201" textAnchor="middle" fill="var(--out)" fontFamily="var(--mono)" fontSize="16" fontWeight="700">b₁</text>
        <path d="M730,135 C760,135 770,180 778,188" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M730,175 H778" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M730,215 C760,215 770,200 778,198" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <path d="M730,255 C760,255 770,210 778,205" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-s)" />
        <text x="815" y="235" fill="var(--out)" fontSize="10.5" textAnchor="middle">= Σ α̂·v</text>
      </svg>
      <div className="fig-cap">图 · 对应 PDF 图 4–9 — 用 q₁ 算出 b₁ 的完整 4 步（b₂ b₃ b₄ 同理并行）</div>
    </div>
  );
}

/* ============================================================
 * SVG 图：矩阵级 Self-Attention（对应 PDF 图10-12）
 * ============================================================ */
function FigAttentionMatrix() {
  return (
    <div className="fig">
      <svg viewBox="0 0 920 340" width="920" role="img" aria-label="self-attention 矩阵级三步">
        <defs>
          <linearGradient id="gq-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--q)" stopOpacity=".5" /><stop offset="1" stopColor="var(--q)" stopOpacity=".1" /></linearGradient>
          <linearGradient id="gk-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--k)" stopOpacity=".5" /><stop offset="1" stopColor="var(--k)" stopOpacity=".1" /></linearGradient>
          <linearGradient id="gv-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--v)" stopOpacity=".5" /><stop offset="1" stopColor="var(--v)" stopOpacity=".1" /></linearGradient>
          <linearGradient id="ga-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--att)" stopOpacity=".55" /><stop offset="1" stopColor="var(--att)" stopOpacity=".12" /></linearGradient>
          <linearGradient id="go-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--out)" stopOpacity=".55" /><stop offset="1" stopColor="var(--out)" stopOpacity=".12" /></linearGradient>
          <marker id="ah-m" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="var(--t3)" /></marker>
        </defs>
        <text x="20" y="24" fill="#c8d4ff" fontSize="14" fontWeight="700">把向量堆成矩阵后，整个过程 = 三次矩阵乘法</text>
        <rect x="20" y="110" width="60" height="120" rx="8" fill="var(--panel-3)" stroke="var(--hairline)" />
        <text x="50" y="175" textAnchor="middle" fill="var(--t1)" fontSize="15" fontWeight="700">I</text>
        <text x="50" y="248" textAnchor="middle" fill="var(--t3)" fontSize="10">(n×d)</text>
        <path d="M80,140 H120" stroke="var(--q)" strokeWidth="1.4" fill="none" markerEnd="url(#ah-m)" />
        <path d="M80,170 H120" stroke="var(--k)" strokeWidth="1.4" fill="none" markerEnd="url(#ah-m)" />
        <path d="M80,200 H120" stroke="var(--v)" strokeWidth="1.4" fill="none" markerEnd="url(#ah-m)" />
        <text x="84" y="133" fill="var(--q)" fontSize="10">×Wᵠ</text>
        <text x="84" y="164" fill="var(--k)" fontSize="10">×Wᵏ</text>
        <text x="84" y="216" fill="var(--v)" fontSize="10">×Wᵛ</text>
        <rect x="120" y="125" width="56" height="34" rx="6" fill="url(#gq-m)" stroke="var(--q)" /><text x="148" y="147" textAnchor="middle" fill="var(--q)" fontWeight="700">Q</text>
        <rect x="120" y="163" width="56" height="34" rx="6" fill="url(#gk-m)" stroke="var(--k)" /><text x="148" y="185" textAnchor="middle" fill="var(--k)" fontWeight="700">K</text>
        <rect x="120" y="201" width="56" height="34" rx="6" fill="url(#gv-m)" stroke="var(--v)" /><text x="148" y="223" textAnchor="middle" fill="var(--v)" fontWeight="700">V</text>
        <text x="215" y="152" fill="var(--t2)" fontSize="12">Q · Kᵀ</text>
        <text x="215" y="170" fill="var(--t3)" fontSize="10">两两相关度</text>
        <path d="M176,152 H210" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <path d="M176,190 H210" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <rect x="295" y="110" width="90" height="120" rx="8" fill="url(#ga-m)" stroke="var(--att)" />
        <text x="340" y="175" textAnchor="middle" fill="var(--att)" fontWeight="700" fontSize="14">A=QKᵀ</text>
        <text x="340" y="248" textAnchor="middle" fill="var(--t3)" fontSize="10">(n×n)</text>
        <path d="M245,158 C270,158 275,175 293,178" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <rect x="420" y="110" width="80" height="120" rx="8" fill="var(--panel-3)" stroke="var(--att)" />
        <text x="460" y="170" textAnchor="middle" fill="var(--att)" fontWeight="700" fontSize="13">softmax</text>
        <text x="460" y="188" textAnchor="middle" fill="var(--att)" fontWeight="700" fontSize="13">按行</text>
        <text x="460" y="248" textAnchor="middle" fill="var(--t3)" fontSize="10">→ 权重 Â</text>
        <path d="M385,170 H418" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <rect x="540" y="110" width="60" height="120" rx="8" fill="url(#ga-m)" stroke="var(--att)" />
        <text x="570" y="175" textAnchor="middle" fill="var(--att)" fontWeight="700" fontSize="14">Â</text>
        <text x="570" y="248" textAnchor="middle" fill="var(--t3)" fontSize="10">(n×n)</text>
        <path d="M500,170 H538" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <path d="M176,228 C400,300 470,300 538,210" stroke="var(--v)" strokeWidth="1.3" fill="none" strokeDasharray="4 3" markerEnd="url(#ah-m)" />
        <rect x="660" y="110" width="70" height="120" rx="8" fill="url(#go-m)" stroke="var(--out)" />
        <text x="695" y="175" textAnchor="middle" fill="var(--out)" fontWeight="700" fontSize="15">O</text>
        <text x="695" y="248" textAnchor="middle" fill="var(--t3)" fontSize="10">(n×d) 输出</text>
        <path d="M600,170 H658" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <g transform="translate(760,110)">
          <rect x="0" y="0" width="140" height="120" rx="9" fill="var(--bg)" stroke="var(--hairline)" />
          <text x="70" y="24" textAnchor="middle" fill="var(--accent)" fontWeight="700" fontSize="12">整个过程 =</text>
          <text x="12" y="52" fill="var(--t2)" fontSize="11">① I×W → Q,K,V</text>
          <text x="12" y="74" fill="var(--t2)" fontSize="11">② QKᵀ → softmax</text>
          <text x="12" y="96" fill="var(--t2)" fontSize="11">③ Â×V → O</text>
          <text x="12" y="114" fill="var(--t3)" fontSize="9.5">三次矩阵乘法</text>
        </g>
      </svg>
      <div className="fig-cap">图 · 对应 PDF 图 10–12 — 压成"一堆矩阵乘法"，可用 GPU 加速</div>
    </div>
  );
}

/* ============================================================
 * SVG 图：经典 Transformer 论文 Figure 1
 * ============================================================ */
function FigTransformer() {
  const Arrow = ({ d, color = "var(--t3)", dash }: { d: string; color?: string; dash?: string }) => (
    <path d={d} stroke={color} strokeWidth="1.4" fill="none" markerEnd="url(#ah-t)" strokeDasharray={dash} />
  );
  const Box = ({ x, y, w, h, fill, stroke, label, sub, lc, sc }: { x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; lc?: string; sc?: string }) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="9" fill={fill} stroke={stroke} />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -4 : 5)} textAnchor="middle" fill={lc || "var(--t1)"} fontSize="13" fontWeight="700">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fill={sc || "var(--t3)"} fontSize="10">{sub}</text>}
    </g>
  );
  return (
    <div className="fig">
      <svg viewBox="0 0 940 600" width="940" role="img" aria-label="经典 Transformer Encoder-Decoder 结构图">
        <defs>
          <marker id="ah-t" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="var(--t3)" /></marker>
        </defs>
        <text x="250" y="26" textAnchor="middle" fill="var(--att)" fontSize="15" fontWeight="700">Encoder × N（左）</text>
        <text x="710" y="26" textAnchor="middle" fill="var(--out)" fontSize="15" fontWeight="700">Decoder × N（右）</text>

        <Box x={60} y={60} w={120} h={40} fill="var(--panel-3)" stroke="var(--hairline)" label="Input Embedding" lc="var(--t2)" />
        <circle cx="200" cy="80" r="15" fill="var(--bg)" stroke="var(--k)" />
        <text x="200" y="85" textAnchor="middle" fill="var(--k)" fontSize="15">+</text>
        <text x="200" y="50" textAnchor="middle" fill="var(--k)" fontSize="10">Positional</text>
        <text x="200" y="62" textAnchor="middle" fill="var(--k)" fontSize="10">Encoding</text>
        <Arrow d="M180,80 H186" />
        <rect x="80" y="125" width="290" height="345" rx="14" fill="none" stroke="var(--hairline)" strokeDasharray="5 5" />
        <text x="225" y="120" textAnchor="middle" fill="var(--t3)" fontSize="11">Encoder Layer (堆叠 N 次)</text>
        <Box x={120} y={150} w={210} h={54} fill="var(--att-soft)" stroke="var(--att)" label="Multi-Head Self-Attention" sub="本节讲的核心算子" lc="var(--att)" sc="var(--t3)" />
        <Box x={150} y={222} w={150} h={36} fill="var(--panel-3)" stroke="var(--hairline)" label="Add &amp; Norm" lc="var(--t2)" />
        <Box x={120} y={278} w={210} h={50} fill="var(--v-soft)" stroke="var(--v)" label="Feed-Forward Network" sub="两层 MLP（逐位置作用）" lc="var(--v)" sc="var(--t3)" />
        <Box x={150} y={346} w={150} h={36} fill="var(--panel-3)" stroke="var(--hairline)" label="Add &amp; Norm" lc="var(--t2)" />
        <Arrow d="M225,204 V218" /><Arrow d="M225,258 V274" /><Arrow d="M225,328 V342" />
        <Arrow d="M120,170 H100 V410 H225" color="var(--q)" dash="4 3" />
        <text x="92" y="395" fill="var(--q)" fontSize="9">残差</text>
        <Box x={150} y={405} w={150} h={34} fill="var(--out-soft)" stroke="var(--out)" label="编码器输出（K, V）" lc="var(--out)" />
        <Arrow d="M215,100 V150" />

        <Box x={560} y={60} w={120} h={40} fill="var(--panel-3)" stroke="var(--hairline)" label="Output Embedding" lc="var(--t2)" />
        <circle cx="700" cy="80" r="15" fill="var(--bg)" stroke="var(--k)" />
        <text x="700" y="85" textAnchor="middle" fill="var(--k)" fontSize="15">+</text>
        <text x="700" y="50" textAnchor="middle" fill="var(--k)" fontSize="10">Positional</text>
        <text x="700" y="62" textAnchor="middle" fill="var(--k)" fontSize="10">Encoding</text>
        <Arrow d="M680,80 H686" />
        <rect x="540" y="125" width="310" height="345" rx="14" fill="none" stroke="var(--hairline)" strokeDasharray="5 5" />
        <text x="695" y="120" textAnchor="middle" fill="var(--t3)" fontSize="11">Decoder Layer (堆叠 N 次)</text>
        <Box x={575} y={150} w={240} h={50} fill="var(--q-soft)" stroke="var(--q)" label="Masked Multi-Head Attention" sub="只能看过去（屏蔽未来位）" lc="var(--q)" sc="var(--t3)" />
        <Box x={620} y={216} w={150} h={34} fill="var(--panel-3)" stroke="var(--hairline)" label="Add &amp; Norm" lc="var(--t2)" />
        <Box x={575} y={268} w={240} h={50} fill="var(--att-soft)" stroke="var(--att)" label="Cross Attention（编码-解码交互）" sub="Q 来自解码器，K,V 来自编码器" lc="var(--att)" sc="var(--t3)" />
        <Box x={620} y={334} w={150} h={34} fill="var(--panel-3)" stroke="var(--hairline)" label="Add &amp; Norm" lc="var(--t2)" />
        <Box x={575} y={386} w={240} h={44} fill="var(--v-soft)" stroke="var(--v)" label="Feed-Forward Network" lc="var(--v)" />
        <Box x={620} y={442} w={150} h={32} fill="var(--panel-3)" stroke="var(--hairline)" label="Add &amp; Norm" lc="var(--t2)" />
        <Arrow d="M695,200 V212" /><Arrow d="M695,250 V264" /><Arrow d="M695,318 V330" /><Arrow d="M695,368 V382" /><Arrow d="M695,430 V438" />
        <Arrow d="M300,422 C440,422 460,293 573,293" color="var(--k)" dash="4 3" />
        <text x="430" y="360" fill="var(--k)" fontSize="10">编码器 K, V 传过来</text>
        <Arrow d="M695,474 V492" />
        <Box x={600} y={495} w={190} h={34} fill="var(--k-soft)" stroke="var(--k)" label="Linear → Softmax → 词概率" lc="var(--k)" />
        <Arrow d="M685,100 V150" />

        <g transform="translate(80,555)">
          <rect x="0" y="0" width="14" height="14" rx="3" fill="var(--att-soft)" stroke="var(--att)" /><text x="20" y="12" fill="var(--t3)" fontSize="11">Attention</text>
          <rect x="110" y="0" width="14" height="14" rx="3" fill="var(--v-soft)" stroke="var(--v)" /><text x="130" y="12" fill="var(--t3)" fontSize="11">FFN</text>
          <rect x="190" y="0" width="14" height="14" rx="3" fill="var(--panel-3)" stroke="var(--hairline)" /><text x="210" y="12" fill="var(--t3)" fontSize="11">Add&amp;Norm</text>
          <rect x="310" y="0" width="14" height="14" rx="3" fill="var(--bg)" stroke="var(--k)" /><text x="330" y="12" fill="var(--t3)" fontSize="11">位置编码</text>
        </g>
      </svg>
      <div className="fig-cap">图 · 论文 Figure 1 重绘 — Attention 在 Encoder/Decoder 中共出现三次，是同一算子</div>
    </div>
  );
}

/* ============================================================
 * SVG 图：RNN vs Self-Attention（对应 PDF 图1-3）
 * ============================================================ */
function FigRnnVsAtt() {
  return (
    <div className="fig">
      <svg viewBox="0 0 900 250" width="900" role="img" aria-label="RNN 与 Self-Attention 对比">
        <defs><marker id="ah-r" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 z" fill="var(--t3)" /></marker></defs>
        <text x="40" y="40" fill="#c8d4ff" fontSize="15" fontWeight="700">RNN：必须串行，算不出 b₄ 就用不了 b₃</text>
        {[80, 180, 280, 380].map((cx, i) => (
          <g key={`r-${cx}`}>
            <circle cx={cx} cy={130} r="20" fill="var(--panel-3)" stroke="var(--q)" />
            <text x={cx} y={135} textAnchor="middle" fill="var(--q)" fontSize="13">x{i + 1}</text>
            <rect x={cx - 20} y={185} width="40" height="32" rx="6" fill="var(--panel-3)" stroke="var(--hairline)" />
            <text x={cx} y={205} textAnchor="middle" fill="var(--t2)" fontSize="12">b{i + 1}</text>
            <path d={`M${cx},150 V182`} stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
          </g>
        ))}
        <path d="M100,205 H155" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
        <path d="M200,205 H255" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
        <path d="M300,205 H355" stroke="var(--t3)" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
        <text x="80" y="245" fill="#ff7a7a" fontSize="11">❌ 无法并行：b₂ 要等 b₁</text>

        <text x="510" y="40" fill="#c8d4ff" fontSize="15" fontWeight="700">Self-Attention：所有位置同时互相看见，可并行</text>
        {[540, 640, 740, 840].map((cx, i) => (
          <g key={`a-${cx}`}>
            <circle cx={cx} cy={80} r="20" fill="var(--panel-3)" stroke="var(--v)" />
            <text x={cx} y={85} textAnchor="middle" fill="var(--v)" fontSize="13">x{i + 1}</text>
            <rect x={cx - 20} y={160} width="40" height="32" rx="6" fill="var(--panel-3)" stroke="var(--hairline)" />
            <text x={cx} y={180} textAnchor="middle" fill="var(--out)" fontSize="12">b{i + 1}</text>
          </g>
        ))}
        <g stroke="var(--att)" strokeWidth="1" opacity="0.5" fill="none">
          {[540, 640, 740, 840].map((cx) =>
            [540, 640, 740, 840].map((tx) => <path key={`l-${cx}-${tx}`} d={`M${cx},100 Q${(cx + tx) / 2},130 ${tx},160`} />),
          )}
        </g>
        <text x="640" y="225" fill="var(--ok)" fontSize="11">✅ 每个输出都看完整序列，且各 b 互不依赖、一起算</text>
      </svg>
      <div className="fig-cap">图 · 对应 PDF 图 1·2·3 — 处理序列数据的两种范式</div>
    </div>
  );
}

/* ---------- 章节头 ---------- */
function SecHead({ idx, title }: { idx: string; title: string }) {
  return (
    <div className="sec-head">
      <span className="idx">{idx}</span>
      <h2>{title}</h2>
    </div>
  );
}

export default function Home() {
  const [scroll, setScroll] = useState(0);
  const [selCell, setSelCell] = useState<[number, number]>([0, 0]);
  const [qIdx, setQIdx] = useState(2);
  const [headIdx, setHeadIdx] = useState(0);
  const [activeNav, setActiveNav] = useState("s0");

  useEffect(() => {
    const upd = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      setScroll(h > 0 ? (window.scrollY / h) * 100 : 0);
    };
    upd();
    window.addEventListener("scroll", upd, { passive: true });
    return () => window.removeEventListener("scroll", upd);
  }, []);

  useEffect(() => {
    const ids = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"];
    const onScroll = () => {
      const y = window.scrollY + 130;
      let cur = ids[0];
      ids.forEach((id) => {
        const el = document.getElementById(id);
        if (el && el.offsetTop <= y) cur = id;
      });
      setActiveNav(cur);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const attn = useMemo(() => {
    const q = queries[qIdx];
    const logits = keys.map((k) => q.reduce((s, c, i) => s + c * k[i], 0));
    const scaled = logits.map((v) => v / Math.sqrt(q.length));
    const weights = softmax(scaled);
    const output = values[0].map((_, d) => weights.reduce((s, w, i) => s + w * values[i][d], 0));
    return { q, logits, scaled, weights, output };
  }, [qIdx]);

  const [mr, mc] = selCell;
  const rowA = matrixA[mr];
  const colB = matrixB.map((r) => r[mc]);

  const navItems = [
    ["s0", "为什么重要"],
    ["s1", "矩阵乘法"],
    ["s2", "RNN 痛点"],
    ["s3", "向量级 Attention"],
    ["s4", "矩阵级 Attention"],
    ["s5", "缩放点积公式"],
    ["s6", "多头注意力"],
    ["s7", "经典 Transformer"],
    ["s8", "代码与测试"],
  ];

  return (
    <>
      <div className="progress" style={{ width: `${scroll}%` }} />
      <nav className="sidenav">
        <div className="brand">
          <span className="glyph">A</span>
          <b>Attention</b>
        </div>
        <div className="sub">Transformer 核心算子<br />面向零基础讲解</div>
        <ol>
          {navItems.map(([id, label]) => (
            <li key={id}>
              <a href={`#${id}`} className={activeNav === id ? "active" : ""}>{label}</a>
            </li>
          ))}
        </ol>
        <div className="legend">
          <span><i style={{ background: "var(--q)" }} />Query 查询</span>
          <span><i style={{ background: "var(--k)" }} />Key 键</span>
          <span><i style={{ background: "var(--v)" }} />Value 值</span>
          <span><i style={{ background: "var(--att)" }} />Attention</span>
          <span><i style={{ background: "var(--out)" }} />Output</span>
        </div>
      </nav>

      <main className="main">
        <div className="wrap">

          {/* ===== Hero ===== */}
          <section className="hero" id="s0">
            <span className="kicker">// Operator Deep-Dive</span>
            <h1>Attention 算子<br /><em>到底在算什么？</em></h1>
            <p className="lead">从「矩阵乘法怎么乘」一路讲到经典 Transformer 全景图。每个公式配数值演示，每个结构都用 SVG 逐格重绘——这张图你今天一定能看懂。</p>
            <div className="chips">
              <span>零基础起点</span>
              <span><b>Q · K · V</b> 全程配色一致</span>
              <span>含 PyTorch 经典代码 + 算子测试要点</span>
            </div>

            <div className="hero-card">
              <div className="tag">CORE EQUATION</div>
              <div className="eq">
                <Formula block tex={String.raw`\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V`} />
              </div>
              <div className="shapes">
                <b className="q">Q [B,H,Sₚ,D]</b>
                <b className="k">K [B,H,Sₖ,D]</b>
                <b className="v">V [B,H,Sₖ,D]</b>
              </div>
            </div>

            <div className="grid2" style={{ marginTop: 28 }}>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>这个算子为什么重要</h3>
                <p className="t3">它是几乎所有现代大模型（GPT / LLaMA / Claude / 文生图、文生视频）的共同骨架。掌握它，等于拿到理解整个生成式 AI 的钥匙。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>算子视角的一句话</h3>
                <p className="t3">Attention 的核心就是<b style={{ color: "var(--t1)" }}>三个矩阵乘法 + 一个 softmax</b>。整堂课的目标，是让你在脑子里把这句话可视化出来。</p>
              </div>
            </div>
          </section>

          {/* ===== 矩阵乘法 ===== */}
          <section className="section" id="s1">
            <SecHead idx="01" title="零基础热身：矩阵乘法到底怎么乘" />
            <p className="sec-lead">Attention 的全部运算都是矩阵乘法。先抛开深度学习，用具体数字搞明白规则——<b style={{ color: "var(--out)" }}>点一下右边结果矩阵的任意格子</b>，左边高亮参与计算的行与列。</p>
            <div className="mbox">
              <div className="mcol">
                <div className="mname"><b>A</b> (2×3)</div>
                <table className="mtable">
                  <tbody>
                    {matrixA.map((row, r) => (
                      <tr key={r}>{row.map((v, c) => (
                        <td key={c}><div className={`mcell ${r === mr ? "hl-row" : ""}`}>{v}</div></td>
                      ))}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <span className="msign">×</span>
              <div className="mcol">
                <div className="mname"><b>B</b> (3×2)</div>
                <table className="mtable">
                  <tbody>
                    {matrixB.map((row, r) => (
                      <tr key={r}>{row.map((v, c) => (
                        <td key={c}><div className={`mcell ${c === mc ? "hl-col" : ""}`}>{v}</div></td>
                      ))}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <span className="msign">=</span>
              <div className="mcol">
                <div className="mname"><b>C</b> (2×2) ← 点击</div>
                <table className="mtable">
                  <tbody>
                    {matrixC.map((row, r) => (
                      <tr key={r}>{row.map((v, c) => (
                        <td key={c}>
                          <button
                            className={`mcell res ${r === mr && c === mc ? "hl-res" : ""}`}
                            onClick={() => setSelCell([r, c])}
                          >
                            {v}
                          </button>
                        </td>
                      ))}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="mcalc">
              C[{mr}][{mc}] = {rowA.map((v, i) => `(${v}×${colB[i]})`).join(" + ")} = <b>{matrixC[mr][mc]}</b>
            </div>
            <div className="note">规则只有一句：结果矩阵 <code>C</code> 的第 <code>i</code> 行第 <code>j</code> 列 = <code>A</code> 的第 <code>i</code> 行 与 <code>B</code> 的第 <code>j</code> 列<b>逐个相乘再相加</b>。记住这句，后面每步都看得懂。</div>
          </section>

          {/* ===== RNN ===== */}
          <section className="section" id="s2">
            <SecHead idx="02" title="从 RNN 的痛点说起" />
            <p className="sec-lead">在 Attention 之前，处理「一串词」的主力是 RNN。它最大的毛病：<b style={{ color: "var(--t1)" }}>只能一个一个往后算，算不出第 4 个就得等前 3 个</b>——无法并行，GPU 干着急。</p>
            <FigRnnVsAtt />
            <div className="note">Self-Attention 的输入输出和 RNN <b>一模一样</b>（序列进、序列出），区别只在：每个输出 <Formula tex="b_i" /> 都<b>并行地</b>看过了整条序列。</div>
          </section>

          {/* ===== 向量级 ===== */}
          <section className="section" id="s3">
            <SecHead idx="03" title="Self-Attention · 向量级（一步一步算）" />
            <div className="legend-row">
              <span><i className="lq" />Query 查询：我想找什么</span>
              <span><i className="lk" />Key 键：我有什么可被匹配</span>
              <span><i className="lv" />Value 值：匹配上后拿走的内容</span>
            </div>
            <p className="sec-lead">这是整堂课的核心一张图。以「算出第 1 个输出 <Formula tex="b_1" />」为例分 4 步。先记住三个词：每个输入词 <Formula tex="x" /> 会变成三份不同身份——<b style={{ color: "var(--q)" }}>Q（去问别人）</b>、<b style={{ color: "var(--k)" }}>K（被别人问）</b>、<b style={{ color: "var(--v)" }}>V（真正的内容）</b>。</p>
            <FigAttentionSteps />
            <div className="steps">
              <article><b>① 生成 Q/K/V</b><p>每个词 x 乘三个可学习矩阵，得到身份三件套：去问、被问、内容。</p></article>
              <article><b>② 点积打分</b><p>q₁ 与所有人的 k 点积，得到相关度分数 α，越像分数越高。</p></article>
              <article><b>③ softmax 变权重</b><p>分数过 softmax，变成加起来=1 的权重，即注意力分配。</p></article>
              <article><b>④ 加权求和</b><p>权重去加权所有人的 v，求和得到融合全局信息的 b₁。</p></article>
            </div>
            <div className="note warn">关键直觉：<b>Q 和 K 决定"该关注谁"，V 决定"被关注后拿走的内容"</b>。点积大 = 两个词语义相关 = 权重大 = b 里这部分 v 占比高。</div>
          </section>

          {/* ===== 矩阵级 ===== */}
          <section className="section" id="s4">
            <SecHead idx="04" title="Self-Attention · 矩阵级（三步搞定）" />
            <p className="sec-lead">把所有词的 q/k/v 堆成矩阵 <Formula tex="Q, K, V" />，整件事就坍缩成<b style={{ color: "var(--t1)" }}>三次矩阵乘法</b>——这正是 GPU 最擅长、能大规模并行的形态。</p>
            <FigAttentionMatrix />
            <div className="eq-box">
              <Formula block tex={String.raw`\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V`} />
            </div>
            <div className="note">现在这句公式对你不再是一串符号：<Formula tex={String.raw`QK^{\mathsf T}`} /> 是「两两算相关度」，softmax 是「分数变权重」，乘 <Formula tex={String.raw`V`} /> 是「按权重取内容」。</div>
          </section>

          {/* ===== 缩放 ===== */}
          <section className="section" id="s5">
            <SecHead idx="05" title="那个 √dₖ 是干嘛的？" />
            <p className="sec-lead">公式里多了一个「除以 <Formula tex={String.raw`\sqrt{d_k}`} />」，叫<b style={{ color: "var(--t1)" }}>缩放（Scale）</b>。原因一句：维度 <Formula tex="d_k" /> 越大，点积数值越大，softmax 会被推向极端（一个 1、其余 0），梯度消失训不动。</p>
            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>不缩放会怎样</h3>
                <p className="t3">点积是 <Formula tex="d_k" /> 个乘积之和。<Formula tex="d_k" /> 大 → 点积方差大 → softmax 近似 one-hot → 梯度接近 0 → 训练停滞。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>除以 √dₖ 的效果</h3>
                <p className="t3">把点积方差<b style={{ color: "var(--t1)" }}>拉回 1 附近</b>，让 softmax 处在温和区间，梯度健康。这是算子实现里最易漏、但必须有的细节。</p>
              </div>
            </div>
            <div className="note">
              <Formula tex={String.raw`\operatorname{Var}(q_i k_j^{\mathsf T})\approx d_k\quad\Rightarrow\quad \operatorname{Var}\!\left(\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}\right)\approx1`} />
            </div>

            {/* 数值演示：选 Query 看权重 */}
            <h3>数值演示：选一个 Query，看它把注意力分给谁</h3>
            <div className="tabs">
              {words.map((w, i) => (
                <button key={w} className={`tab ${qIdx === i ? "active" : ""}`} onClick={() => setQIdx(i)}>q{i + 1} · {w}</button>
              ))}
            </div>
            <div className="card">
              <div style={{ display: "flex", gap: 22, flexWrap: "wrap", justifyContent: "center", alignItems: "center" }}>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">scaled logits <span style={{ color: "var(--att)" }}>α = q·k/√d</span></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.scaled.map((v, i) => (
                      <div key={i} style={{ textAlign: "center" }}>
                        <div className="mcell" style={{ cursor: "default" }}>{v.toFixed(2)}</div>
                        <div className="mname" style={{ marginTop: 4 }}>k{i + 1}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <span className="msign">→</span>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">权重 <span style={{ color: "var(--att)" }}>softmax → 和=1</span></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.weights.map((v, i) => (
                      <div key={i} style={{ textAlign: "center" }}>
                        <div className="mcell" style={{ cursor: "default", background: `rgba(56,189,248,${0.12 + v * 0.6})`, borderColor: "var(--att)" }}>{(v * 100).toFixed(0)}%</div>
                        <div className="mname" style={{ marginTop: 4 }}>k{i + 1}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <span className="msign">→</span>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">输出 <span style={{ color: "var(--out)" }}>z = Σ α̂·v</span></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.output.map((v, i) => (
                      <div key={i} className="mcell" style={{ cursor: "default", color: "var(--out)", borderColor: "var(--out)" }}>{v.toFixed(2)}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ===== 多头 ===== */}
          <section className="section" id="s6">
            <SecHead idx="06" title="多头注意力（Multi-Head）" />
            <p className="sec-lead">只做一次 attention 只能学到「一种关注方式」。拆成<b style={{ color: "var(--t1)" }}>多个头</b>，各自独立算 attention，等于从多个角度（语法、长程、局部……）同时看序列，最后拼回来。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,\,KW_i^K,\,VW_i^V)`} />
              <Formula block tex={String.raw`\operatorname{MHA}=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\,W^O`} />
            </div>
            <div className="note">实践要点：每个头把维度 <Formula tex="d" /> 切成 <Formula tex="d/h" />，所以<b>总计算量和单头接近</b>，但表达能力更强——几乎免费获得多视角。</div>

            <h3>不同头学到了不同的关注方式</h3>
            <div className="tabs">
              {heads.map((h, i) => (
                <button key={h.name} className={`tab ${headIdx === i ? "active" : ""}`} onClick={() => setHeadIdx(i)}>{h.name}</button>
              ))}
            </div>
            <div className="fig">
              <div style={{ display: "flex", justifyContent: "center", overflowX: "auto" }}>
                <table style={{ borderCollapse: "separate", borderSpacing: 4, margin: "0 auto" }}>
                  <tbody>
                    <tr>
                      <td></td>
                      {words.map((w) => <td key={w} style={{ textAlign: "center", fontFamily: "var(--mono)", fontSize: 12, color: "var(--t3)", width: 64 }}>{w}</td>)}
                    </tr>
                    {heads[headIdx].matrix.map((row, i) => (
                      <tr key={i}>
                        <td style={{ textAlign: "right", paddingRight: 10, fontFamily: "var(--mono)", fontSize: 12, color: "var(--t3)" }}>{words[i]}</td>
                        {row.map((v, j) => (
                          <td key={j}>
                            <div style={{
                              width: 58, height: 50, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
                              fontFamily: "var(--mono)", fontSize: 13, fontWeight: 700,
                              background: `rgba(56,189,248,${0.08 + v * 0.7})`, border: "1px solid rgba(56,189,248,0.3)",
                              color: v > 0.4 ? "#fff" : "var(--t2)",
                            }}>{v.toFixed(2)}</div>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="fig-cap">{heads[headIdx].note} · 行=Query（谁在问）· 列=Key（看谁）</div>
            </div>
          </section>

          {/* ===== 经典 Transformer ===== */}
          <section className="section" id="s7">
            <SecHead idx="07" title="经典 Transformer 全景图（论文 Figure 1）" />
            <p className="sec-lead">把 Attention 装进完整模型长什么样？这就是被引用几万次的结构图——<b style={{ color: "var(--att)" }}>左 Encoder</b>、<b style={{ color: "var(--out)" }}>右 Decoder</b>，各堆叠 N 层。</p>
            <FigTransformer />
            <div className="grid2">
              <div className="note"><b>Encoder</b>：对源序列做 Self-Attention + FFN，逐层提炼表示，最后把 K、V 交给 Decoder。</div>
              <div className="note"><b>Decoder</b>：先用 <b>Masked</b> Self-Attention（屏蔽未来位防作弊），再通过 <b>Cross-Attention</b> 读取编码器的 K/V，最后预测下一个词。</div>
            </div>
            <div className="note warn">这张图里 <b>Attention 出现了三次</b>（Encoder 自注意、Decoder 掩码自注意、Decoder 交叉注意）。我们前面学的那个公式，是这三处共用的同一个算子——这就是它"为什么重要"的最终答案。</div>
          </section>

          {/* ===== 位置编码 ===== */}
          <section className="section" id="s8">
            <h3>位置编码</h3>
            <p className="sec-lead">Self-Attention 对「顺序」无感——「我打他」和「他打我」算出来一样。所以在输入 embedding 上<b style={{ color: "var(--t1)" }}>直接加一个位置向量</b>，把顺序信息喂回去。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\quad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)`} />
            </div>
            <div className="note">用不同频率的正余弦，让每个位置拿到<b>唯一</b>编码；且对任意固定间距 <Formula tex="k" />，<Formula tex={String.raw`PE_{pos+k}`} /> 是 <Formula tex={String.raw`PE_{pos}`} /> 的线性函数——模型因此能泛化到比训练更长的序列。</div>
          </section>

          {/* ===== 代码 ===== */}
          <section className="section" id="s9">
            <SecHead idx="08" title="经典代码 + 算子测试要点" />
            <p className="sec-lead">原理看懂了，落到代码就几十行。下面是哈佛 The Annotated Transformer 的经典实现，逐行对应步骤。</p>

            <div className="code-title">① 缩放点积注意力 — 对应 softmax(QKᵀ/√dk)V</div>
            <pre><code>{`def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, value), p_attn`}</code></pre>

            <div className="code-title">② 多头注意力 — 对应分头、拼接、Wᴼ 映射</div>
            <pre><code>{`class MultiHeadedAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 线性映射 + 拆头: (b, seq, d_model) -> (b, h, seq, d_k)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask)
        # 拼头: (b, h, seq, d_k) -> (b, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)`}</code></pre>

            <h3>算子测试要点（leader 关心的）</h3>
            <div className="grid2">
              <div className="card"><h3 style={{ marginTop: 0 }}>数值正确性</h3><p className="t3">小矩阵手算 softmax(QKᵀ/√dk)V 比对；验证每行和=1、padding 权重≈0。</p></div>
              <div className="card"><h3 style={{ marginTop: 0 }}>形状与边界</h3><p className="t3">校验 (B,h,n,dₖ) 变换；dₖ 不整除头数报错、空序列、单 token。</p></div>
              <div className="card"><h3 style={{ marginTop: 0 }}>mask 正确性</h3><p className="t3">因果 mask 严格下三角；屏蔽位 softmax 后为 0，不受未来影响。</p></div>
              <div className="card"><h3 style={{ marginTop: 0 }}>性能与精度</h3><p className="t3">fp32/fp16/bf16 相对误差（&lt;1e-3）；显存与耗时随 seq/heads/dₖ 的曲线。</p></div>
            </div>

            <div className="note ok"><b>一句话总结整堂课</b>：Attention 把「每个位置该关注谁」变成 <Formula tex={String.raw`QK^{\mathsf T}\!/\sqrt{d_k}`} /> 算分、softmax 变权重、再乘 <Formula tex="V" /> 取内容——三步矩阵乘法。多头扩展视角，加位置编码补顺序，组装成 Encoder/Decoder，就是撑起所有现代大模型的 Transformer。</div>
          </section>

          <div className="foot">
            参考：李宏毅 Self-Attention 讲解（PDF 图 1–15）· Vaswani et al. <i>Attention Is All You Need</i>（Figure 1）· The Annotated Transformer。<br />
            全篇 Q/K/V/Attention/Output 配色一致，SVG 可自由放大讲解。
          </div>

        </div>
      </main>
    </>
  );
}
