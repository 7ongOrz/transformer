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
 * SVG 图：矩阵级 Self-Attention（对应 PDF 图10-12）
 * ============================================================ */
function FigAttentionMatrix() {
  return (
    <div className="fig">
      <svg viewBox="0 0 920 340" width="920" role="img" aria-label="self-attention 矩阵级三步">
        <defs>
          <linearGradient id="gq-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#f5b042" stopOpacity=".5" /><stop offset="1" stopColor="#f5b042" stopOpacity=".1" /></linearGradient>
          <linearGradient id="gk-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#a78bfa" stopOpacity=".5" /><stop offset="1" stopColor="#a78bfa" stopOpacity=".1" /></linearGradient>
          <linearGradient id="gv-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#2dd4bf" stopOpacity=".5" /><stop offset="1" stopColor="#2dd4bf" stopOpacity=".1" /></linearGradient>
          <linearGradient id="ga-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#38bdf8" stopOpacity=".55" /><stop offset="1" stopColor="#38bdf8" stopOpacity=".12" /></linearGradient>
          <linearGradient id="go-m" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#f472b6" stopOpacity=".55" /><stop offset="1" stopColor="#f472b6" stopOpacity=".12" /></linearGradient>
          <marker id="ah-m" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="#6e7aab" /></marker>
        </defs>
        <text x="20" y="24" fill="#c8d4ff" fontSize="14" fontWeight="700">把向量堆成矩阵后，整个过程 = 三次矩阵乘法</text>
        <rect x="20" y="110" width="60" height="120" rx="8" fill="#0c1430" stroke="rgba(255,255,255,0.08)" />
        <text x="50" y="175" textAnchor="middle" fill="#eef3ff" fontSize="15" fontWeight="700">I</text>
        <text x="50" y="248" textAnchor="middle" fill="#6e7aab" fontSize="10">(n×d)</text>
        <path d="M80,140 H120" stroke="#f5b042" strokeWidth="1.4" fill="none" markerEnd="url(#ah-m)" />
        <path d="M80,170 H120" stroke="#a78bfa" strokeWidth="1.4" fill="none" markerEnd="url(#ah-m)" />
        <path d="M80,200 H120" stroke="#2dd4bf" strokeWidth="1.4" fill="none" markerEnd="url(#ah-m)" />
        <text x="84" y="133" fill="#f5b042" fontSize="10">×Wᵠ</text>
        <text x="84" y="164" fill="#a78bfa" fontSize="10">×Wᵏ</text>
        <text x="84" y="216" fill="#2dd4bf" fontSize="10">×Wᵛ</text>
        <rect x="120" y="125" width="56" height="34" rx="6" fill="url(#gq-m)" stroke="#f5b042" /><text x="148" y="147" textAnchor="middle" fill="#f5b042" fontWeight="700">Q</text>
        <rect x="120" y="163" width="56" height="34" rx="6" fill="url(#gk-m)" stroke="#a78bfa" /><text x="148" y="185" textAnchor="middle" fill="#a78bfa" fontWeight="700">K</text>
        <rect x="120" y="201" width="56" height="34" rx="6" fill="url(#gv-m)" stroke="#2dd4bf" /><text x="148" y="223" textAnchor="middle" fill="#2dd4bf" fontWeight="700">V</text>
        <text x="215" y="152" fill="#a9b4dc" fontSize="12">Q · Kᵀ</text>
        <text x="215" y="170" fill="#6e7aab" fontSize="10">两两相关度</text>
        <path d="M176,152 H210" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <path d="M176,190 H210" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <rect x="295" y="110" width="90" height="120" rx="8" fill="url(#ga-m)" stroke="#38bdf8" />
        <text x="340" y="175" textAnchor="middle" fill="#38bdf8" fontWeight="700" fontSize="14">A=QKᵀ</text>
        <text x="340" y="248" textAnchor="middle" fill="#6e7aab" fontSize="10">(n×n)</text>
        <path d="M245,158 C270,158 275,175 293,178" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <rect x="420" y="110" width="80" height="120" rx="8" fill="#0c1430" stroke="#38bdf8" />
        <text x="460" y="170" textAnchor="middle" fill="#38bdf8" fontWeight="700" fontSize="13">softmax</text>
        <text x="460" y="188" textAnchor="middle" fill="#38bdf8" fontWeight="700" fontSize="13">按行</text>
        <text x="460" y="248" textAnchor="middle" fill="#6e7aab" fontSize="10">→ 权重 Â</text>
        <path d="M385,170 H418" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <rect x="540" y="110" width="60" height="120" rx="8" fill="url(#ga-m)" stroke="#38bdf8" />
        <text x="570" y="175" textAnchor="middle" fill="#38bdf8" fontWeight="700" fontSize="14">Â</text>
        <text x="570" y="248" textAnchor="middle" fill="#6e7aab" fontSize="10">(n×n)</text>
        <path d="M500,170 H538" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <path d="M176,228 C400,300 470,300 538,210" stroke="#2dd4bf" strokeWidth="1.3" fill="none" strokeDasharray="4 3" markerEnd="url(#ah-m)" />
        <rect x="660" y="110" width="70" height="120" rx="8" fill="url(#go-m)" stroke="#f472b6" />
        <text x="695" y="175" textAnchor="middle" fill="#f472b6" fontWeight="700" fontSize="15">O</text>
        <text x="695" y="248" textAnchor="middle" fill="#6e7aab" fontSize="10">(n×d) 输出</text>
        <path d="M600,170 H658" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-m)" />
        <g transform="translate(760,110)">
          <rect x="0" y="0" width="140" height="120" rx="9" fill="#070b18" stroke="rgba(255,255,255,0.08)" />
          <text x="70" y="24" textAnchor="middle" fill="#5eead4" fontWeight="700" fontSize="12">整个过程 =</text>
          <text x="12" y="52" fill="#a9b4dc" fontSize="11">① I×W → Q,K,V</text>
          <text x="12" y="74" fill="#a9b4dc" fontSize="11">② QKᵀ → softmax</text>
          <text x="12" y="96" fill="#a9b4dc" fontSize="11">③ Â×V → O</text>
          <text x="12" y="114" fill="#6e7aab" fontSize="9.5">三次矩阵乘法</text>
        </g>
      </svg>
      <div className="fig-shapes">
        <span><b>X</b> <em>[N,d]</em></span><em>→</em>
        <span><b>Q,K,V</b> <em>[N,d_k/d_v]</em></span><em>→</em>
        <span><b>A=QKᵀ</b> <em>[N,N]</em></span><em>→</em>
        <span><b>Â</b> <em>[N,N]</em></span><em>→</em>
        <span><b>O=ÂV</b> <em>[N,d_v]</em></span>
      </div>
      <div className="fig-cap">图 · 向量级推导收束成矩阵乘法，可用 GPU 并行加速</div>
    </div>
  );
}

/* ============================================================
 * 数值矩阵流水线：用真实小数值从 X 一步步算到 O
 * （N=2, d=2；为演示构造的小矩阵，非真实模型权重）
 * ============================================================ */
const DEMO = {
  X: [[1, 2], [3, 1]],
  WQ: [[1, 0], [0, 1]],
  WK: [[0, 1], [1, 0]],
  WV: [[1, 1], [1, 0]],
};
// 预计算（与 Python 验证一致）
const DEMO_Q = [[1, 2], [3, 1]];
const DEMO_K = [[2, 1], [1, 3]];
const DEMO_V = [[3, 1], [4, 3]];
const DEMO_S = [[2.83, 4.95], [4.95, 4.24]];
const DEMO_A = [[0.11, 0.89], [0.67, 0.33]];
const DEMO_O = [[3.89, 2.79], [3.33, 1.66]];

function NumMatrix({ data, heat, warm, digits = 2 }: { data: number[][]; heat?: boolean; warm?: boolean; digits?: number }) {
  const cls = heat ? "heat" : warm ? "warm" : "";
  return (
    <table className={`nm ${cls}`}>
      <tbody>
        {data.map((row, i) => (
          <tr key={i}>
            {row.map((v, j) => {
              let bg: string | undefined;
              let color = "#fff";
              if (heat) {
                // 按值映射蓝色阶
                const t = Math.min(1, Math.max(0, v));
                bg = `rgba(56,189,248,${0.15 + t * 0.7})`;
                color = t > 0.45 ? "#fff" : "#a9b4dc";
              }
              return (
                <td key={j} style={bg ? { background: bg, color } : undefined}>
                  {v.toFixed(digits)}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ---------- 注意力大网格矩阵（带行列标签，仿参考文档 mask-table）---------- */
const WORDS4 = ["我", "爱", "深", "度"];
// 4 个 token 的真实数值（d=2，为演示构造）
const GRID_S = [
  [0.71, 2.83, 2.83, 2.12],
  [0.0, 2.83, 1.41, 1.41],
  [1.41, 7.07, 6.36, 4.95],
  [0.71, 4.24, 3.54, 2.83],
];
const GRID_A = [
  [0.05, 0.38, 0.38, 0.19],
  [0.04, 0.65, 0.16, 0.16],
  [0.0, 0.62, 0.3, 0.07],
  [0.02, 0.56, 0.28, 0.14],
];

function AttnGrid({ data, rowColor, digits = 2 }: { data: number[][]; rowColor: string; digits?: number }) {
  // 找全局最大值用于色阶归一
  const max = Math.max(...data.flat());
  return (
    <div className="attn-grid">
      <table>
        <thead>
          <tr>
            <th className="corner">Q ＼ K</th>
            {WORDS4.map((w) => <th key={w} className="collab">{w}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <th className="rowlab" style={{ color: rowColor }}>{WORDS4[i]}</th>
              {row.map((v, j) => {
                const t = max > 0 ? v / max : 0;
                const bg = `rgba(56,189,248,${0.1 + t * 0.75})`;
                const color = t > 0.4 ? "#fff" : "#a9b4dc";
                return (
                  <td key={j}>
                    <div className="cell" style={{ background: bg, color }}>{v.toFixed(digits)}</div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="axis">行 = Query（谁在问）· 列 = Key（看谁）· 颜色越深权重越大</div>
    </div>
  );
}

/* ---------- 逐图步骤卡片（向量级，对应文章图4-8 的颗粒度）---------- */
type StepCardProps = {
  num: string;
  title: string;
  summary: string;
  color: string; // 编号底色
  steps: string[];
  tex: string;
  shapes: string[];
  note?: string;
  workout?: React.ReactNode; // 逐项演算展开式
  children?: React.ReactNode; // 数值矩阵展示区
};

function StepCard({ num, title, summary, color, steps, tex, shapes, note, workout, children }: StepCardProps) {
  return (
    <div className="step-card">
      <div className="sc-head">
        <span className="sc-num" style={{ background: color }}>{num}</span>
        <span className="sc-title">{title}</span>
      </div>
      <p className="sc-summary">{summary}</p>
      <div className="sc-body">
        <div className="sc-mat">{children}</div>
        <div className="sc-right">
          <ol className="sc-steps">
            {steps.map((s, i) => <li key={i}>{s}</li>)}
          </ol>
          <div className="sc-formula">
            <Formula tex={tex} />
          </div>
          <div className="sc-shapes">
            {shapes.map((s, i) => <span key={i}>{s}</span>)}
          </div>
        </div>
      </div>
      {workout ? <div className="sc-workout">{workout}</div> : null}
      {note ? <div className="sc-note">{note}</div> : null}
    </div>
  );
}

function FigNumPipeline() {
  return (
    <div className="pipeline">
      <div className="pl-note">
        下面以「我 爱 深 度」4 个 token 为例，看注意力矩阵怎么从分数一步步变成权重。
        每一步都用<b>带行列标签的网格矩阵</b>呈现——行是 Query（谁在问），列是 Key（看谁）。
      </div>

      {/* 第 1 步：Q/K/V 投影（公式说明，小矩阵示意）*/}
      <div className="pipeline-row" style={{ alignItems: "flex-start" }}>
        <div className="pl-step">
          <div className="pl-title">输入 <b>X</b>　<span style={{ color: "#6e7aab" }}>[4, d]</span></div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 12.5, color: "#a9b4dc", marginTop: 6, lineHeight: 1.9 }}>
            我 → x₁<br />爱 → x₂<br />深 → x₃<br />度 → x₄
          </div>
        </div>
        <div className="pl-op">→</div>
        <div className="pl-step" style={{ flex: "1 1 320px", minWidth: 260 }}>
          <div className="pl-title">三路投影</div>
          <div className="sc-formula" style={{ margin: "6px 0" }}>
            <Formula block tex={String.raw`Q = XW^Q,\quad K = XW^K,\quad V = XW^V`} />
          </div>
          <div style={{ fontSize: 12.5, color: "#6e7aab", lineHeight: 1.8 }}>
            每个 token 乘三个权重矩阵，得到各自的 <b style={{ color: "#f5b042" }}>q</b>（查询）、<b style={{ color: "#a78bfa" }}>k</b>（键）、<b style={{ color: "#2dd4bf" }}>v</b>（值）。
          </div>
        </div>
      </div>

      {/* 第 2 步：分数 S 与权重 A 的大网格（核心可视化）*/}
      <h3 style={{ marginTop: 30 }}>分数矩阵 → softmax → 权重矩阵</h3>
      <p className="t3" style={{ fontSize: 14, marginTop: 0 }}>
        左边是<b>原始分数</b> <Formula tex={String.raw`S=QK^{\mathsf T}/\sqrt{d_k}`} />（每行一个 Query 对所有 Key 的相关度）；
        右边是 <b>softmax 后的权重</b> <Formula tex={String.raw`A`} />（每行归一化、和为 1，即注意力分配）。
      </p>
      <div className="attn-pair">
        <div className="attn-block">
          <div className="ab-title">分数 <b>S = QKᵀ/√dₖ</b>　[4,4]</div>
          <AttnGrid data={GRID_S} rowColor="#f5b042" digits={2} />
        </div>
        <div className="attn-block">
          <div className="ab-title">权重 <b style={{ color: "#38bdf8" }}>A = softmax(S)</b>　[4,4]</div>
          <AttnGrid data={GRID_A} rowColor="#f5b042" digits={2} />
        </div>
      </div>

      {/* 第 3 步：乘 V 得输出 */}
      <div className="eq-box" style={{ marginTop: 24 }}>
        <Formula block tex={String.raw`O = AV`} />
      </div>
      <div className="pl-note">
        权重矩阵 <b>A</b> 的每一行，就是对 <b>V</b> 各行做加权求和的配比——比如「深」这一行权重是
        <b>[0.00, 0.62, 0.30, 0.07]</b>，说明它主要关注「爱」，于是输出 O₃ 里「爱」的值 v₂ 占了 62%。
        这就是「该关注谁、拿走什么」的完整数值化呈现。
      </div>
    </div>
  );
}

/* ============================================================
 * SVG 图：经典 Transformer 论文 Figure 1
 * ============================================================ */
function FigTransformer() {
  const Arrow = ({ d, color = "#6e7aab", dash }: { d: string; color?: string; dash?: string }) => (
    <path d={d} stroke={color} strokeWidth="1.4" fill="none" markerEnd="url(#ah-t)" strokeDasharray={dash} />
  );
  const Box = ({ x, y, w, h, fill, stroke, label, sub, lc, sc }: { x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; lc?: string; sc?: string }) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="9" fill={fill} stroke={stroke} />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -4 : 5)} textAnchor="middle" fill={lc || "#eef3ff"} fontSize="13" fontWeight="700">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fill={sc || "#6e7aab"} fontSize="10">{sub}</text>}
    </g>
  );
  return (
    <div className="fig">
      <svg viewBox="0 0 940 600" width="940" role="img" aria-label="经典 Transformer Encoder-Decoder 结构图">
        <defs>
          <marker id="ah-t" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="#6e7aab" /></marker>
        </defs>
        <text x="250" y="26" textAnchor="middle" fill="#38bdf8" fontSize="15" fontWeight="700">Encoder × N（左）</text>
        <text x="710" y="26" textAnchor="middle" fill="#f472b6" fontSize="15" fontWeight="700">Decoder × N（右）</text>

        <Box x={60} y={60} w={120} h={40} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Input Embedding" lc="#a9b4dc" />
        <circle cx="200" cy="80" r="15" fill="#070b18" stroke="#a78bfa" />
        <text x="200" y="85" textAnchor="middle" fill="#a78bfa" fontSize="15">+</text>
        <text x="200" y="50" textAnchor="middle" fill="#a78bfa" fontSize="10">Positional</text>
        <text x="200" y="62" textAnchor="middle" fill="#a78bfa" fontSize="10">Encoding</text>
        <Arrow d="M180,80 H186" />
        <rect x="80" y="125" width="290" height="345" rx="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeDasharray="5 5" />
        <text x="225" y="120" textAnchor="middle" fill="#6e7aab" fontSize="11">Encoder Layer (堆叠 N 次)</text>
        <Box x={120} y={150} w={210} h={54} fill="rgba(56,189,248,0.14)" stroke="#38bdf8" label="Multi-Head Self-Attention" sub="本节讲的核心算子" lc="#38bdf8" sc="#6e7aab" />
        <Box x={150} y={222} w={150} h={36} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={120} y={278} w={210} h={50} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" label="Feed-Forward Network" sub="两层 MLP（逐位置作用）" lc="#2dd4bf" sc="#6e7aab" />
        <Box x={150} y={346} w={150} h={36} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Arrow d="M225,204 V218" /><Arrow d="M225,258 V274" /><Arrow d="M225,328 V342" />
        <Arrow d="M120,170 H100 V410 H225" color="#f5b042" dash="4 3" />
        <text x="92" y="395" fill="#f5b042" fontSize="9">残差</text>
        <Box x={150} y={405} w={150} h={34} fill="rgba(244,114,182,0.14)" stroke="#f472b6" label="编码器输出（K, V）" lc="#f472b6" />
        <Arrow d="M215,100 V150" />

        <Box x={560} y={60} w={120} h={40} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Output Embedding" lc="#a9b4dc" />
        <circle cx="700" cy="80" r="15" fill="#070b18" stroke="#a78bfa" />
        <text x="700" y="85" textAnchor="middle" fill="#a78bfa" fontSize="15">+</text>
        <text x="700" y="50" textAnchor="middle" fill="#a78bfa" fontSize="10">Positional</text>
        <text x="700" y="62" textAnchor="middle" fill="#a78bfa" fontSize="10">Encoding</text>
        <Arrow d="M680,80 H686" />
        <rect x="540" y="125" width="310" height="345" rx="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeDasharray="5 5" />
        <text x="695" y="120" textAnchor="middle" fill="#6e7aab" fontSize="11">Decoder Layer (堆叠 N 次)</text>
        <Box x={575} y={150} w={240} h={50} fill="rgba(245,176,66,0.14)" stroke="#f5b042" label="Masked Multi-Head Attention" sub="只能看过去（屏蔽未来位）" lc="#f5b042" sc="#6e7aab" />
        <Box x={620} y={216} w={150} h={34} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={575} y={268} w={240} h={50} fill="rgba(56,189,248,0.14)" stroke="#38bdf8" label="Cross Attention（编码-解码交互）" sub="Q 来自解码器，K,V 来自编码器" lc="#38bdf8" sc="#6e7aab" />
        <Box x={620} y={334} w={150} h={34} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={575} y={386} w={240} h={44} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" label="Feed-Forward Network" lc="#2dd4bf" />
        <Box x={620} y={442} w={150} h={32} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Arrow d="M695,200 V212" /><Arrow d="M695,250 V264" /><Arrow d="M695,318 V330" /><Arrow d="M695,368 V382" /><Arrow d="M695,430 V438" />
        <Arrow d="M300,422 C440,422 460,293 573,293" color="#a78bfa" dash="4 3" />
        <text x="430" y="360" fill="#a78bfa" fontSize="10">编码器 K, V 传过来</text>
        <Arrow d="M695,474 V492" />
        <Box x={600} y={495} w={190} h={34} fill="rgba(167,139,250,0.14)" stroke="#a78bfa" label="Linear → Softmax → 词概率" lc="#a78bfa" />
        <Arrow d="M685,100 V150" />

        <g transform="translate(80,555)">
          <rect x="0" y="0" width="14" height="14" rx="3" fill="rgba(56,189,248,0.14)" stroke="#38bdf8" /><text x="20" y="12" fill="#6e7aab" fontSize="11">Attention</text>
          <rect x="110" y="0" width="14" height="14" rx="3" fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" /><text x="130" y="12" fill="#6e7aab" fontSize="11">FFN</text>
          <rect x="190" y="0" width="14" height="14" rx="3" fill="#0c1430" stroke="rgba(255,255,255,0.08)" /><text x="210" y="12" fill="#6e7aab" fontSize="11">Add&amp;Norm</text>
          <rect x="310" y="0" width="14" height="14" rx="3" fill="#070b18" stroke="#a78bfa" /><text x="330" y="12" fill="#6e7aab" fontSize="11">位置编码</text>
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
        <defs><marker id="ah-r" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L7,4 L0,8 z" fill="#6e7aab" /></marker></defs>
        <text x="40" y="40" fill="#c8d4ff" fontSize="15" fontWeight="700">RNN：必须串行，算不出 b₄ 就用不了 b₃</text>
        {[80, 180, 280, 380].map((cx, i) => (
          <g key={`r-${cx}`}>
            <circle cx={cx} cy={130} r="20" fill="#0c1430" stroke="#f5b042" />
            <text x={cx} y={135} textAnchor="middle" fill="#f5b042" fontSize="13">x{i + 1}</text>
            <rect x={cx - 20} y={185} width="40" height="32" rx="6" fill="#0c1430" stroke="rgba(255,255,255,0.08)" />
            <text x={cx} y={205} textAnchor="middle" fill="#a9b4dc" fontSize="12">b{i + 1}</text>
            <path d={`M${cx},150 V182`} stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
          </g>
        ))}
        <path d="M100,205 H155" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
        <path d="M200,205 H255" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
        <path d="M300,205 H355" stroke="#6e7aab" strokeWidth="1.3" fill="none" markerEnd="url(#ah-r)" />
        <text x="80" y="245" fill="#ff7a7a" fontSize="11">❌ 无法并行：b₂ 要等 b₁</text>

        <text x="510" y="40" fill="#c8d4ff" fontSize="15" fontWeight="700">Self-Attention：所有位置同时互相看见，可并行</text>
        {[540, 640, 740, 840].map((cx, i) => (
          <g key={`a-${cx}`}>
            <circle cx={cx} cy={80} r="20" fill="#0c1430" stroke="#2dd4bf" />
            <text x={cx} y={85} textAnchor="middle" fill="#2dd4bf" fontSize="13">x{i + 1}</text>
            <rect x={cx - 20} y={160} width="40" height="32" rx="6" fill="#0c1430" stroke="rgba(255,255,255,0.08)" />
            <text x={cx} y={180} textAnchor="middle" fill="#f472b6" fontSize="12">b{i + 1}</text>
          </g>
        ))}
        <g stroke="#38bdf8" strokeWidth="1" opacity="0.5" fill="none">
          {[540, 640, 740, 840].map((cx) =>
            [540, 640, 740, 840].map((tx) => <path key={`l-${cx}-${tx}`} d={`M${cx},100 Q${(cx + tx) / 2},130 ${tx},160`} />),
          )}
        </g>
        <text x="640" y="225" fill="#34d399" fontSize="11">✅ 每个输出都看完整序列，且各 b 互不依赖、一起算</text>
      </svg>
      <div className="fig-cap">图 · 处理序列数据的两种范式：RNN 串行 vs Self-Attention 并行</div>
    </div>
  );
}

/* ---------- 章节头 ---------- */
function SecHead({ idx, title }: { idx: string; title: React.ReactNode }) {
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
        <div className="sub">Transformer 核心算子<br />从矩阵乘法讲到大模型全景</div>
        <ol>
          {navItems.map(([id, label]) => (
            <li key={id}>
              <a href={`#${id}`} className={activeNav === id ? "active" : ""}>{label}</a>
            </li>
          ))}
        </ol>
        <div className="legend">
          <span><i style={{ background: "#f5b042" }} />Query 查询</span>
          <span><i style={{ background: "#a78bfa" }} />Key 键</span>
          <span><i style={{ background: "#2dd4bf" }} />Value 值</span>
          <span><i style={{ background: "#38bdf8" }} />Attention</span>
          <span><i style={{ background: "#f472b6" }} />Output</span>
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
              <span>从矩阵乘法起步</span>
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
                <p className="t3">Attention 的核心就是<b style={{ color: "#eef3ff" }}>三个矩阵乘法 + 一个 softmax</b>。这篇分享的目标，是让你在脑子里把这句话可视化出来。</p>
              </div>
            </div>
          </section>

          {/* ===== 矩阵乘法 ===== */}
          <section className="section" id="s1">
            <SecHead idx="01" title="热身：矩阵乘法到底怎么乘" />
            <p className="sec-lead">Attention 的全部运算都是矩阵乘法。先抛开深度学习，用具体数字搞明白规则——<b style={{ color: "#f472b6" }}>点一下右边结果矩阵的任意格子</b>，左边高亮参与计算的行与列。</p>
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
            <p className="sec-lead">在 Attention 之前，处理「一串词」的主力是 RNN。它最大的毛病：<b style={{ color: "#eef3ff" }}>只能一个一个往后算，算不出第 4 个就得等前 3 个</b>——无法并行，GPU 干着急。</p>
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
            <p className="sec-lead">这是整篇的核心。以「算出第 1 个输出 <Formula tex="b_1" />」为例分 4 步——每步配真实数值，跟着算一遍就懂。先记住三个词：每个输入词 <Formula tex="x" /> 会变成三份不同身份——<b style={{ color: "#f5b042" }}>Q（去问别人）</b>、<b style={{ color: "#a78bfa" }}>K（被别人问）</b>、<b style={{ color: "#2dd4bf" }}>V（真正的内容）</b>。</p>

            <StepCard
              num="1"
              title="生成 Q / K / V"
              color="#f5b042"
              summary="每个输入向量 x 分别乘三个可学习矩阵，得到它的查询 q、键 k、值 v（身份三件套）。"
              steps={[
                "输入 X=[[1,2],[3,1]]（2 个 token，2 维）。",
                "X 乘 Wᵠ 得 Q，乘 Wᵏ 得 K，乘 Wᵛ 得 V——三组独立权重。",
                "每个 W 都是可学习参数，训练时更新。",
              ]}
              tex={String.raw`Q = XW^Q,\quad K = XW^K,\quad V = XW^V`}
              shapes={["X: [N,d_model]", "WQ,WK: [d_model,d_k]", "WV: [d_model,d_v]", "Q,K: [N,d_k]", "V: [N,d_v]"]}
              note="同一组 X 经过不同 W，变成三种不同角色。这里为演示用了具体的 W 值。"
              workout={
                <>
                  <div className="wo-title">逐项演算（以 x₁=[1,2] 为例，先展开 q₁ 的规则）</div>
                  <div className="wo-line">Wᵠ = [[1,0],[0,1]]　Wᵏ = [[0,1],[1,0]]　Wᵛ = [[1,1],[1,0]]</div>
                  <div className="wo-line">q₁[0] = 1×1 + 2×0 = 1</div>
                  <div className="wo-line">q₁[1] = 1×0 + 2×1 = 2　→　q₁ = <b>[1, 2]</b></div>
                  <div className="wo-line">同理 x₁·Wᵏ = [2,1] = k₁，x₁·Wᵛ = [3,1] = v₁</div>
                  <div className="wo-line">同理 x₂=[3,1] → q₂=[3,1], k₂=[1,3], v₂=[4,3]</div>
                </>
              }
            >
              <div className="pl-step">
                <div className="pl-title">输入 <b>X</b></div>
                <NumMatrix data={DEMO.X} />
                <div className="nm-caption">×Wᵠ / Wᵏ / Wᵛ ↓ 三分支</div>
                <div style={{ display: "flex", gap: 10, marginTop: 8, justifyContent: "center" }}>
                  <div className="pl-step" style={{ margin: 0 }}>
                    <NumMatrix data={DEMO_Q} />
                    <div className="nm-caption" style={{ color: "#f5b042" }}>Q</div>
                  </div>
                  <div className="pl-step" style={{ margin: 0 }}>
                    <NumMatrix data={DEMO_K} />
                    <div className="nm-caption" style={{ color: "#a78bfa" }}>K</div>
                  </div>
                  <div className="pl-step" style={{ margin: 0 }}>
                    <NumMatrix data={DEMO_V} />
                    <div className="nm-caption" style={{ color: "#2dd4bf" }}>V</div>
                  </div>
                </div>
              </div>
            </StepCard>

            <StepCard
              num="2"
              title="点积打分"
              color="#a78bfa"
              summary="拿 q₁ 去和每个 k 做点积，得到「相关度分数」α——两个向量越像，分数越高。"
              steps={[
                "固定查询 q₁=[1,2]。",
                "算 q₁·k₁、q₁·k₂ 的内积，再除以 √d_k 缩放。",
                "q₁ 与 k₂ 更像（4.95 > 2.83），所以更关注第 2 个 token。",
              ]}
              tex={String.raw`\alpha_{1,j} = \frac{q_1 \cdot k_j}{\sqrt{d_k}}`}
              shapes={["q₁: [d_k]", "K: [N,d_k]", "logits: [N]"]}
              note="除以 √d_k 是为了防止维度增大时分数过大、softmax 饱和（第 05 节详述）。"
              workout={
                <>
                  <div className="wo-title">逐项演算（d_k=2，√2≈1.414）</div>
                  <div className="wo-line">α₁,₁ = (1×2 + 2×1) / √2 = 4 / 1.414 = <span className="hl">2.83</span></div>
                  <div className="wo-line">α₁,₂ = (1×1 + 2×3) / √2 = 7 / 1.414 = <span className="hl">4.95</span></div>
                </>
              }
            >
              <div className="pl-step">
                <div className="pl-title">分数 <b>α₁,₁ / α₁,₂</b></div>
                <NumMatrix data={[[2.83, 4.95]]} heat />
              </div>
            </StepCard>

            <StepCard
              num="3"
              title="softmax → 权重"
              color="#38bdf8"
              summary="把这行分数过 softmax，变成非负、加起来等于 1 的「注意力权重」。"
              steps={[
                "对每个分数取指数。",
                "用这行所有指数之和做归一化。",
                "权重代表「生成 b₁ 时从每个位置取多少信息」。",
              ]}
              tex={String.raw`\hat\alpha_{1,j} = \frac{\exp(\alpha_{1,j})}{\sum_t \exp(\alpha_{1,t})},\quad \sum_j \hat\alpha_{1,j}=1`}
              shapes={["logits: [N]", "weights: [N]"]}
              workout={
                <>
                  <div className="wo-title">数值代入（用未舍入分数，2.83、4.95 仅为显示近似）</div>
                  <div className="wo-line">e^(4/√2) ≈ 16.919　　e^(7/√2) ≈ 141.139</div>
                  <div className="wo-line">总和 = 16.919 + 141.139 = <b>158.058</b></div>
                  <div className="wo-line">α̂₁,₁ = 16.919 / 158.058 = <span className="hl">0.107</span></div>
                  <div className="wo-line">α̂₁,₂ = 141.139 / 158.058 = <span className="hl">0.893</span></div>
                </>
              }
            >
              <div className="pl-step">
                <div className="pl-title">权重 <b style={{ color: "#38bdf8" }}>α̂₁</b>（和=1）</div>
                <NumMatrix data={[[0.107, 0.893]]} heat digits={3} />
              </div>
            </StepCard>

            <StepCard
              num="4"
              title="加权求和 → b₁"
              color="#2dd4bf"
              summary="用权重去乘对应的 v，全部加起来，就得到融合了整个序列信息的第一个输出 b₁。"
              steps={[
                "用步骤 3 的权重 α̂₁=[0.107, 0.893] 乘对应的 v。",
                "逐维相加得到 b₁。",
                "换 q₂ 重复步骤 2-4，就得到 b₂——所有 b 可并行计算。",
              ]}
              tex={String.raw`b_1 = \sum_{j=1}^{N} \hat\alpha_{1,j}\, v_j`}
              shapes={["weights: [N]", "V: [N,d_v]", "b₁: [d_v]"]}
              note="关键直觉：Q 和 K 决定「该关注谁」，V 决定「被关注后拿走的内容」。权重大的位置，它的 v 在 b 里占比就高。"
              workout={
                <>
                  <div className="wo-title">逐维演算（用三位小数近似权重 0.107、0.893）</div>
                  <div className="wo-line">b₁[0] = 0.107×3 + 0.893×4 = 0.321 + 3.572 = <span className="res">3.893</span></div>
                  <div className="wo-line">b₁[1] = 0.107×1 + 0.893×3 = 0.107 + 2.679 = <span className="res">2.786</span></div>
                  <div className="wo-line">∴ b₁ = [<span className="res">3.89, 2.79</span>]</div>
                </>
              }
            >
              <div className="pl-step">
                <div className="pl-title">输出 <b style={{ color: "#f472b6" }}>b₁</b></div>
                <NumMatrix data={[[3.89, 2.79]]} warm />
              </div>
            </StepCard>
          </section>

          {/* ===== 矩阵级 ===== */}
          <section className="section" id="s4">
            <SecHead idx="04" title="Self-Attention · 矩阵级（三步搞定）" />
            <p className="sec-lead">把所有词的 q/k/v 堆成矩阵 <Formula tex="Q, K, V" />，整件事就坍缩成<b style={{ color: "#eef3ff" }}>三次矩阵乘法</b>——这正是 GPU 最擅长、能大规模并行的形态。</p>
            <div className="note"><b>记号约定（先说清楚，避免和代码对不上）</b>：本文用 PyTorch 行向量约定 <Formula tex={String.raw`Q=XW^Q`} />，所以是 <Formula tex={String.raw`QK^{\mathsf T}`} />；有的教材用列向量 <Formula tex={String.raw`Q=W^Q I`} />，对应 <Formula tex={String.raw`K^{\mathsf T}Q`} />。两者数学等价，只差一个转置——这也是代码里写 <code>key.transpose(-2, -1)</code> 的原因。</div>
            <FigAttentionMatrix />
            <div className="eq-box">
              <Formula block tex={String.raw`\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V`} />
            </div>
            <FigNumPipeline />
            <div className="note">现在这句公式对你不再是一串符号：<Formula tex={String.raw`QK^{\mathsf T}`} /> 是「两两算相关度」，softmax 是「分数变权重」，乘 <Formula tex={String.raw`V`} /> 是「按权重取内容」。上面这组数值，正是把向量级 4 步压成矩阵后一次性算出的结果。</div>
          </section>

          {/* ===== 缩放 ===== */}
          <section className="section" id="s5">
            <SecHead idx="05" title={<>那个 <Formula tex={String.raw`\sqrt{d_k}`} /> 是干嘛的？</>} />
            <p className="sec-lead">公式里多了一个「除以 <Formula tex={String.raw`\sqrt{d_k}`} />」，叫<b style={{ color: "#eef3ff" }}>缩放（Scale）</b>。原因一句：维度 <Formula tex="d_k" /> 越大，点积数值越大，softmax 会被推向极端（一个 1、其余 0），梯度消失训不动。</p>
            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>不缩放会怎样</h3>
                <p className="t3">点积是 <Formula tex="d_k" /> 个乘积之和。<Formula tex="d_k" /> 大 → 点积方差大 → softmax 近似 one-hot → 梯度接近 0 → 训练停滞。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>除以 <Formula tex={String.raw`\sqrt{d_k}`} /> 的效果</h3>
                <p className="t3">把点积方差<b style={{ color: "#eef3ff" }}>拉回 1 附近</b>，让 softmax 处在温和区间，梯度健康。这是算子实现里最易漏、但必须有的细节。</p>
              </div>
            </div>
            <div className="note">
              <Formula block tex={String.raw`\operatorname{Var}(q_i k_j^{\mathsf T})\approx d_k\quad\Rightarrow\quad \operatorname{Var}\!\left(\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}\right)\approx1`} />
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
                  <div className="mname">scaled logits <span style={{ color: "#38bdf8" }}>α = q·k/sqrt(d)</span></div>
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
                  <div className="mname">权重 <span style={{ color: "#38bdf8" }}>softmax → 和=1</span></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.weights.map((v, i) => (
                      <div key={i} style={{ textAlign: "center" }}>
                        <div className="mcell" style={{ cursor: "default", background: `rgba(56,189,248,${0.12 + v * 0.6})`, borderColor: "#38bdf8" }}>{(v * 100).toFixed(0)}%</div>
                        <div className="mname" style={{ marginTop: 4 }}>k{i + 1}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <span className="msign">→</span>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">输出 <span style={{ color: "#f472b6" }}>z = Σ α̂·v</span></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.output.map((v, i) => (
                      <div key={i} className="mcell" style={{ cursor: "default", color: "#f472b6", borderColor: "#f472b6" }}>{v.toFixed(2)}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* ===== 多头 ===== */}
          <section className="section" id="s6">
            <SecHead idx="06" title="多头注意力（Multi-Head）" />
            <p className="sec-lead">只做一次 attention 只能学到「一种关注方式」。拆成<b style={{ color: "#eef3ff" }}>多个头</b>，各自独立算 attention，等于从多个角度（语法、长程、局部……）同时看序列，最后拼回来。</p>
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
                      {words.map((w) => <td key={w} style={{ textAlign: "center", fontFamily: "var(--mono)", fontSize: 12, color: "#6e7aab", width: 64 }}>{w}</td>)}
                    </tr>
                    {heads[headIdx].matrix.map((row, i) => (
                      <tr key={i}>
                        <td style={{ textAlign: "right", paddingRight: 10, fontFamily: "var(--mono)", fontSize: 12, color: "#6e7aab" }}>{words[i]}</td>
                        {row.map((v, j) => (
                          <td key={j}>
                            <div style={{
                              width: 58, height: 50, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
                              fontFamily: "var(--mono)", fontSize: 13, fontWeight: 700,
                              background: `rgba(56,189,248,${0.08 + v * 0.7})`, border: "1px solid rgba(56,189,248,0.3)",
                              color: v > 0.4 ? "#fff" : "#a9b4dc",
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
            <p className="sec-lead">把 Attention 装进完整模型长什么样？这就是被引用几万次的结构图——<b style={{ color: "#38bdf8" }}>左 Encoder</b>、<b style={{ color: "#f472b6" }}>右 Decoder</b>，各堆叠 N 层。</p>
            <FigTransformer />
            <div className="grid2">
              <div className="note"><b>Encoder</b>：对源序列做 Self-Attention + FFN，逐层提炼表示，最后把 K、V 交给 Decoder。</div>
              <div className="note"><b>Decoder</b>：先用 <b>Masked</b> Self-Attention（屏蔽未来位防作弊），再通过 <b>Cross-Attention</b> 读取编码器的 K/V，最后预测下一个词。</div>
            </div>
            <div className="note warn">这张图里 <b>Attention 出现了三次</b>（Encoder 自注意、Decoder 掩码自注意、Decoder 交叉注意）。我们前面学的那个公式，是这三处共用的同一个算子——这就是它"为什么重要"的最终答案。</div>

            <h3>为什么是 Add &amp; Norm，且用 LayerNorm</h3>
            <p className="sec-lead">每个子层都套着 <b style={{ color: "#eef3ff" }}>残差连接 + 归一化</b>：<Formula tex={String.raw`y=\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))`} />。残差让深层可训，归一化稳定数值。而归一化为什么选 LayerNorm 而非更常见的 BatchNorm？</p>
            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>BatchNorm（跨样本）</h3>
                <p className="t3">对<b>同一特征、跨 batch 内所有样本</b>统计均值方差。依赖 batch 大小——batch 小或序列长度变化时，统计量不稳。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>LayerNorm（单样本）</h3>
                <p className="t3">对<b>单个 token 的全部特征维</b>统计，不依赖 batch 中其他样本。<b style={{ color: "#eef3ff" }}>序列长度、batch 大小都能变</b>，所以 Transformer 选它。</p>
              </div>
            </div>

            <h3>Decoder 的 Mask：为什么训练能并行</h3>
            <p className="sec-lead">Decoder 生成时要"看到过去、看不到未来"。实现上用一个<b style={{ color: "#f5b042" }}>下三角因果掩码（Causal Mask）</b>：第 i 个位置只允许看第 0..i 个 Key。</p>

            <div className="mask-grid">
              <table>
                <tbody>
                  <tr>
                    <th></th>
                    <th>k₀<br /><small style={{ color: "#4d577f" }}>&lt;BOS&gt;</small></th>
                    <th>k₁<br /><small style={{ color: "#4d577f" }}>I</small></th>
                    <th>k₂<br /><small style={{ color: "#4d577f" }}>have</small></th>
                    <th>k₃<br /><small style={{ color: "#4d577f" }}>a</small></th>
                    <th>k₄<br /><small style={{ color: "#4d577f" }}>cat</small></th>
                  </tr>
                  <tr>
                    <th>q₀ &lt;BOS&gt;</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th>q₁ I</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th>q₂ have</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th>q₃ a</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th>q₄ cat</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="fig-cap">因果掩码矩阵：绿色 <b style={{ color: "#34d399" }}>0</b> = 允许看，红色 <b style={{ color: "#f5b042" }}>−∞</b> = 屏蔽。第 i 行只允许列 0..i</div>

            <h3 style={{ marginTop: 30 }}>Mask 加在哪一步</h3>
            <div className="flow-chain">
              <b>Q·Kᵀ</b><em>→</em>
              <b>÷ <Formula tex={String.raw`\sqrt{d_k}`} /></b><em>→</em>
              <b className="hi">+ Mask（−∞）</b><em>→</em>
              <b>softmax</b><em>→</em>
              <b>· V</b>
            </div>
            <div className="eq-box">
              <Formula block tex={String.raw`A=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V,\quad M_{ij}=\begin{cases}0 & i\ge j\\ -\infty & i<j\end{cases}`} />
            </div>
            <div className="note warn"><b>实现陷阱</b>：Mask 必须在 softmax <b>之前</b>加 <code>−∞</code>。若在 softmax 后再乘 0，屏蔽位虽然归零，但剩余权重之和不再为 1，输出尺度会出错。</div>

            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>推理（逐 token，串行）</h3>
                <p className="t3" style={{ fontFamily: "var(--mono)", fontSize: 13 }}>
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt;</b> → 预测 I<br />
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt; I</b> → 预测 have<br />
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt; I have</b> → 预测 a<br />
                  ……直到 <b style={{ color: "#f5b042" }}>&lt;end&gt;</b>
                </p>
                <p className="t3">每步只能用已生成的内容，天生串行。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>训练（整句并行）</h3>
                <p className="t3" style={{ fontFamily: "var(--mono)", fontSize: 13 }}>
                  输入 <b style={{ color: "#2dd4bf" }}>&lt;BOS&gt; I have a cat</b>（右移一位）<br />
                  目标 <b style={{ color: "#f472b6" }}>I have a cat &lt;end&gt;</b><br />
                  一次前向 + 因果 Mask
                </p>
                <p className="t3">右移序列 + Causal Mask 让训练像 Encoder 一样并行，但每个位置"假装只看到过去"。</p>
              </div>
            </div>
          </section>

          {/* ===== 位置编码 ===== */}
          <section className="section" id="s8">
            <h3>位置编码</h3>
            <p className="sec-lead">Self-Attention 对「顺序」无感——「我打他」和「他打我」算出来一样。所以在输入 embedding 上<b style={{ color: "#eef3ff" }}>直接加一个位置向量</b>，把顺序信息喂回去。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\quad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)`} />
            </div>
            <div className="note">用不同频率的正余弦，让每个位置拿到<b>唯一</b>编码；且对任意固定间距 <Formula tex="k" />，<Formula tex={String.raw`PE_{pos+k}`} /> 是 <Formula tex={String.raw`PE_{pos}`} /> 的线性函数——模型因此能泛化到比训练更长的序列。</div>
          </section>

          {/* ===== 代码 ===== */}
          <section className="section" id="s9">
            <SecHead idx="08" title="经典代码 + 算子测试要点" />
            <p className="sec-lead">原理看懂了，落到代码就几十行。下面是哈佛 The Annotated Transformer 的经典实现，逐行对应步骤。</p>

            <div className="code-title">① 缩放点积注意力 — 对应 softmax(QKᵀ/sqrt(dk))V</div>
            <pre><code>{`def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
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

            <h3>算子测试要点</h3>
            <div className="grid2">
              <div className="card"><h3 style={{ marginTop: 0 }}>数值正确性</h3><p className="t3">小矩阵手算 softmax(QKᵀ/sqrt(dk))V 比对；验证每行和=1、padding 权重≈0。</p></div>
              <div className="card"><h3 style={{ marginTop: 0 }}>形状与边界</h3><p className="t3">校验 (B,h,n,dₖ) 变换；d_model 不能被 h 整除时报错、空序列、单 token。</p></div>
              <div className="card"><h3 style={{ marginTop: 0 }}>mask 正确性</h3><p className="t3">因果 mask 为下三角（含主对角线）；屏蔽位 softmax 后为 0，不受未来影响。</p></div>
              <div className="card"><h3 style={{ marginTop: 0 }}>性能与精度</h3><p className="t3">fp32/fp16/bf16 相对误差（&lt;1e-3）；显存与耗时随 seq/heads/dₖ 的曲线。</p></div>
            </div>

            <div className="note ok"><b>一句话总结</b>：Attention 把「每个位置该关注谁」变成 <Formula tex={String.raw`QK^{\mathsf T}\!/\sqrt{d_k}`} /> 算分、softmax 变权重、再乘 <Formula tex="V" /> 取内容——三步矩阵乘法。多头扩展视角，加位置编码补顺序，组装成 Encoder/Decoder，就是撑起所有现代大模型的 Transformer。</div>
          </section>

          <div className="foot">
            参考：Vaswani et al. <i>Attention Is All You Need</i>（Figure 1）· The Annotated Transformer。<br />
            全篇 Q/K/V/Attention/Output 配色一致，SVG 可自由放大查看。
          </div>

        </div>
      </main>
    </>
  );
}
