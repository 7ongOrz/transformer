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

/* ---------- Attention 演示数据（与 s3/s4 主线统一：我/爱/深/度，d=2） ---------- */
const words = ["token₁", "token₂", "token₃", "token₄"];
const queries = [
  [0.04, 1.16],
  [1.41, 0.99],
  [0.53, 1.12],
  [0.65, 1.75],
];
const keys = [
  [0.56, 1.16],
  [1.26, -0.27],
  [0.82, 0.67],
  [1.18, 1.21],
];
const values = [
  [0.4, 1.28],
  [1.08, 0.6],
  [0.65, 1.06],
  [0.92, 1.72],
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
 * 向量级四阶段图 + 矩阵级收束（统一 4-token 数据：我/爱/深/度，d=2）
 * ============================================================ */

/* ============================================================
 * 向量阶段图1：X → Q/K/V 投影
 * 统一数据：我/爱/深/度，d=2
 * ============================================================ */

function QKVMat({
  data,
  accent,
  label,
  sub,
  rowLabels,
  heroRow = -1,
  heroColor,
}: {
  data: number[][];
  accent: string;
  label?: string;
  sub?: string;
  rowLabels?: string[];
  heroRow?: number;   // 高亮的主角行下标，-1 表示无
  heroColor?: string;
}) {
  const hc = heroColor ?? accent;
  const cellStyle: React.CSSProperties = {
    width: 54,
    minWidth: 54,
    height: 42,
    textAlign: "center",
    verticalAlign: "middle",
    fontFamily: "var(--mono)",
    fontSize: 15,
    fontWeight: 600,
    color: "var(--t1)",
    border: "1px solid var(--hairline)",
    background: `${accent}14`,
    boxSizing: "border-box",
  };
  return (
    <div style={{ display: "inline-flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      {label ? (
        <div
          style={{
            color: accent,
            fontWeight: 700,
            fontSize: 13,
            fontFamily: "var(--mono)",
            letterSpacing: 0.4,
          }}
        >
          {label}
        </div>
      ) : null}
      <div style={{ display: "inline-flex", alignItems: "stretch" }}>
        {rowLabels ? (
          <table style={{ borderCollapse: "collapse", marginRight: 8 }}>
            <tbody>
              {rowLabels.map((w, i) => (
                <tr key={i}>
                  <td
                    style={{
                      height: 42,
                      width: 30,
                      textAlign: "right",
                      paddingRight: 6,
                      verticalAlign: "middle",
                      fontFamily: "var(--mono)",
                      fontSize: 13,
                      color: i === heroRow ? hc : "var(--t2)",
                      fontWeight: i === heroRow ? 800 : 400,
                      border: "1px solid transparent",
                      boxSizing: "border-box",
                    }}
                  >
                    {w}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
        {/* 左括号 */}
        <div
          style={{
            width: 9,
            alignSelf: "stretch",
            borderLeft: `3px solid ${accent}`,
            borderTop: `3px solid ${accent}`,
            borderBottom: `3px solid ${accent}`,
            borderTopLeftRadius: 5,
            borderBottomLeftRadius: 5,
          }}
        />
        <table style={{ borderCollapse: "collapse" }}>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td
                    key={j}
                    style={
                      i === heroRow
                        ? {
                            ...cellStyle,
                            fontWeight: 800,
                            color: "#fff",
                            border: `1px solid ${hc}`,
                            background: `${hc}33`,
                          }
                        : cellStyle
                    }
                  >
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {/* 右括号 */}
        <div
          style={{
            width: 9,
            alignSelf: "stretch",
            borderRight: `3px solid ${accent}`,
            borderTop: `3px solid ${accent}`,
            borderBottom: `3px solid ${accent}`,
            borderTopRightRadius: 5,
            borderBottomRightRadius: 5,
          }}
        />
      </div>
      {sub ? (
        <div style={{ color: "var(--t3)", fontSize: 11, fontFamily: "var(--mono)" }}>{sub}</div>
      ) : null}
    </div>
  );
}

function FigStageQKV() {
  const WORDS = ["token₁", "token₂", "token₃", "token₄"];
  const X = [[0.4, 1.2], [1.5, 0.3], [0.8, 0.9], [1.1, 1.5]];
  const WQ = [[1, 0.5], [-0.3, 0.8]];
  const WK = [[0.8, -0.4], [0.2, 1.1]];
  const WV = [[0.7, 0.2], [0.1, 1]];
  const Q = [[0.04, 1.16], [1.41, 0.99], [0.53, 1.12], [0.65, 1.75]];
  const K = [[0.56, 1.16], [1.26, -0.27], [0.82, 0.67], [1.18, 1.21]];
  const V = [[0.4, 1.28], [1.08, 0.6], [0.65, 1.06], [0.92, 1.72]];

  const CQ = "#f5b042"; // --q
  const CK = "#a78bfa"; // --k
  const CV = "#2dd4bf"; // --v
  const NEUTRAL = "#6e7aab"; // --t3，X 作为共享输入用中性色

  type RowDef = {
    key: "Q" | "K" | "V";
    color: string;
    wlabel: string;
    W: number[][];
    R: number[][];
    tex: string;
    note: string;
  };
  const rows: RowDef[] = [
    {
      key: "Q",
      color: CQ,
      wlabel: "Wᵠ",
      W: WQ,
      R: Q,
      tex: String.raw`Q = XW^Q`,
      note: "查询向量 qᵢ = xᵢ·Wᵠ → 决定「我想找什么」",
    },
    {
      key: "K",
      color: CK,
      wlabel: "Wᵏ",
      W: WK,
      R: K,
      tex: String.raw`K = XW^K`,
      note: "键向量 kᵢ = xᵢ·Wᵏ → 决定「我有什么可被匹配」",
    },
    {
      key: "V",
      color: CV,
      wlabel: "Wᵛ",
      W: WV,
      R: V,
      tex: String.raw`V = XW^V`,
      note: "值向量 vᵢ = xᵢ·Wᵛ → 决定「匹配上后拿走的内容」",
    },
  ];

  return (
    <div
      role="img"
      aria-label="向量阶段投影图：X 分别乘 WQ/WK/WV 得到 Q/K/V"
      style={{
        background: "var(--panel-3)",
        border: "1px solid var(--hairline)",
        borderRadius: 14,
        padding: "20px 18px",
        margin: "20px 0",
        overflowX: "auto",
      }}
    >
      {/* 标题 + 公式条 */}
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ color: "var(--t1)", fontWeight: 700, fontSize: 15, marginBottom: 10 }}>
          向量阶段 · 投影：同一份 <span style={{ color: "var(--t2)" }}>X</span> 乘三个权重矩阵，得到{" "}
          <span style={{ color: CQ }}>Q</span> / <span style={{ color: CK }}>K</span> /{" "}
          <span style={{ color: CV }}>V</span>
        </div>
        <div
          style={{
            display: "inline-flex",
            gap: 28,
            flexWrap: "wrap",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <Formula tex={String.raw`Q = XW^Q`} />
          <Formula tex={String.raw`K = XW^K`} />
          <Formula tex={String.raw`V = XW^V`} />
        </div>
      </div>

      {/* 共享 X + 三个投影分支 */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "center", gap: 20, flexWrap: "wrap" }}>
        {/* 左侧：共享输入 X（只出现一次） */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8, paddingTop: 36 }}>
          <QKVMat data={X} accent={NEUTRAL} label="X" sub="[4×2] 共享输入" rowLabels={WORDS} heroRow={0} heroColor={CQ} />
          <div style={{ color: "var(--t3)", fontSize: 11, fontFamily: "var(--mono)" }}>同一份输入，同时送入三路 ↓</div>
        </div>

        {/* 右侧：三个投影分支 */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {rows.map((r) => (
            <div
              key={r.key}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 12,
                flexWrap: "wrap",
                padding: "12px 12px",
                borderRadius: 10,
                background: `${r.color}0d`,
                border: `1px solid ${r.color}40`,
              }}
            >
              <span style={{ fontSize: 16, color: "var(--t2)", fontWeight: 700, fontFamily: "var(--mono)" }}>X</span>
              <span style={{ fontSize: 20, color: "var(--t3)", fontWeight: 600, fontFamily: "var(--mono)" }}>×</span>
              <QKVMat data={r.W} accent={r.color} label={r.wlabel} sub="[2×2]" />
              <span style={{ fontSize: 20, color: "var(--t3)", fontWeight: 600, fontFamily: "var(--mono)" }}>=</span>
              <QKVMat data={r.R} accent={r.color} label={r.key} sub="[4×2]" heroRow={r.key === "Q" ? 0 : -1} heroColor={r.color} />
              <div
                style={{
                  borderLeft: "1px solid var(--hairline)",
                  paddingLeft: 12,
                  display: "flex",
                  flexDirection: "column",
                  gap: 4,
                  minWidth: 168,
                }}
              >
                <Formula block tex={r.tex} />
                <div style={{ fontSize: 11.5, color: "var(--t3)", lineHeight: 1.5 }}>{r.note}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 逐项演算示例（以 x₁=[0.4,1.2] 为例，呼应主角 q₁）*/}
      <div
        style={{
          marginTop: 16,
          display: "flex",
          justifyContent: "center",
          gap: 18,
          flexWrap: "wrap",
          color: "var(--t3)",
          fontSize: 12,
          fontFamily: "var(--mono)",
        }}
      >
        <span>例：x₁ = [0.4,1.2]</span>
        <span style={{ color: CQ }}>→ q₁ = [0.4×1.0+1.2×(-0.3), 0.4×0.5+1.2×0.8] = [0.04,1.16] ★</span>
        <span style={{ color: CK }}>→ k₁ = [0.4×0.8+1.2×0.2, 0.4×(-0.4)+1.2×1.1] = [0.56,1.16]</span>
        <span style={{ color: CV }}>→ v₁ = [0.4×0.7+1.2×0.1, 0.4×0.2+1.2×1.0] = [0.40,1.28]</span>
      </div>
      <div style={{ textAlign: "center", color: "var(--t3)", fontSize: 12, marginTop: 10 }}>
        图 · 同一组输入 <b style={{ color: "var(--t2)" }}>X</b> 经三个独立可学习的权重矩阵，投影成 Q / K / V；
        下一章将固定第 1 行的 <b style={{ color: CQ }}>q₁ = [0.04,1.16]</b> 作为主角，追踪它如何算出 b₁。
      </div>
    </div>
  );
}

/* ============================================================
 * 向量阶段图2：固定 q₁ 扇出打分（FigStageScore）
 * 主角 q₁ 固定左侧（★），向 k₁..k₄ 扇出橙色连线，
 * 每条连线终点标注点积分数 α₁,ⱼ；关键连线上写出 q₁·kⱼ 的乘加过程。
 * 数据全程使用统一 4-token（我/爱/深/度，d=2）。
 * ============================================================ */
function FigStageScore() {
  // —— 统一数据（token₁/token₂/token₃/token₄，d=2，√dₖ≈1.414）——
  const q1: [number, number] = [0.04, 1.16];
  const ks: { label: string; word: string; v: [number, number]; a: number; calc: string }[] = [
    { label: "k₁", word: "token₁", v: [0.56, 1.16], a: 0.967, calc: "(0.04×0.56 + 1.16×1.16)/√2" },
    { label: "k₂", word: "token₂", v: [1.26, -0.27], a: -0.186, calc: "(0.04×1.26 + 1.16×-0.27)/√2" },
    { label: "k₃", word: "token₃", v: [0.82, 0.67], a: 0.573, calc: "(0.04×0.82 + 1.16×0.67)/√2" },
    { label: "k₄", word: "token₄", v: [1.18, 1.21], a: 1.026, calc: "(0.04×1.18 + 1.16×1.21)/√2" },
  ];
  const MAX_A = 1.026;

  // —— 布局坐标 ——
  const YS = [100, 225, 350, 475]; // 4 个 k 块中心 y
  const qY = 287;                  // q1 块中心 y（≈ YS 首尾中点）
  const qRight = 190;              // q1 块右边 x（连线起点）
  const kLeft = 710;               // k 块左边 x（连线终点）
  const pillX = (qRight + kLeft) / 2; // 分数药丸中心 x = 450

  return (
    <div className="fig">
      <svg viewBox="0 0 940 580" width="940" role="img" aria-label="固定 q1 扇出打分：q1 向 k1..k4 计算点积分数">
        <defs>
          <marker id="ss-ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
            <path d="M0,0 L8,4.5 L0,9 z" fill="#f5b042" />
          </marker>
        </defs>

        {/* 标题 */}
        <text x="30" y="34" fill="#c8d4ff" fontSize="15" fontWeight="700">
          固定 q₁，向所有 k 扇出打分
        </text>
        <text x="30" y="55" fill="#a9b4dc" fontSize="12.5">
          α₁,ⱼ = q₁ · kⱼ / √dₖ　（d=2，√dₖ≈1.414）　点积越大 → 越像 → 注意力分数越高
        </text>

        {/* 扇出连线（先画线，后画块以遮住端点） */}
        {ks.map((k, i) => {
          const ky = YS[i];
          const isTop = Math.abs(k.a - MAX_A) < 1e-6;
          return (
            <path
              key={`ln-${i}`}
              d={`M${qRight},${qY} C${pillX},${qY} ${pillX},${ky} ${kLeft},${ky}`}
              stroke="#f5b042"
              strokeWidth={isTop ? 2.4 : 1.5}
              fill="none"
              markerEnd="url(#ss-ah)"
              opacity={isTop ? 0.95 : 0.5}
            />
          );
        })}

        {/* q₁ 主角块（左侧，高亮） */}
        <g>
          <rect x="40" y={qY - 54} width="150" height="108" rx="14" fill="rgba(245,176,66,0.2)" stroke="#f5b042" strokeWidth="2.4" />
          <text x="115" y={qY - 20} textAnchor="middle" fill="#f5b042" fontSize="22" fontWeight="800" fontFamily="JetBrains Mono, monospace">q₁ ★</text>
          <text x="115" y={qY + 8} textAnchor="middle" fill="#fbbf24" fontSize="15" fontFamily="JetBrains Mono, monospace" fontWeight="700">[ {q1[0]}, {q1[1]} ]</text>
          <text x="115" y={qY + 33} textAnchor="middle" fill="#a9b4dc" fontSize="11.5">主角 Query · token₁</text>
        </g>

        {/* k 块 + 分数药丸 + 乘加过程 */}
        {ks.map((k, i) => {
          const ky = YS[i];
          const isTop = Math.abs(k.a - MAX_A) < 1e-6;
          const pillY = (qY + ky) / 2; // 曲线 t=0.5 恰好穿过此点
          // 在 k₁（最低分，含 0 项）与 k₂（最高分）两条连线上写出乘加过程
          const showCalc = i === 0 || i === 1;
          return (
            <g key={`k-${i}`}>
              {/* k 块 */}
              <rect
                x={kLeft}
                y={ky - 36}
                width="160"
                height="72"
                rx="11"
                fill={isTop ? "rgba(167,139,250,0.22)" : "rgba(167,139,250,0.09)"}
                stroke="#a78bfa"
                strokeWidth={isTop ? 2 : 1.2}
              />
              <text x={kLeft + 80} y={ky - 10} textAnchor="middle" fill="#a78bfa" fontSize="16" fontWeight="700" fontFamily="JetBrains Mono, monospace">{k.label}</text>
              <text x={kLeft + 80} y={ky + 12} textAnchor="middle" fill="#c4b5fd" fontSize="13" fontFamily="JetBrains Mono, monospace" fontWeight="700">[ {k.v[0]}, {k.v[1]} ]</text>
              <text x={kLeft + 80} y={ky + 31} textAnchor="middle" fill="#6e7aab" fontSize="11">Key · {k.word}</text>

              {/* 分数药丸 */}
              <rect
                x={pillX - 60}
                y={pillY - 17}
                width="120"
                height="34"
                rx="17"
                fill={isTop ? "rgba(56,189,248,0.3)" : "rgba(56,189,248,0.14)"}
                stroke="#38bdf8"
                strokeWidth={isTop ? 1.9 : 1.2}
              />
              <text x={pillX} y={pillY + 5} textAnchor="middle" fill="#7dd3fc" fontSize="14" fontFamily="JetBrains Mono, monospace" fontWeight="700">
                α₁,{i + 1} = {k.a.toFixed(3)}
              </text>

              {/* 乘加过程（仅 k₁、k₂） */}
              {showCalc && (
                <text x={pillX} y={pillY + 32} textAnchor="middle" fill="#a9b4dc" fontSize="11.5" fontFamily="JetBrains Mono, monospace">
                  {k.calc} = {k.a.toFixed(3)}
                </text>
              )}
            </g>
          );
        })}

        {/* 底部一句话结论 */}
        <text x="470" y="555" textAnchor="middle" fill="#6e7aab" fontSize="12.5">
          q₁ 与 <tspan fill="#a78bfa" fontWeight="700">token₄</tspan> 最像（分数最高 1.026）→ softmax 后会重点看向「token₄」
        </text>
      </svg>
      <div className="fig-cap">图 · 向量阶段 ② 打分：固定主角 q₁ 向所有 k 扇出，点积分数量化「谁更像我」</div>
    </div>
  );
}

/* ============================================================
 * 向量阶段图3：整行 softmax（row-wise softmax）
 * 强调：4 个分数作为「一整行」联合归一化，而非 4 个独立操作
 * 数据：q1 主角行 alphas=[0.967,-0.186,0.573,1.026]  ahats=[0.328,0.103,0.221,0.348]
 * ============================================================ */
function FigStageSoftmax() {
  // 分数用 3 位小数（与矩阵级 S、e^α 计算口径一致）
  const alphas = [0.967, -0.186, 0.573, 1.026];
  const ahats = [0.328, 0.103, 0.221, 0.348];
  const exps = ["2.63", "0.83", "1.77", "2.79"];
  const Z = "8.02";

  // 单元格几何
  const cellW = 56;
  const cellH = 74;
  const cellY = 110;
  const scoreX = [36, 98, 160, 222];
  const weightX = [640, 702, 764, 826];

  // "总线箭头"：一根粗宽箭头，表示整行一起流动（整体进 / 整体出）
  const busArrow = (x1: number, x2: number, y: number) => {
    const head = 14;
    const sh = 16; // shaft half-height
    const hh = 13; // head half-height
    const xs = x2 - head;
    return (
      <polygon
        points={`${x1},${y - sh} ${xs},${y - sh} ${xs},${y - hh} ${x2},${y} ${xs},${y + hh} ${xs},${y + sh} ${x1},${y + sh}`}
        fill="rgba(56,189,248,0.22)"
        stroke="#38bdf8"
        strokeWidth={1.6}
      />
    );
  };

  return (
    <div className="fig">
      <svg viewBox="0 0 920 250" width="920" role="img" aria-label="整行 softmax：4 个分数联合归一化">
        {/* 标题 */}
        <text x="460" y="30" textAnchor="middle" fill="#eef3ff" fontSize="14" fontWeight="700">
          整行 softmax · 4 个分数作为「一行」整体联合归一化
        </text>
        <text x="460" y="50" textAnchor="middle" fill="#a9b4dc" fontSize="11">
          整行一起进 → 整行一起出（非 4 个独立的逐元素 softmax）
        </text>

        {/* 两行的列标签 */}
        <text x="160" y="92" textAnchor="middle" fill="#7dd3fc" fontSize="11" fontFamily="JetBrains Mono,monospace" fontWeight="700">
          α₁,ⱼ = q₁·kⱼ / √dₖ
        </text>
        <text x="764" y="92" textAnchor="middle" fill="#7dd3fc" fontSize="11" fontFamily="JetBrains Mono,monospace" fontWeight="700">
          α̂₁,ⱼ  （Σⱼ α̂₁,ⱼ ≈ 1）
        </text>

        {/* 分数行容器（整行） */}
        <rect x="30" y="100" width="260" height="92" rx="10" fill="#0c1430" stroke="rgba(255,255,255,0.10)" />
        {alphas.map((a, i) => (
          <g key={`s${i}`}>
            <rect x={scoreX[i]} y={cellY} width={cellW} height={cellH} rx={6} fill="rgba(56,189,248,0.10)" stroke="#38bdf8" />
            <text x={scoreX[i] + cellW / 2} y={cellY + 26} textAnchor="middle" fontSize="11" fill="#7dd3fc" fontFamily="JetBrains Mono,monospace" fontWeight="700">α₁,{i + 1}</text>
            <text x={scoreX[i] + cellW / 2} y={cellY + 54} textAnchor="middle" fontSize="16" fill="#c8d4ff" fontFamily="JetBrains Mono,monospace" fontWeight="700">{a.toFixed(3)}</text>
          </g>
        ))}
        {/* 分数行下方括号：强调「一整行」 */}
        <path d="M30,200 L30,208 L290,208 L290,200" fill="none" stroke="#6e7aab" strokeWidth="1.2" />
        <text x="160" y="224" textAnchor="middle" fill="#a9b4dc" fontSize="10.5">整行 4 个分数 · 一起送入</text>

        {/* 总线箭头 1：整行分数 → softmax（单根粗箭头） */}
        {busArrow(294, 370, 147)}
        <text x="332" y="132" textAnchor="middle" fill="#38bdf8" fontSize="10" fontFamily="JetBrains Mono,monospace" fontWeight="700">整行</text>

        {/* softmax 大框：单一算子，横跨整行 */}
        <rect x="374" y="86" width="176" height="122" rx="12" fill="rgba(56,189,248,0.14)" stroke="#38bdf8" strokeWidth={2.4} />
        <text x="462" y="110" textAnchor="middle" fill="#7dd3fc" fontSize="12" fontFamily="JetBrains Mono,monospace" fontWeight="700">row-wise</text>
        <text x="462" y="136" textAnchor="middle" fill="#38bdf8" fontSize="18" fontWeight="800">SOFTMAX</text>
        <text x="462" y="158" textAnchor="middle" fill="#eef3ff" fontSize="11">整行联合归一化</text>
        <text x="462" y="178" textAnchor="middle" fill="#a9b4dc" fontSize="10" fontFamily="JetBrains Mono,monospace">Z = Σⱼ e^α₁,ⱼ  ·  α̂ = e^α / Z</text>
        <text x="462" y="195" textAnchor="middle" fill="#6e7aab" fontSize="9.5">（一个算子，同时吃整行）</text>

        {/* 总线箭头 2：softmax → 整行权重（单根粗箭头） */}
        {busArrow(554, 630, 147)}
        <text x="592" y="132" textAnchor="middle" fill="#38bdf8" fontSize="10" fontFamily="JetBrains Mono,monospace" fontWeight="700">整行</text>

        {/* 权重行容器（整行，热力色 = 归一化强度） */}
        <rect x="634" y="100" width="260" height="92" rx="10" fill="#0c1430" stroke="rgba(255,255,255,0.10)" />
        {ahats.map((w, i) => {
          const t = Math.min(1, w / 0.4);
          const op = 0.12 + t * 0.6;
          const col = t > 0.5 ? "#ffffff" : "#c8d4ff";
          return (
            <g key={`w${i}`}>
              <rect x={weightX[i]} y={cellY} width={cellW} height={cellH} rx={6} fill={`rgba(56,189,248,${op.toFixed(3)})`} stroke="#38bdf8" />
              <text x={weightX[i] + cellW / 2} y={cellY + 26} textAnchor="middle" fontSize="11" fill="#7dd3fc" fontFamily="JetBrains Mono,monospace" fontWeight="700">α̂₁,{i + 1}</text>
              <text x={weightX[i] + cellW / 2} y={cellY + 54} textAnchor="middle" fontSize="16" fill={col} fontFamily="JetBrains Mono,monospace" fontWeight="700">{w.toFixed(3)}</text>
            </g>
          );
        })}
        {/* 权重行下方括号：强调「一整行 · 和为 1」 */}
        <path d="M634,200 L634,208 L894,208 L894,200" fill="none" stroke="#6e7aab" strokeWidth="1.2" />
        <text x="764" y="224" textAnchor="middle" fill="#a9b4dc" fontSize="10.5">整行 4 个权重 · 三位小数近似，和 = 1</text>
      </svg>

      {/* 数值代入（softmax 层下方）*/}
      <div style={{ marginTop: 14, padding: "14px 16px", background: "var(--panel-3)", borderRadius: 12, border: "1px solid var(--hairline)" }}>
        <div style={{ color: "var(--t1)", fontSize: 13, fontWeight: 700, marginBottom: 10 }}>
          数值代入 · q₁ 这一行
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, flexWrap: "wrap" }}>
          {alphas.map((a, i) => (
            <div key={`e${i}`} style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", width: 92, height: 60, borderRadius: 10, background: "rgba(56,189,248,0.08)", border: "1px solid var(--hairline)" }}>
              <span style={{ color: "var(--t3)", fontSize: 11, fontFamily: "JetBrains Mono,monospace" }}>e^{a.toFixed(3)}</span>
              <span style={{ color: "var(--att)", fontSize: 15, fontWeight: 700, fontFamily: "JetBrains Mono,monospace" }}>≈ {exps[i]}</span>
            </div>
          ))}
          <span style={{ color: "var(--t3)", fontSize: 18, fontWeight: 700 }}>+</span>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", width: 112, height: 60, borderRadius: 10, background: "rgba(56,189,248,0.16)", border: "1.5px solid var(--att)" }}>
            <span style={{ color: "var(--t2)", fontSize: 11, fontFamily: "JetBrains Mono,monospace" }}>Z = Σ e^α</span>
            <span style={{ color: "var(--att)", fontSize: 16, fontWeight: 700, fontFamily: "JetBrains Mono,monospace" }}>≈ {Z}</span>
          </div>
        </div>

        <div style={{ marginTop: 12, display: "block", maxWidth: "100%" }}>
          <div style={{ marginBottom: 6, overflowX: "auto", overflowY: "hidden" }}>
            <Formula block tex={"\\hat\\alpha_{1,j}=\\dfrac{e^{\\alpha_{1,j}}}{Z}=\\dfrac{e^{\\alpha_{1,j}}}{\\sum_{j=1}^{4}e^{\\alpha_{1,j}}},\\qquad Z=2.63+0.83+1.77+2.79\\approx 8.02"} />
          </div>
          <div style={{ overflowX: "auto", overflowY: "hidden" }}>
            <Formula block tex={"\\hat\\alpha_{1,:}=\\dfrac{[\\,2.63,\\ 0.83,\\ 1.77,\\ 2.79\\,]}{8.02}\\approx[\\,0.328,\\ 0.103,\\ 0.221,\\ 0.348\\,],\\qquad \\sum_{j}\\hat\\alpha_{1,j}\\approx 1\\ (\\text{三位小数近似})"} />
          </div>
        </div>

        <div style={{ marginTop: 10, color: "var(--t2)", fontSize: 12, lineHeight: 1.6 }}>
          <b style={{ color: "var(--att)" }}>关键</b>：分母 <span style={{ fontFamily: "JetBrains Mono,monospace", color: "var(--att)" }}>Z</span> 是整行 4 个
          <span style={{ fontFamily: "JetBrains Mono,monospace" }}> e^α </span>之和 —— 每个权重都依赖整行所有分数。所以 softmax 是
          <b style={{ color: "var(--t1)" }}>「整行联合归一化」</b>，而非 4 个独立的逐元素运算；改动任意一个分数，整行权重都会重新分布。
        </div>
      </div>

      <div className="fig-cap">
        图 · 整行 softmax：4 个分数打包成「一行」整体送入单一算子，输出 4 个和为 1 的权重。整体进、整体出。
      </div>
    </div>
  );
}

function FigStageAggregate() {
  // 统一数据：q1 主角路径，权重用三位小数近似
  const ahats3 = [0.328, 0.103, 0.221, 0.348];
  const vvals = ["[0.40,1.28]", "[1.08,0.60]", "[0.65,1.06]", "[0.92,1.72]"];
  // 乘积 α̂·v（保留 2 位便于标注；最终 b1 用三位小数近似权重算）
  const products = [
    ["0.13", "0.42"],
    ["0.11", "0.06"],
    ["0.14", "0.23"],
    ["0.32", "0.60"],
  ];
  const ys = [150, 230, 310, 390];
  const b1y = (ys[0] + ys[3]) / 2; // 270 扇入汇聚点

  const wrap: React.CSSProperties = {
    width: "100%",
    maxWidth: 1040,
    margin: "0 auto",
    background: "rgba(12,20,48,0.5)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 14,
    padding: "18px 16px 22px",
    boxSizing: "border-box",
    overflowX: "auto",
  };
  const svgStyle: React.CSSProperties = { width: "100%", minWidth: 720, height: "auto", display: "block" };
  const workoutWrap: React.CSSProperties = {
    marginTop: 14,
    padding: "14px 18px",
    background: "rgba(7,11,24,0.6)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 10,
    fontFamily: "var(--mono)",
  };
  const line: React.CSSProperties = {
    display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px 8px",
    fontSize: 15, lineHeight: 1.9,
  };
  const lbl: React.CSSProperties = { color: "#a9b4dc", minWidth: 52 };
  const op: React.CSSProperties = { color: "#6e7aab" };
  const note: React.CSSProperties = {
    marginTop: 10, fontSize: 12.5, color: "#6e7aab", lineHeight: 1.7,
  };

  return (
    <div style={wrap}>
      <svg viewBox="0 0 1040 500" style={svgStyle} role="img" aria-label="权重连向 v，扇入汇聚成 b1">
        <defs>
          <marker id="ag-ahv" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
            <path d="M0,0 L8,4.5 L0,9 z" fill="#2dd4bf" />
          </marker>
          <marker id="ag-ahb" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
            <path d="M0,0 L8,4.5 L0,9 z" fill="#f472b6" />
          </marker>
        </defs>

        {/* 阶段标题 */}
        <text x="105" y="44" textAnchor="middle" fill="#eef3ff" fontSize="13" fontWeight="700">① 权重 α̂₁,ⱼ</text>
        <text x="105" y="62" textAnchor="middle" fill="#6e7aab" fontSize="10.5" fontFamily="JetBrains Mono,monospace">（3 位小数）</text>
        <text x="290" y="44" textAnchor="middle" fill="#eef3ff" fontSize="13" fontWeight="700">② 连向 vⱼ（乘积）</text>
        <text x="470" y="44" textAnchor="middle" fill="#eef3ff" fontSize="13" fontWeight="700">③ 值向量 vⱼ</text>
        <text x="905" y="44" textAnchor="middle" fill="#eef3ff" fontSize="13" fontWeight="700">④ 扇入汇聚 b₁</text>

        {/* 列1：权重块 */}
        {ys.map((y, i) => (
          <g key={`w${i}`}>
            <rect x={40} y={y - 26} width={130} height={52} rx={8} fill="rgba(56,189,248,0.18)" stroke="#38bdf8" strokeWidth={1.4} />
            <text x={105} y={y - 5} textAnchor="middle" fontSize="13" fill="#7dd3fc" fontFamily="JetBrains Mono,monospace" fontWeight="700">α̂₁,{i + 1}</text>
            <text x={105} y={y + 16} textAnchor="middle" fontSize="15.5" fill="#7dd3fc" fontFamily="JetBrains Mono,monospace" fontWeight="700">{ahats3[i].toFixed(3)}</text>
          </g>
        ))}

        {/* 列2：权重 → v 连线（青），标注乘积 */}
        {ys.map((y, i) => (
          <g key={`c${i}`}>
            <path d={`M170,${y} C240,${y} 330,${y} 398,${y}`} stroke="#2dd4bf" strokeWidth={1.6} fill="none" markerEnd="url(#ag-ahv)" />
            <rect x={205} y={y - 31} width={160} height={22} rx={5} fill="#0c1430" stroke="rgba(45,212,191,0.55)" />
            <text x={285} y={y - 15} textAnchor="middle" fontSize="10.5" fill="#2dd4bf" fontFamily="JetBrains Mono,monospace" fontWeight="700">
              {ahats3[i].toFixed(3)}×{vvals[i]}=[{products[i][0]},{products[i][1]}]
            </text>
          </g>
        ))}

        {/* 列3：v 块 */}
        {ys.map((y, i) => (
          <g key={`v${i}`}>
            <rect x={400} y={y - 26} width={140} height={52} rx={8} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" strokeWidth={1.4} />
            <text x={470} y={y - 5} textAnchor="middle" fontSize="13" fill="#2dd4bf" fontFamily="JetBrains Mono,monospace" fontWeight="700">v{i + 1}</text>
            <text x={470} y={y + 16} textAnchor="middle" fontSize="15.5" fill="#2dd4bf" fontFamily="JetBrains Mono,monospace" fontWeight="700">{vvals[i]}</text>
          </g>
        ))}

        {/* 列4：v → b1 扇入连线（粉） */}
        {ys.map((y, i) => (
          <path key={`f${i}`} d={`M540,${y} C680,${y} 740,${b1y} 828,${b1y}`} stroke="#f472b6" strokeWidth={1.5} fill="none" markerEnd="url(#ag-ahb)" opacity={0.9} />
        ))}

        {/* Σ 汇聚标注 */}
        <text x={685} y={b1y - 20} textAnchor="middle" fill="#f472b6" fontSize="13" fontFamily="JetBrains Mono,monospace" fontWeight="700">Σ 加权求和 ↓</text>

        {/* b1 输出块（显示精确值，与 s4 矩阵 O 第一行一致） */}
        <rect x={830} y={b1y - 40} width={150} height={80} rx={12} fill="rgba(244,114,182,0.2)" stroke="#f472b6" strokeWidth={2.2} />
        <text x={905} y={b1y - 10} textAnchor="middle" fontSize="16" fill="#f9a8d4" fontFamily="JetBrains Mono,monospace" fontWeight="700">b₁ ★</text>
        <text x={905} y={b1y + 16} textAnchor="middle" fontSize="15" fill="#f9a8d4" fontFamily="JetBrains Mono,monospace" fontWeight="700">[0.71, 1.31]</text>

        {/* 底部导引 */}
        <text x={520} y={455} textAnchor="middle" fill="#6e7aab" fontSize="11.5" fontFamily="JetBrains Mono,monospace">
          每个 α̂₁,ⱼ 乘对应 vⱼ 得一份贡献，4 份相加 → b₁（用三位小数近似权重算）
        </text>
      </svg>

      {/* 下方加权求和展开（用三位小数近似权重） */}
      <div style={workoutWrap}>
        <div style={line}>
          <span style={lbl}>b₁ =</span>
          <span style={{ color: "#7dd3fc" }}>0.328·[0.40,1.28]</span>
          <span style={op}>+</span>
          <span style={{ color: "#7dd3fc" }}>0.103·[1.08,0.60]</span>
          <span style={op}>+</span>
          <span style={{ color: "#7dd3fc" }}>0.221·[0.65,1.06]</span>
          <span style={op}>+</span>
          <span style={{ color: "#7dd3fc" }}>0.348·[0.92,1.72]</span>
        </div>
        <div style={line}>
          <span style={lbl}>　 =</span>
          <span style={{ color: "#2dd4bf" }}>[0.131, 0.420]</span>
          <span style={op}>+</span>
          <span style={{ color: "#2dd4bf" }}>[0.111, 0.062]</span>
          <span style={op}>+</span>
          <span style={{ color: "#2dd4bf" }}>[0.144, 0.234]</span>
          <span style={op}>+</span>
          <span style={{ color: "#2dd4bf" }}>[0.320, 0.599]</span>
        </div>
        <div style={line}>
          <span style={lbl}>　 =</span>
          <span style={{ color: "#f472b6", fontWeight: 700 }}>[0.706, 1.314] ≈ [0.71, 1.31]</span>
        </div>
        <div style={note}>
          注：连线上的乘积与 b₁ 均用<b style={{ color: "#a9b4dc" }}>三位小数近似</b>权重（0.328 / 0.103 / 0.221 / 0.348）计算；
          若用未舍入权重，结果为 [0.706, 1.314]（即矩阵级 O 的第一行）。
        </div>
      </div>
    </div>
  );
}

/* ============================================================
 * FigMatrixStage · 矩阵级收束
 * 把向量级 4 步（投影→打分→softmax→汇聚）压成矩阵运算链。
 * 依赖：同文件内已有的 Formula 组件（KaTeX 渲染）。
 * 用法：在 section#s4 内 <FigMatrixStage /> 即可。
 * ============================================================ */

type Mat = number[][];

/* ---- 统一数据（我/爱/深/度，d=2；禁止改值）---- */
const FMS_WORDS = ["token₁", "token₂", "token₃", "token₄"];
const FMS_DATA: { X: Mat; WQ: Mat; WK: Mat; WV: Mat; Q: Mat; K: Mat; V: Mat; S: Mat; A: Mat; O: Mat } = {
  X:  [[0.4, 1.2], [1.5, 0.3], [0.8, 0.9], [1.1, 1.5]],
  WQ: [[1, 0.5], [-0.3, 0.8]],
  WK: [[0.8, -0.4], [0.2, 1.1]],
  WV: [[0.7, 0.2], [0.1, 1]],
  Q:  [[0.04, 1.16], [1.41, 0.99], [0.53, 1.12], [0.65, 1.75]],
  K:  [[0.56, 1.16], [1.26, -0.27], [0.82, 0.67], [1.18, 1.21]],
  V:  [[0.4, 1.28], [1.08, 0.6], [0.65, 1.06], [0.92, 1.72]],
  S:  [[0.967, -0.186, 0.573, 1.026], [1.37, 1.067, 1.287, 2.024], [1.129, 0.258, 0.838, 1.4], [1.693, 0.245, 1.206, 2.04]],
  A:  [[0.328, 0.103, 0.221, 0.348], [0.218, 0.161, 0.201, 0.42], [0.287, 0.12, 0.215, 0.377], [0.306, 0.072, 0.188, 0.433]],
  O:  [[0.706, 1.314], [0.778, 1.311], [0.732, 1.317], [0.721, 1.38]],
};

/* 配色（字面色值，行内 style 用）——与全局 Q/K/V/Att/Out 一致 */
const FMS_PAL: Record<string, { c: string; t: string }> = {
  X: { c: "#a9b4dc", t: "rgba(169,180,220,0.08)" },
  W: { c: "#6e7aab", t: "rgba(110,122,171,0.10)" },
  Q: { c: "#f5b042", t: "rgba(245,176,66,0.10)" },
  K: { c: "#a78bfa", t: "rgba(167,139,250,0.10)" },
  V: { c: "#2dd4bf", t: "rgba(45,212,191,0.10)" },
  S: { c: "#38bdf8", t: "rgba(56,189,248,0.10)" },
  A: { c: "#38bdf8", t: "rgba(56,189,248,0.10)" },
  O: { c: "#f472b6", t: "rgba(244,114,182,0.10)" },
};

const transpose = (m: Mat): Mat => m[0].map((_, j) => m.map((r) => r[j]));
const fmt = (v: number, d: number): string => (v === 0 ? 0 : v).toFixed(d);

/* ---- 单个带行列标签的矩阵网格 ---- */
function FmsMatGrid({
  data, name, shape, pal,
  rowLabels, colLabels, cornerLabel,
  heat = false, digits = 2,
}: {
  data: Mat;
  name: string;
  shape: string;
  pal: { c: string; t: string };
  rowLabels?: string[];
  colLabels?: string[];
  cornerLabel?: string;
  heat?: boolean;
  digits?: number;
}) {
  const max = heat ? Math.max(...data.flat(), 1e-9) : 1;
  const showHeader = !!colLabels;
  const showRowLab = !!colLabels || !!rowLabels;
  return (
    <div className="fms-mat">
      <div className="fms-mat-name">
        <span style={{ color: pal.c }}>{name}</span>
        <span className="fms-shape">{shape}</span>
      </div>
      <div className="fms-grid">
        <table>
          {showHeader && (
            <thead>
              <tr>
                <th className="fms-corner">{cornerLabel ?? ""}</th>
                {colLabels!.map((c, j) => (
                  <th key={j} className="fms-collab">{c}</th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                {showRowLab && (
                  <th className="fms-rowlab" style={{ color: pal.c }}>
                    {rowLabels ? rowLabels[i] : ""}
                  </th>
                )}
                {row.map((v, j) => {
                  const t = heat ? v / max : 0;
                  const bg = heat
                    ? `rgba(56,189,248,${0.1 + t * 0.75})`
                    : pal.t;
                  const color = heat ? (t > 0.4 ? "#ffffff" : "#a9b4dc") : "#eef3ff";
                  const border = heat
                    ? `rgba(56,189,248,${0.25 + t * 0.5})`
                    : "rgba(255,255,255,0.10)";
                  return (
                    /* 结构是 td > div.fms-cell：CSS 直接选 .fms-cell，别写 td.cell */
                    <td key={j} className="fms-td">
                      <div
                        className="fms-cell"
                        style={{ background: bg, color, borderColor: border }}
                      >
                        {fmt(v, digits)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ---- 运算符 / 函数算子 ---- */
function Op({ children, kind = "op" }: { children: React.ReactNode; kind?: "op" | "fn" }) {
  return <div className={kind === "fn" ? "fms-op fms-op-fn" : "fms-op"}>{children}</div>;
}

/* ---- 主组件：四阶段矩阵计算链 ---- */
function FigMatrixStage() {
  const KT = transpose(FMS_DATA.K); // Kᵀ：2×4
  return (
    <div className="fms-wrap">
      {/* 总公式 */}
      <div className="fms-banner">
        <span className="fms-banner-tag">总公式</span>
        <Formula block tex={String.raw`\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V`} />
      </div>

      {/* Stage 1：投影 X → Q / K / V */}
      <div className="fms-stage">
        <div className="fms-stage-head">
          <span className="fms-stage-num" style={{ background: FMS_PAL.Q.c }}>1</span>
          <span className="fms-stage-title">
            投影 · <span style={{ color: FMS_PAL.X.c }}>X</span> 乘三个权重 →&nbsp;
            <span style={{ color: FMS_PAL.Q.c }}>Q</span> /&nbsp;
            <span style={{ color: FMS_PAL.K.c }}>K</span> /&nbsp;
            <span style={{ color: FMS_PAL.V.c }}>V</span>
          </span>
          <span className="fms-stage-sub">dₖ = dᵥ = 2</span>
        </div>
        <div className="fms-stage1">
          <FmsMatGrid name="X" shape="[4×2]" pal={FMS_PAL.X} data={FMS_DATA.X}
            rowLabels={FMS_WORDS} colLabels={["d₁", "d₂"]} cornerLabel="token＼维" digits={2} />
          <div className="fms-branches">
            <div className="fms-chain">
              <Op>×</Op>
              <FmsMatGrid name="WQ" shape="[2×2]" pal={FMS_PAL.W} data={FMS_DATA.WQ}
                rowLabels={["d₁", "d₂"]} colLabels={["d₁", "d₂"]} cornerLabel="维＼维" digits={2} />
              <Op>=</Op>
              <FmsMatGrid name="Q" shape="[4×2]" pal={FMS_PAL.Q} data={FMS_DATA.Q}
                rowLabels={FMS_WORDS} colLabels={["d₁", "d₂"]} cornerLabel="token＼维" digits={2} />
            </div>
            <div className="fms-chain">
              <Op>×</Op>
              <FmsMatGrid name="WK" shape="[2×2]" pal={FMS_PAL.W} data={FMS_DATA.WK}
                rowLabels={["d₁", "d₂"]} colLabels={["d₁", "d₂"]} cornerLabel="维＼维" digits={2} />
              <Op>=</Op>
              <FmsMatGrid name="K" shape="[4×2]" pal={FMS_PAL.K} data={FMS_DATA.K}
                rowLabels={FMS_WORDS} colLabels={["d₁", "d₂"]} cornerLabel="token＼维" digits={2} />
            </div>
            <div className="fms-chain">
              <Op>×</Op>
              <FmsMatGrid name="WV" shape="[2×2]" pal={FMS_PAL.W} data={FMS_DATA.WV}
                rowLabels={["d₁", "d₂"]} colLabels={["d₁", "d₂"]} cornerLabel="维＼维" digits={2} />
              <Op>=</Op>
              <FmsMatGrid name="V" shape="[4×2]" pal={FMS_PAL.V} data={FMS_DATA.V}
                rowLabels={FMS_WORDS} colLabels={["d₁", "d₂"]} cornerLabel="token＼维" digits={2} />
            </div>
          </div>
        </div>
      </div>

      <div className="fms-arrow-down">↓</div>

      {/* Stage 2：打分 Q × Kᵀ ÷ √dₖ = S */}
      <div className="fms-stage">
        <div className="fms-stage-head">
          <span className="fms-stage-num" style={{ background: FMS_PAL.S.c }}>2</span>
          <span className="fms-stage-title">
            打分 · <span style={{ color: FMS_PAL.Q.c }}>Q</span> ×&nbsp;
            <span style={{ color: FMS_PAL.K.c }}>Kᵀ</span> ÷ √dₖ → 分数&nbsp;
            <span style={{ color: FMS_PAL.S.c }}>S</span>
          </span>
          <span className="fms-stage-sub">两两点积；行 = Query，列 = Key</span>
        </div>
        <div className="fms-chain">
          <FmsMatGrid name="Q" shape="[4×2]" pal={FMS_PAL.Q} data={FMS_DATA.Q}
            rowLabels={FMS_WORDS} colLabels={["d₁", "d₂"]} cornerLabel="token＼维" digits={2} />
          <Op>×</Op>
          <FmsMatGrid name="Kᵀ" shape="[2×4]" pal={FMS_PAL.K} data={KT}
            rowLabels={["d₁", "d₂"]} colLabels={FMS_WORDS} cornerLabel="维＼token" digits={2} />
          <Op kind="fn">÷ √dₖ</Op>
          <Op>=</Op>
          <FmsMatGrid name="S" shape="[4×4]" pal={FMS_PAL.S} data={FMS_DATA.S} heat
            rowLabels={FMS_WORDS} colLabels={FMS_WORDS} cornerLabel="Q＼K" digits={3} />
        </div>
      </div>

      <div className="fms-arrow-down">↓</div>

      {/* Stage 3：softmax(S) = A */}
      <div className="fms-stage">
        <div className="fms-stage-head">
          <span className="fms-stage-num" style={{ background: FMS_PAL.A.c }}>3</span>
          <span className="fms-stage-title">
            归一化 · 对 <span style={{ color: FMS_PAL.S.c }}>S</span> 按行 softmax → 权重&nbsp;
            <span style={{ color: FMS_PAL.A.c }}>A</span>
          </span>
          <span className="fms-stage-sub">未舍入时每行和 = 1（表内为三位小数近似，少数行和为 0.999）</span>
        </div>
        <div className="fms-chain">
          <FmsMatGrid name="S" shape="[4×4]" pal={FMS_PAL.S} data={FMS_DATA.S} heat
            rowLabels={FMS_WORDS} colLabels={FMS_WORDS} cornerLabel="Q＼K" digits={3} />
          <Op kind="fn">softmax 按行</Op>
          <Op>=</Op>
          <FmsMatGrid name="A" shape="[4×4]" pal={FMS_PAL.A} data={FMS_DATA.A} heat
            rowLabels={FMS_WORDS} colLabels={FMS_WORDS} cornerLabel="Q＼K" digits={3} />
        </div>
      </div>

      <div className="fms-arrow-down">↓</div>

      {/* Stage 4：汇聚 A × V = O */}
      <div className="fms-stage">
        <div className="fms-stage-head">
          <span className="fms-stage-num" style={{ background: FMS_PAL.O.c }}>4</span>
          <span className="fms-stage-title">
            汇聚 · <span style={{ color: FMS_PAL.A.c }}>A</span> ×&nbsp;
            <span style={{ color: FMS_PAL.V.c }}>V</span> → 输出&nbsp;
            <span style={{ color: FMS_PAL.O.c }}>O</span>
          </span>
          <span className="fms-stage-sub">每个 token 按权重拿走各 Value，融合成新表示（用未舍入权重计算，表内保留三位小数）</span>
        </div>
        <div className="fms-chain">
          <FmsMatGrid name="A" shape="[4×4]" pal={FMS_PAL.A} data={FMS_DATA.A} heat
            rowLabels={FMS_WORDS} colLabels={FMS_WORDS} cornerLabel="Q＼K" digits={3} />
          <Op>×</Op>
          <FmsMatGrid name="V" shape="[4×2]" pal={FMS_PAL.V} data={FMS_DATA.V}
            rowLabels={FMS_WORDS} colLabels={["d₁", "d₂"]} cornerLabel="token＼维" digits={2} />
          <Op>≈</Op>
          <FmsMatGrid name="O" shape="[4×2]" pal={FMS_PAL.O} data={FMS_DATA.O}
            rowLabels={FMS_WORDS} colLabels={["d₁", "d₂"]} cornerLabel="token＼维" digits={3} />
        </div>
      </div>

      {/* 色阶图例 + 读法 */}
      <div className="fms-legend-row">
        <div className="fms-legend">
          <span className="fms-legend-label">权重色阶</span>
          <span className="fms-legend-bar" />
          <span className="fms-legend-text">低 → 高（越深越大）</span>
        </div>
        <div className="fms-read">
          四步压成矩阵后，整条链只用了<b>矩阵乘法 + softmax</b>（投影×3、QKᵀ、AV 共 5 次；工程常融合成 3 次 GEMM）——
          和向量级结果完全一致，但可被 GPU 整块并行算出。
        </div>
      </div>
    </div>
  );
}


/* ============================================================
 * SVG 图：经典 Transformer 论文 Figure 1
 * ============================================================ */
function TfArrow({ d, color = "#6e7aab", dash }: { d: string; color?: string; dash?: string }) {
  return <path d={d} stroke={color} strokeWidth="1.4" fill="none" markerEnd="url(#ah-t)" strokeDasharray={dash} />;
}
function TfBox({ x, y, w, h, fill, stroke, label, sub, lc, sc }: { x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; lc?: string; sc?: string }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx="9" fill={fill} stroke={stroke} />
      <text x={x + w / 2} y={y + h / 2 + (sub ? -4 : 5)} textAnchor="middle" fill={lc || "#eef3ff"} fontSize="13" fontWeight="700">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fill={sc || "#6e7aab"} fontSize="10">{sub}</text>}
    </g>
  );
}

function FigTransformer() {
  const Arrow = TfArrow;
  const Box = TfBox;
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
        <Box x={150} y={405} w={150} h={34} fill="rgba(244,114,182,0.14)" stroke="#f472b6" label="编码器输出 Memory" lc="#f472b6" />
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
        <Box x={575} y={268} w={240} h={50} fill="rgba(56,189,248,0.14)" stroke="#38bdf8" label="Cross Attention（编码-解码交互）" sub="Q 来自解码器，K/V 由编码器 Memory 投影" lc="#38bdf8" sc="#6e7aab" />
        <Box x={620} y={334} w={150} h={34} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={575} y={386} w={240} h={44} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" label="Feed-Forward Network" lc="#2dd4bf" />
        <Box x={620} y={442} w={150} h={32} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Arrow d="M695,200 V212" /><Arrow d="M695,250 V264" /><Arrow d="M695,318 V330" /><Arrow d="M695,368 V382" /><Arrow d="M695,430 V438" />
        <Arrow d="M300,422 C440,422 460,293 573,293" color="#a78bfa" dash="4 3" />
        <text x="430" y="360" fill="#a78bfa" fontSize="10">编码器 Memory → 投影成 K/V</text>
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
 * SVG 图：普通 Attention vs FlashAttention 数据搬运对照
 * ============================================================ */
function FigFlashCompare() {
  // 每步：label 文字，bad=true 表示这是"写回/读回 HBM"的瓶颈步
  const normal: { t: string; bad?: boolean }[] = [
    { t: "Q × Kᵀ" },
    { t: "完整 S 写入 HBM", bad: true },
    { t: "读回 S，softmax" },
    { t: "完整 P 写入 HBM", bad: true },
    { t: "读回 P，× V" },
    { t: "O" },
  ];
  const flash: { t: string; good?: boolean }[] = [
    { t: "切分 Q / K / V" },
    { t: "小块加载进 SRAM", good: true },
    { t: "片上算局部分数" },
    { t: "在线更新 m / l / o", good: true },
    { t: "处理下一个 K/V 块" },
    { t: "只写出最终 O" },
  ];
  const Step = ({ t, tone }: { t: string; tone?: "bad" | "good" }) => (
    <div className={`fc-step ${tone ?? ""}`}>{t}</div>
  );
  return (
    <div className="fig">
      <div className="fc-cols">
        <div className="fc-col">
          <div className="fc-col-h bad">普通 Attention · 全矩阵在显存来回</div>
          {normal.map((s, i) => (
            <div key={i} className="fc-line">
              <Step t={s.t} tone={s.bad ? "bad" : undefined} />
              {i < normal.length - 1 && <span className="fc-dn">↓</span>}
            </div>
          ))}
        </div>
        <div className="fc-col">
          <div className="fc-col-h good">FlashAttention · 分块进片上，只写最终 O</div>
          {flash.map((s, i) => (
            <div key={i} className="fc-line">
              <Step t={s.t} tone={s.good ? "good" : undefined} />
              {i < flash.length - 1 && <span className="fc-dn">↓</span>}
            </div>
          ))}
        </div>
      </div>
      <div className="fig-cap">图 · 左侧每物化一次 N×N 矩阵都要一次 HBM 写+读；右侧在 SRAM 内完成累加，O(N²) 中间矩阵从不落地显存</div>
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
    let raf = 0;
    const upd = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      setScroll(h > 0 ? (window.scrollY / h) * 100 : 0);
      raf = 0;
    };
    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(upd);
    };
    upd();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  useEffect(() => {
    const ids = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"];
    let raf = 0;
    const onScroll = () => {
      const y = window.scrollY + 130;
      let cur = ids[0];
      ids.forEach((id) => {
        const el = document.getElementById(id);
        if (el && el.offsetTop <= y) cur = id;
      });
      setActiveNav(cur);
      raf = 0;
    };
    const throttled = () => {
      if (raf) return;
      raf = requestAnimationFrame(onScroll);
    };
    window.addEventListener("scroll", throttled, { passive: true });
    onScroll();
    return () => {
      window.removeEventListener("scroll", throttled);
      if (raf) cancelAnimationFrame(raf);
    };
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
    ["s2", "向量级 Attention"],
    ["s3", "矩阵级 Attention"],
    ["s4", "缩放与 Mask"],
    ["s5", "多头注意力"],
    ["s6", "FlashAttention"],
    ["s7", "代码与算子测试"],
    ["s8", "Transformer 定位"],
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
            <span className="kicker">{"// Operator Deep-Dive"}</span>
            <h1>Attention 算子<br /><em>到底在算什么？</em></h1>
            <p className="lead">从矩阵乘法一路讲到经典 Transformer 全景图：投影、点积、缩放、softmax、加权汇聚，每一步都用真实数值矩阵演示。</p>
            <div className="chips">
              <span>从矩阵乘法起步</span>
              <span><b>Q · K · V</b> 全程配色一致</span>
              <span>含 PyTorch 经典代码</span>
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
                <p className="t3">Attention 的核心就是<b style={{ color: "#eef3ff" }}>两次矩阵乘法（QKᵀ 算分、AV 取值）+ 一个 softmax</b>；算上三个投影 W 是五次，工程常融合成三次 GEMM。这篇分享的目标，是让你在脑子里把这句话可视化出来。</p>
              </div>
            </div>
          </section>

          {/* ===== 矩阵乘法 ===== */}
          <section className="section" id="s1">
            <SecHead idx="01" title="热身：矩阵乘法到底怎么乘" />
            <p className="sec-lead">Attention 的主要线性运算由<b style={{ color: "#eef3ff" }}>矩阵乘法</b>完成，中间穿插缩放、mask 和 softmax。先用具体数字把矩阵乘法规则搞明白——<b style={{ color: "#f472b6" }}>点一下右边结果矩阵的任意格子</b>，左边高亮参与计算的行与列。</p>
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

          {/* ===== 向量级 ===== */}
          <section className="section" id="s2">
            <SecHead idx="02" title="Self-Attention · 向量级（一步一步算）" />
            <div className="legend-row">
              <span><i className="lq" />Query 查询：我想找什么</span>
              <span><i className="lk" />Key 键：我有什么可被匹配</span>
              <span><i className="lv" />Value 值：匹配上后拿走的内容</span>
            </div>
            <p className="sec-lead">在 Attention 之前，处理序列的主力是 RNN——但它<b style={{ color: "#eef3ff" }}>只能逐个往后算，算不出第 4 个就得等前 3 个</b>，无法并行。Self-Attention 换了个思路：让所有位置<b>同时互相看见</b>，每个输出都直接看完整序列、彼此不依赖。下面是整篇的核心——固定主角 <b style={{ color: "#f5b042" }}>q₁</b>，用<b style={{ color: "#eef3ff" }}>四张连续的图</b>追踪它如何一步步走到 <b style={{ color: "#f472b6" }}>b₁</b>：① 投影得 q/k/v → ② q₁ 向所有 k 扇出打分 → ③ 整行 softmax → ④ 权重乘 v 汇聚成 b₁。每个输入词 <Formula tex="x" /> 会变成三份不同身份：<b style={{ color: "#f5b042" }}>q（去查询）</b>、<b style={{ color: "#a78bfa" }}>k（被匹配）</b>、<b style={{ color: "#2dd4bf" }}>v（被汇聚的内容）</b>。</p>

            <FigStageQKV />
            <FigStageScore />
            <FigStageSoftmax />
            <FigStageAggregate />

            <div className="note">换 q₂、q₃、q₄ 走同样路径，就得 b₂、b₃、b₄——所有位置可并行计算。下一节把这套向量运算收束成矩阵形式。</div>
          </section>

          {/* ===== 矩阵级 ===== */}
          <section className="section" id="s3">
            <SecHead idx="03" title="Self-Attention · 矩阵级" />
            <p className="sec-lead">把所有词的 q/k/v 堆成矩阵 <Formula tex="Q, K, V" />，整件事就坍缩成<b style={{ color: "#eef3ff" }}>几次矩阵乘法 + 一次 softmax</b>（工程上 Q/K/V 投影常融合成一次 GEMM，所以常说"三次"）——这正是 GPU 最擅长、能大规模并行的形态。</p>
            <div className="note"><b>记号约定（先说清楚，避免和代码对不上）</b>：本文用 PyTorch 行向量约定 <Formula tex={String.raw`Q=XW^Q`} />，所以是 <Formula tex={String.raw`QK^{\mathsf T}`} />；有的教材用列向量 <Formula tex={String.raw`Q=W^Q I`} />，对应 <Formula tex={String.raw`K^{\mathsf T}Q`} />。两者数学等价，只差一个转置——这也是代码里写 <code>key.transpose(-2, -1)</code> 的原因。</div>
            <FigMatrixStage />
            <div className="note">现在这句公式对你不再是一串符号：<Formula tex={String.raw`QK^{\mathsf T}`} /> 是「两两算相关度」，softmax 是「分数变权重」，乘 <Formula tex={String.raw`V`} /> 是「按权重取内容」。上面这组数值，正是把向量级 4 步压成矩阵后一次性算出的结果。</div>
          </section>

          {/* ===== Scale 与 Mask ===== */}
          <section className="section" id="s4">
            <SecHead idx="04" title={<>缩放 <Formula tex={String.raw`\sqrt{d_k}`} /> 与 Mask</>} />
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
                  <div className="mname">scaled logits <span style={{ color: "#38bdf8" }}>α = q·k/sqrt(d_k)</span></div>
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

            {/* —— Mask：因果掩码与 padding —— */}
            <h3>Mask：让每个位置"只看到该看的"</h3>
            <p className="sec-lead">生成时每个位置必须"看到过去、看不到未来"。实现上用一个<b style={{ color: "#f5b042" }}>下三角因果掩码（Causal Mask）</b>：第 i 个位置只允许看第 0..i 个 Key；padding 位则用 padding mask 屏蔽。</p>

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
            <div className="fig-cap">因果掩码矩阵：绿色 <b style={{ color: "#34d399" }}>0</b> = 允许看，红色 <b style={{ color: "#f472b6" }}>−∞</b> = 屏蔽。第 i 行只允许列 0..i</div>

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

            <h3 style={{ marginTop: 24 }}>同一个 Mask，两种用法</h3>
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

          {/* ===== 多头 ===== */}
          <section className="section" id="s5">
            <SecHead idx="05" title="多头注意力（Multi-Head）" />
            <p className="sec-lead">只做一次 attention 只能学到「一种关注方式」。拆成<b style={{ color: "#eef3ff" }}>多个头</b>，各自独立算 attention，等于从多个角度（语法、长程、局部……）同时看序列，最后拼回来。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`\operatorname{head}_i=\operatorname{Attention}(QW_i^Q,\,KW_i^K,\,VW_i^V)`} />
              <Formula block tex={String.raw`\operatorname{MHA}=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\,W^O`} />
            </div>
            <div className="note">实践要点：每个头把维度 <Formula tex="d" /> 切成 <Formula tex="d/h" />，主 FLOPs 量级与单头接近、表达能力更强；但投影层、显存占用与调度开销并不为零，并非真的"免费"。</div>

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

          {/* ===== FlashAttention ===== */}
          <section className="section" id="s6">
            <SecHead idx="06" title="FlashAttention：不改变数学，改变算的方式" />
            <p className="sec-lead">前面每一步都算清楚了。现在换个角度：这套 <Formula tex={String.raw`\operatorname{softmax}(QK^{\mathsf T}/\sqrt{d_k})V`} /> 在 GPU 上到底怎么跑？结论一句话——<b style={{ color: "#eef3ff" }}>FlashAttention 没有改变 Attention 的数学定义，改变的是计算顺序和数据搬运方式</b>。</p>

            <div className="eq-box">
              <Formula block tex={String.raw`O=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V`} />
            </div>

            <h3>瓶颈在显存，不在算力</h3>
            <p className="sec-lead">GPU 的<b>计算</b>极快，慢在<b>显存搬运</b>。朴素 Attention 会把整张 <Formula tex="N\times N" /> 的分数矩阵 <Formula tex="S" /> 和权重矩阵 <Formula tex="P" /> 写进 HBM（显存）再读回，序列一长，显存读写就成墙——而 HBM 带宽远低于片上 SRAM。FlashAttention 的做法：<b style={{ color: "#2dd4bf" }}>把 Q/K/V 切成小块，分批搬进 SRAM，在片上算、在线更新，只写回最终输出</b>。</p>

            <FigFlashCompare />

            <h3>在线 softmax：不存全矩阵也能归一化</h3>
            <p className="sec-lead">难点在 softmax 的分母 <Formula tex="l=\sum_j e^{s_j}" /> 需要"看到整行"。分块后，新块的分数可能含更大的值，旧的累加值必须按新最大值<b style={{ color: "#eef3ff" }}>重新缩放</b>——这就是"在线 softmax"的核心。每处理一个 K/V 块，维护三个量：行最大值 <Formula tex="m" />、归一化系数 <Formula tex="l" />、输出累加 <Formula tex="o" />。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`m_\text{new}=\max\!\left(m,\,\operatorname{rowmax}(S_t)\right)`} />
              <Formula block tex={String.raw`l_\text{new}=e^{m-m_\text{new}}\,l+\operatorname{rowsum}\!\left(e^{S_t-m_\text{new}}\right)`} />
              <Formula block tex={String.raw`o_\text{new}=e^{m-m_\text{new}}\,o+e^{S_t-m_\text{new}}\,V_t`} />
            </div>
            <div className="note">全部块处理完后 <Formula tex={String.raw`O=o/l`} />。当新块出现更大分数时，旧累加按 <Formula tex="e^{m-m_\text{new}}" /> 缩小——所以<b>无需保存过去全部分数</b>，整张 <Formula tex="N\times N" /> 矩阵从不在显存物化。</div>

            <h3>三个要点</h3>
            <div className="grid3">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>精确，不是近似</h3>
                <p className="t3">FlashAttention 与朴素 Attention 数学上等价，输出一致——它不是稀疏或低秩近似。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>省的是显存与读写</h3>
                <p className="t3">中间 <Formula tex="S/P" /> 不落地 HBM，显存占用从 <Formula tex={String.raw`O(N^2)`} /> 降到 <Formula tex="O(N)" />，HBM 访问量大幅减少。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>计算量不变</h3>
                <p className="t3">仍是 <Formula tex={String.raw`O(N^2d)`} /> 乘法；省下的全是访存与中间存储。序列越长收益越大。</p>
              </div>
            </div>
            <div className="note">参考：FlashAttention（Dao 等，2022）、FlashAttention-2（Dao，2023）在分块与并行划分上进一步优化。下一节看它如何对应到真实的算子调用与测试。</div>
          </section>

          {/* ===== Transformer 收尾定位 ===== */}
          <section className="section" id="s7">
            <SecHead idx="07" title="代码与算子测试：从原理到真实算子" />
            <p className="sec-lead">原理看懂了，落到代码就两层：一层<b style={{ color: "#eef3ff" }}>透明的参考实现</b>用于对照，一层<b style={{ color: "#2dd4bf" }}>真实算子</b>用于生产。算子测试就围着这两层展开。</p>

            <div className="code-title">① 透明参考实现 — 逐行对应公式</div>
            <pre><code>{`def attention_ref(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)          # QK^T
    scores = scores / math.sqrt(q.size(-1))   # / sqrt(d_k)
    if mask is not None:
        scores = scores + mask                # + M (causal/padding)
    weights = torch.softmax(scores, dim=-1)   # softmax 按行
    return weights @ v                         # · V`}</code></pre>
            <div className="note">这几行就是前面所有图的代码化身。它<b>正确但不够快</b>：会把整张分数矩阵物化到显存，长序列下又慢又费显存。生产里用下面这行替代。</div>

            <div className="code-title">② 真实算子 — PyTorch SDPA（自动选 FlashAttention 后端）</div>
            <pre><code>{`output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask,     # causal / padding / 自定义
    dropout_p=0.0,
    is_causal=False,
)
# 需要对照不同后端时，可强制选择内核：
# with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION): ...`}</code></pre>
            <div className="note"><b>F.scaled_dot_product_attention</b>（SDPA）根据输入形状、数据类型和硬件，自动在 math / mem-efficient / flash 三种后端里选最快的——FlashAttention 就是其中之一。数学上它与参考实现等价，但避免了中间矩阵的显存物化。</div>

            <h3>算子测试，只看这五件事</h3>
            <div className="grid3">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>① 前向正确性</h3>
                <p className="t3">Flash / SDPA 输出与透明参考实现逐元素比对。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>② 反向正确性</h3>
                <p className="t3">比较 <Formula tex="dQ,dK,dV" />，不能只验前向——梯度路径才是算子真正容易出错的地方。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>③ Mask 语义</h3>
                <p className="t3">覆盖 causal、padding、无 mask；改动未来 token 后，过去位置的输出必须保持不变。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>④ 精度与容差</h3>
                <p className="t3">FP32 / FP16 / BF16。归约顺序不同导致不能逐位相等，用合理的绝对+相对误差判定。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>⑤ 性能与显存</h3>
                <p className="t3">预热 + GPU 同步 + 多轮统计延迟、吞吐、峰值显存，并记录实际命中的后端。</p>
              </div>
            </div>

            <div className="note ok"><b>一句话总结</b>：Attention 把「每个位置该关注谁」变成 <Formula tex={String.raw`QK^{\mathsf T}\!/\sqrt{d_k}`} /> 算分、softmax 变权重、再乘 <Formula tex="V" /> 取内容；多头扩展视角，FlashAttention 换算的方式不换数学，配上位置编码装进 Encoder/Decoder，就是撑起所有现代大模型的 Transformer。</div>
          </section>

          {/* ===== 代码 ===== */}
          <section className="section" id="s8">
            <SecHead idx="08" title="Transformer 全景：Attention 被装在哪里" />
            <p className="sec-lead">Attention 本身只是一个算子。把它装进完整模型，就是这篇被引用几万次的论文——<b style={{ color: "#38bdf8" }}>左 Encoder</b>、<b style={{ color: "#f472b6" }}>右 Decoder</b>，各堆叠 N 层。</p>
            <FigTransformer />
            <div className="grid2">
              <div className="note"><b>Encoder</b>：对源序列做 Self-Attention + FFN，逐层提炼表示，输出的 Memory 作为 Cross-Attention 的 K/V 来源。</div>
              <div className="note"><b>Decoder</b>：先用 <b>Masked</b> Self-Attention（屏蔽未来位防作弊），再通过 <b>Cross-Attention</b> 把编码器 Memory 投影成 K/V 来读取，最后预测下一个词。</div>
            </div>
            <div className="note warn">这张图里 <b>Attention 出现了三次</b>（Encoder 自注意、Decoder 掩码自注意、Decoder 交叉注意）——是同一个算子的三种调用。这一节只为定位：讲清楚 Attention 在整个模型里扮演什么角色。</div>

            <h3>位置编码：把"顺序"补回去</h3>
            <p className="sec-lead">Self-Attention 对「顺序」无感——所以在输入 embedding 上<b style={{ color: "#eef3ff" }}>直接加一个位置向量</b>，把顺序信息喂回去。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\quad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)`} />
            </div>
            <div className="note">用不同频率的正余弦，让每个位置拿到<b>唯一</b>编码；且对任意固定间距 <Formula tex="k" />，<Formula tex={String.raw`PE_{pos+k}`} /> 是 <Formula tex={String.raw`PE_{pos}`} /> 的线性函数——模型因此能泛化到比训练更长的序列。</div>
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
