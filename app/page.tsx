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

/* ---------- Attention 演示数据（全页统一使用 4 个位置标签，d=2） ---------- */
const tokenLabels = ["token₁", "token₂", "token₃", "token₄"];
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
    note: "权重大量出现在非对角线位置 → 可能形成长距离关注。",
    matrix: [
      [0.58, 0.12, 0.22, 0.08],
      [0.18, 0.16, 0.12, 0.54],
      [0.66, 0.08, 0.20, 0.06],
      [0.14, 0.46, 0.09, 0.31],
    ],
  },
  {
    name: "Head 2 · 局部",
    note: "权重集中在主对角线附近 → 可能形成局部关注。",
    matrix: [
      [0.62, 0.27, 0.07, 0.04],
      [0.24, 0.48, 0.22, 0.06],
      [0.07, 0.24, 0.48, 0.21],
      [0.03, 0.08, 0.30, 0.59],
    ],
  },
  {
    name: "Head h · 全局",
    note: "权重分布较平均 → 可能形成全局关注。",
    matrix: [
      [0.28, 0.24, 0.25, 0.23],
      [0.23, 0.29, 0.22, 0.26],
      [0.27, 0.20, 0.30, 0.23],
      [0.22, 0.27, 0.21, 0.30],
    ],
  },
];


/* ============================================================
 * 向量级四阶段图 + 矩阵级收束（统一 4-token 数据，d=2）
 * ============================================================ */

/* ============================================================
 * 向量阶段图1：X → Q/K/V 投影
 * 统一数据：token₁/token₂/token₃/token₄，d=2
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
  label?: React.ReactNode;
  sub?: React.ReactNode;
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
  const valueColumn = rowLabels ? 2 : 1;
  return (
    <div
      style={{
        display: "inline-grid",
        gridTemplateColumns: rowLabels ? "auto auto" : "auto",
        gridTemplateRows: "auto auto auto",
        alignItems: "center",
        columnGap: 8,
        rowGap: 6,
      }}
    >
      {label ? (
        <div
          style={{
            gridColumn: valueColumn,
            gridRow: 1,
            justifySelf: "center",
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
      {rowLabels ? (
        <table style={{ gridColumn: 1, gridRow: 2, borderCollapse: "collapse" }}>
          <tbody>
            {rowLabels.map((w, i) => (
              <tr key={i}>
                <td
                  style={{
                    height: 42,
                    minWidth: 52,
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
      <div style={{ gridColumn: valueColumn, gridRow: 2, display: "inline-flex", alignItems: "stretch" }}>
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
        <div
          style={{
            gridColumn: valueColumn,
            gridRow: 3,
            justifySelf: "center",
            color: "var(--t3)",
            fontSize: 11,
            fontFamily: "var(--mono)",
          }}
        >
          {sub}
        </div>
      ) : null}
    </div>
  );
}

function AttentionSetupGuide() {
  return (
    <div className="setup-guide">
      <div className="setup-guide-head">
        <span>计算前置</span>
        <b>先把 Token、<Formula tex="x_i" />、<Formula tex="X" />、<Formula tex="q_1" /> 放回同一条链路</b>
        <p>本页只用 <code>Token 1～4</code> 表示序列中的四个位置，不把它们映射成具体词语；后面的每一行矩阵都与这四个位置一一对应。</p>
      </div>

      <div className="setup-flow">
        <article className="setup-node">
          <span className="setup-step">1</span>
          <strong>分词后的四个位置</strong>
          <div className="setup-tokens">
            {tokenLabels.map((token, i) => (
              <span key={token}><b>Token {i + 1}</b><small>位置 {i + 1}</small></span>
            ))}
          </div>
          <p>真实文本先经 tokenizer 变成离散的 token ID。这里省略具体文本，只保留位置标签。</p>
        </article>

        <article className="setup-node">
          <span className="setup-step">2</span>
          <strong>每个位置得到输入向量 <Formula tex="x_i" /></strong>
          <Formula block tex={String.raw`e_i=E_{\mathrm{tok}}[\operatorname{tokenId}_i,:]`} />
          <Formula block tex={String.raw`x_i=e_i+p_i`} />
          <p>首层通常由 token embedding 与位置信息组成；后续层的 <Formula tex="x_i" /> 来自上一层输出。现代模型也可能用 RoPE 等方式注入位置。</p>
        </article>

        <article className="setup-node">
          <span className="setup-step">3</span>
          <strong>按位置堆叠成输入矩阵 <Formula tex="X" /></strong>
          <Formula block tex={String.raw`X=\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}=\begin{bmatrix}0.4&1.2\\1.5&0.3\\0.8&0.9\\1.1&1.5\end{bmatrix}`} />
          <p><Formula tex={String.raw`X\in\mathbb{R}^{4\times2}`} />：4 表示四个位置，2 表示每个位置暂用两个数描述。</p>
        </article>

        <article className="setup-node">
          <span className="setup-step">4</span>
          <strong><Formula tex="X" /> 经三组参数投影为 <Formula tex={String.raw`Q,\ K,\ V`} /></strong>
          <Formula block tex={String.raw`\begin{aligned}Q&=XW^Q\\K&=XW^K\\V&=XW^V\end{aligned}`} />
          <Formula block tex={String.raw`q_1=x_1W^Q`} />
          <p><Formula tex="q_1" /> 不是额外生成的变量：它就是 <Formula tex="Q" /> 的第一行，对应 Token 1；<Formula tex={String.raw`k_1,\ v_1`} /> 同理。</p>
        </article>
      </div>

      <div className="setup-dimension-note">
        <b>为什么只有二维？</b>
        <span>为了能在页面上完整展开每一次乘法，本例设 <Formula tex={String.raw`d_{\mathrm{model}}=d_k=d_v=2`} />。真实模型通常是数百到数千维，算法完全相同，只是矩阵更大。</span>
      </div>

      <div className="setup-table-wrap">
        <table className="setup-table">
          <thead>
            <tr><th>符号</th><th>本页中表示什么</th><th>从哪里来</th></tr>
          </thead>
          <tbody>
            <tr>
              <th>Token i</th>
              <td>序列中的第 i 个位置标签；它本身不是向量</td>
              <td>tokenizer 输出 token ID 序列后，由它在序列中的索引确定</td>
            </tr>
            <tr>
              <th><Formula tex="x_i" /></th>
              <td>第 i 个位置送入当前 Attention 层的行向量</td>
              <td>首层来自 embedding 与位置信息；后续层来自上一层</td>
            </tr>
            <tr>
              <th><Formula tex="X" /></th>
              <td>把 <Formula tex={String.raw`x_1,\ldots,x_4`} /> 按行堆叠后的输入矩阵</td>
              <td>本例 <Formula tex={String.raw`X\in\mathbb{R}^{4\times2}`} /></td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`W^Q,W^K,W^V`} /></th>
              <td>三组独立的线性投影参数：<Formula tex={String.raw`W^Q,W^K\in\mathbb{R}^{d_{\mathrm{model}}\times d_k}`} />，<Formula tex={String.raw`W^V\in\mathbb{R}^{d_{\mathrm{model}}\times d_v}`} />；本例均为 <Formula tex={String.raw`2\times2`} /></td>
              <td>训练开始时初始化，训练中由反向传播学习；推理时保持固定</td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`q_i,k_i,v_i`} /></th>
              <td>位置 <Formula tex="i" /> 的 Query、Key、Value 行向量；<Formula tex={String.raw`q_i,k_i\in\mathbb{R}^{d_k}`} />，<Formula tex={String.raw`v_i\in\mathbb{R}^{d_v}`} /></td>
              <td><Formula tex={String.raw`q_i=x_iW^Q,\ k_i=x_iW^K,\ v_i=x_iW^V`} /></td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`S,A`} /></th>
              <td>缩放后的相关分数矩阵、softmax 后的权重矩阵；本例均属于 <Formula tex={String.raw`\mathbb{R}^{4\times4}`} /></td>
              <td><Formula tex={String.raw`S=QK^{\mathsf T}/\sqrt{d_k},\quad A=\operatorname{softmax}(S)`} /></td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`b_i,O`} /></th>
              <td><Formula tex={String.raw`b_i\in\mathbb{R}^{d_v}`} /> 是位置 <Formula tex="i" /> 的新表示；四行堆起来就是 <Formula tex={String.raw`O\in\mathbb{R}^{4\times d_v}`} /></td>
              <td><Formula tex={String.raw`b_i=\sum_j A_{i,j}v_j,\quad O=AV`} /></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="setup-types">
        <div><b>可训练参数</b><span>当前单头链路包含 Token Embedding 表 <Formula tex="E_{\mathrm{tok}}" /> 与投影矩阵 <Formula tex={String.raw`W^Q,\ W^K,\ W^V`} />；若采用可学习位置表，还包括 <Formula tex="P_{\mathrm{pos}}" />。它们训练初始通常随机，随后被学习；推理时不再重新随机。</span></div>
        <div><b>前向中间量</b><span><Formula tex={String.raw`X,\ Q,\ K,\ V,\ S,\ A,\ O`} />：随输入变化，每次前向重新计算，不是模型单独保存的参数。</span></div>
        <div><b>本页固定数字</b><span>为便于手算而选定的教学样例；它们不是从某个真实模型中导出的数值，也不会在刷新页面时重新随机。</span></div>
      </div>
    </div>
  );
}

function ScoreMatrixReadingGuide() {
  return (
    <div className="score-reading-guide">
      <div className="score-reading-head">
        <b><Formula tex={String.raw`4\times4`} /> 矩阵只需认清三个坐标</b>
        <span>下面直接把坐标规则标在真实数值矩阵上，不再重复画 16 个语义格子。</span>
      </div>
      <div className="score-reading-rules">
        <span><b>行 <Formula tex="i" /></b>：Token <Formula tex="i" /> 发出 Query，决定“谁在看”</span>
        <span><b>列 <Formula tex="j" /></b>：Token <Formula tex="j" /> 提供 Key，决定“正在看谁”</span>
        <span><b>格 <Formula tex="(i,j)" /></b>：<Formula tex={String.raw`S_{i,j}=q_i k_j^{\mathsf T}/\sqrt{d_k}`} />，即 Token <Formula tex="i" /> 看 Token <Formula tex="j" /> 的分数</span>
      </div>
    </div>
  );
}

function FigStageQKV() {
  const WORDS = tokenLabels;
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
    wlabel: React.ReactNode;
    W: number[][];
    R: number[][];
    tex: string;
    note: React.ReactNode;
  };
  const rows: RowDef[] = [
    {
      key: "Q",
      color: CQ,
      wlabel: <Formula tex="W^Q" />,
      W: WQ,
      R: Q,
      tex: String.raw`Q = XW^Q`,
      note: <><Formula tex={String.raw`q_i=x_iW^Q`} />：位置 <Formula tex="i" /> 用于发起匹配的查询特征</>,
    },
    {
      key: "K",
      color: CK,
      wlabel: <Formula tex="W^K" />,
      W: WK,
      R: K,
      tex: String.raw`K = XW^K`,
      note: <><Formula tex={String.raw`k_i=x_iW^K`} />：位置 <Formula tex="i" /> 用于被 Query 匹配的索引特征</>,
    },
    {
      key: "V",
      color: CV,
      wlabel: <Formula tex="W^V" />,
      W: WV,
      R: V,
      tex: String.raw`V = XW^V`,
      note: <><Formula tex={String.raw`v_i=x_iW^V`} />：位置 <Formula tex="i" /> 最终参与加权汇聚的内容特征</>,
    },
  ];

  return (
    <div
      role="img"
      aria-label="向量阶段投影图：X 分别乘 Query、Key、Value 投影矩阵得到 Q、K、V"
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
          第 1 步 · 投影：同一份 <span style={{ color: "var(--t2)" }}><Formula tex="X" /></span> 乘三个权重矩阵，准备全部{" "}
          <span style={{ color: CQ }}><Formula tex="Q" /></span> / <span style={{ color: CK }}><Formula tex="K" /></span> /{" "}
          <span style={{ color: CV }}><Formula tex="V" /></span>
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

      {/* 共享 X + 一进三出的投影分支 */}
      <div className="qkv-projection-flow">
        {/* 左侧：共享输入 X（只出现一次） */}
        <div className="qkv-source">
          <QKVMat data={X} accent={NEUTRAL} label={<Formula tex="X" />} sub={<><Formula tex={String.raw`[4\times2]`} /> 共享输入</>} rowLabels={WORDS} heroRow={0} heroColor={CQ} />
          <div>同一份 <Formula tex="X" /> 作为三路投影的共同输入</div>
        </div>

        {/* 中间：一条输入主干分成 Q/K/V 三路 */}
        <div className="qkv-fanout" aria-hidden="true"><span /><span /><span /></div>

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
              <span style={{ fontSize: 20, color: "var(--t3)", fontWeight: 600, fontFamily: "var(--mono)" }}>×</span>
              <QKVMat data={r.W} accent={r.color} label={r.wlabel} sub={<Formula tex={String.raw`[2\times2]`} />} />
              <span style={{ fontSize: 20, color: "var(--t3)", fontWeight: 600, fontFamily: "var(--mono)" }}>=</span>
              <QKVMat data={r.R} accent={r.color} label={<Formula tex={r.key} />} sub={<Formula tex={String.raw`[4\times2]`} />} heroRow={r.key === "Q" ? 0 : -1} heroColor={r.color} />
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
        <span>例：<Formula tex={String.raw`x_1=[0.4,\ 1.2]`} /></span>
        <span style={{ color: CQ }}>→ <Formula tex={String.raw`q_1=[0.4\times1.0+1.2\times(-0.3),\ 0.4\times0.5+1.2\times0.8]=[0.04,\ 1.16]`} /> ★</span>
        <span style={{ color: CK }}>→ <Formula tex={String.raw`k_1=[0.4\times0.8+1.2\times0.2,\ 0.4\times(-0.4)+1.2\times1.1]=[0.56,\ 1.16]`} /></span>
        <span style={{ color: CV }}>→ <Formula tex={String.raw`v_1=[0.4\times0.7+1.2\times0.1,\ 0.4\times0.2+1.2\times1.0]=[0.40,\ 1.28]`} /></span>
      </div>
      <div style={{ textAlign: "center", color: "var(--t3)", fontSize: 12, marginTop: 10 }}>
        图 · 每一行都对应同序号 Token。后面只追踪第 1 行的 <b style={{ color: CQ }}><Formula tex={String.raw`q_1=[0.04,\ 1.16]`} /></b>，
        但它仍要与全部 <Formula tex="k_j" /> 比较，并用全部 <Formula tex="v_j" /> 形成 <Formula tex="b_1" />。
      </div>
    </div>
  );
}

function FirstRowAttentionMatrix({
  symbol,
  values,
  ariaLabel,
}: {
  symbol: "S" | "A";
  values: string[];
  ariaLabel: string;
}) {
  return (
    <div className="score-mini-matrix" aria-label={ariaLabel}>
      <span className="corner"><Formula tex={symbol} /></span>
      {[1, 2, 3, 4].map((j) => <span className={`col ${j === 1 ? "active" : ""}`} key={`col-${j}`}><Formula tex={`k_${j}`} /></span>)}
      {[1, 2, 3, 4].map((i) => (
        <div className="score-mini-row" key={`row-${i}`}>
          <span className={`row ${i === 1 ? "active" : ""}`}><Formula tex={`q_${i}`} /></span>
          {[1, 2, 3, 4].map((j) => (
            <span className={`cell ${i === 1 ? "scored" : ""} ${i === 1 && j === 1 ? "active" : ""}`} key={`cell-${i}-${j}`}>
              <Formula tex={`${symbol}_{${i},${j}}`} />
              {i === 1 ? <strong>{values[j - 1]}</strong> : null}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

/* ============================================================
 * 向量阶段图2：用 q₁ 生成分数矩阵 S 的第一行
 * 数据全程使用统一 4-token（token₁..token₄，d=2）。
 * ============================================================ */
function FigStageScore() {
  const comparisons = [
    { key: "k_1", scoreLabel: "S_{1,1}", token: "Token 1", vector: "[0.56,1.16]", score: "0.967", tex: String.raw`\dfrac{0.04\times0.56+1.16\times1.16}{\sqrt{2}}` },
    { key: "k_2", scoreLabel: "S_{1,2}", token: "Token 2", vector: "[1.26,-0.27]", score: "−0.186", tex: String.raw`\dfrac{0.04\times1.26+1.16\times(-0.27)}{\sqrt{2}}` },
    { key: "k_3", scoreLabel: "S_{1,3}", token: "Token 3", vector: "[0.82,0.67]", score: "0.573", tex: String.raw`\dfrac{0.04\times0.82+1.16\times0.67}{\sqrt{2}}` },
    { key: "k_4", scoreLabel: "S_{1,4}", token: "Token 4", vector: "[1.18,1.21]", score: "1.026", tex: String.raw`\dfrac{0.04\times1.18+1.16\times1.21}{\sqrt{2}}` },
  ];

  return (
    <div className="fig">
      <div className="score-explainer" role="img" aria-label="用 Token 1 的 Query 与四个 Key 分别计算缩放点积，并组成分数矩阵 S 的第一行">
        <div className="score-explainer-head">
          <span>只展开 Token 1 的输出路径</span>
          <b>用 <Formula tex="q_1" /> 生成分数矩阵 <Formula tex="S" /> 的第 1 行</b>
          <p>“用 <Formula tex="q_1" /> 打分”不是给 <Formula tex="q_1" /> 自己评分，而是为了计算 <Formula tex="b_1" />，取 <Formula tex="Q" /> 的第 1 行 <Formula tex="q_1" />，分别衡量它与四个 Key 的匹配程度。</p>
        </div>

        <div className="score-symbol-guide">
          <section className="score-symbol-origin">
            <div className="score-symbol-heading">
              <span>先追踪一个格</span>
              <b><Formula tex="S_{1,1}" /> 从哪里来</b>
              <p>同一个 <Formula tex="x_1" /> 经过两组不同的投影参数，得到角色不同的 <Formula tex="q_1" /> 与 <Formula tex="k_1" />；它们做缩放点积，才产生一个分数。</p>
            </div>
            <div className="score-symbol-flow">
              <div className="score-symbol-node input">
                <span>Token 1 当前层输入</span>
                <Formula block tex={String.raw`x_1=[0.40, 1.20]`} />
                <small><Formula tex="X" /> 的第 1 行</small>
              </div>

              <div className="score-symbol-projections" aria-hidden="true">
                <div><span><Formula tex={String.raw`\times W^Q`} /></span><b>→</b></div>
                <div><span><Formula tex={String.raw`\times W^K`} /></span><b>→</b></div>
              </div>

              <div className="score-symbol-roles">
                <div className="query">
                  <span><Formula tex="Q" /> 的第 1 行 · 发起匹配</span>
                  <Formula block tex={String.raw`q_1=[0.04, 1.16]`} />
                </div>
                <div className="key">
                  <span><Formula tex="K" /> 的第 1 行 · 提供匹配</span>
                  <Formula block tex={String.raw`k_1=[0.56, 1.16]`} />
                </div>
              </div>

              <div className="score-symbol-operation" aria-hidden="true">
                <span>缩放点积</span>
                <b>→</b>
                <small><Formula tex={String.raw`\div\sqrt{d_k}`} /></small>
              </div>

              <div className="score-symbol-node result">
                <span><Formula tex="S" /> 的第 1 行、第 1 列</span>
                <Formula block tex={String.raw`S_{1,1}=\dfrac{q_1k_1^{\mathsf T}}{\sqrt{2}}`} />
                <strong><Formula tex={String.raw`=0.967`} /></strong>
              </div>
            </div>
          </section>

          <section className="score-index-map">
            <div className="score-symbol-heading">
              <span>再认两个下标</span>
              <b><Formula tex="S_{i,j}" /> 在矩阵中的坐标</b>
            </div>
            <FirstRowAttentionMatrix
              symbol="S"
              values={comparisons.map((item) => item.score)}
              ariaLabel="S 矩阵行由 q_i 决定，列由 k_j 决定；第一行依次为 0.967、负 0.186、0.573、1.026"
            />
            <div className="score-index-notes">
              <span><b>第一个下标 <Formula tex="i" /></b> → 选择 <Formula tex="q_i" /> → 确定第 <Formula tex="i" /> 行</span>
              <span><b>第二个下标 <Formula tex="j" /></b> → 选择 <Formula tex="k_j" /> → 确定第 <Formula tex="j" /> 列</span>
              <strong>一般地，<Formula tex="S_{i,j}" /> 表示 Token <Formula tex="i" /> 对 Token <Formula tex="j" /> 的原始关联分数；例如 <Formula tex="S_{1,2}=-0.186" /> 就是 Token 1 对 Token 2 的原始分数。</strong>
            </div>
          </section>
        </div>

        <div className="score-explainer-flow">
          <article className="score-query-card">
            <span>① 取 <Formula tex="Q" /> 的第 1 行</span>
            <b>Token 1 的 Query</b>
            <Formula block tex={String.raw`q_1=x_1W^Q=[0.04,\ 1.16]`} />
            <p>它是 Token 1 的检索向量。这里只算 <Formula tex="b_1" />，所以使用 <Formula tex="q_1" />；若算 <Formula tex="b_2" />，就改用 <Formula tex="q_2" />。</p>
          </article>

          <div className="score-flow-arrow" aria-hidden="true"><b>→</b><span>同一种计算<br />重复 4 次</span></div>

          <div className="score-comparisons">
            <div className="score-column-head">
              <span>② 分别与 <Formula tex="K" /> 的四行比较</span>
              <Formula tex={String.raw`S_{1,j}=q_1k_j^{\mathsf T}/\sqrt{d_k}`} />
            </div>
            {comparisons.map((item, i) => (
              <div className="score-comparison" key={item.key}>
                <div className="score-key-id">
                  <span>第 {i + 1} 列 · <Formula tex={item.scoreLabel} /></span>
                  <b><Formula tex={item.key} /></b>
                  <small>{item.token} · {item.vector}</small>
                </div>
                <Formula block tex={item.tex} />
                <strong>= {item.score}</strong>
              </div>
            ))}
          </div>

          <div className="score-flow-arrow" aria-hidden="true"><b>→</b><span>按列号<br />依次排好</span></div>

          <article className="score-row-card">
            <span>③ 得到 <Formula tex="S" /> 的第 1 行</span>
            <b>Token 1 看各 Token 的原始分数</b>
            <div className="score-row-matrix" aria-label="S 的第一行等于 0.967、负 0.186、0.573、1.026">
              <Formula tex="S_{1,:}=" />
              <div>{comparisons.map((item) => <span key={item.key}>{item.score}</span>)}</div>
            </div>
            <div className="score-row-labels">
              {comparisons.map((item, i) => (
                <div key={item.key}><span>列 {i + 1}</span><b>{item.token}</b><strong>{item.score}</strong></div>
              ))}
            </div>
            <p><b>行 1</b>来自 <Formula tex="q_1" />；<b>列 <Formula tex="j" /></b>来自 <Formula tex="k_j" />。因此 <Formula tex="S_{1,4}" /> 就是 Token 1 看 Token 4 的原始分数。</p>
          </article>
        </div>

        <div className="score-next-step">
          <b>此时还不是注意力权重。</b>
          <span>四个 <Formula tex="S_{1,j}" /> 只是可正可负的原始分数；下一步对整行做 <Formula tex={String.raw`A_{1,:}=\operatorname{softmax}(S_{1,:})`} />，才得到和为 1 的权重。</span>
        </div>
      </div>
      <div className="fig-cap">图 · 向量阶段 ② 打分：<Formula tex="q_1" /> 与四个 <Formula tex="k_j" /> 的缩放点积，依次组成分数矩阵 <Formula tex="S" /> 的第 1 行</div>
    </div>
  );
}

/* ============================================================
 * 向量阶段图3：整行 softmax（row-wise softmax）
 * 强调：4 个分数作为「一整行」联合归一化，而非 4 个独立操作
 * 数据：q1 主角行 scores=[0.967,-0.186,0.573,1.026]  weights=[0.328,0.103,0.221,0.348]
 * ============================================================ */
function FigStageSoftmax() {
  const scores = [0.967, -0.186, 0.573, 1.026];
  const weights = [0.328, 0.103, 0.221, 0.348];
  const exps = [2.631, 0.830, 1.773, 2.790];

  return (
    <div className="fig">
      <div className="softmax-figure" role="img" aria-label="整行 softmax：四个分数联合归一化为四个注意力权重">
        <div className="softmax-title">
          <b>整行 softmax · 四个分数联合归一化</b>
          <span>不是四次独立运算：每个输出权重都依赖这一行的全部分数</span>
        </div>
        <div className="softmax-flow">
          <div className="softmax-panel">
            <div className="softmax-step">① 缩放后的相关分数</div>
            <Formula block tex={String.raw`S_{1,j}=\dfrac{q_1\cdot k_j}{\sqrt{d_k}}`} />
            <Formula block tex={String.raw`S_{1,:}=[\,0.967,\ -0.186,\ 0.573,\ 1.026\,]`} />
            <p>整行输入：四个分数共同进入同一个 softmax。</p>
          </div>

          <div className="softmax-arrow"><span>整行</span>→</div>

          <div className="softmax-panel operator">
            <div className="softmax-step">② row-wise softmax</div>
            <strong className="softmax-name">SOFTMAX</strong>
            <Formula block tex={String.raw`A_{1,j}=\dfrac{e^{S_{1,j}}}{\sum_{u=1}^{4}e^{S_{1,u}}}`} />
            <p>分母使用另一索引 <Formula tex="u" /> 求和，表示整行四项共同决定归一化系数。</p>
          </div>

          <div className="softmax-arrow"><span>整行</span>→</div>

          <div className="softmax-panel">
            <div className="softmax-step">③ 注意力权重</div>
            <Formula block tex={String.raw`A_{1,:}=\operatorname{softmax}(S_{1,:})`} />
            <Formula block tex={String.raw`A_{1,:}\approx[\,0.328,\ 0.103,\ 0.221,\ 0.348\,]`} />
            <Formula block tex={String.raw`\sum_{j=1}^{4}A_{1,j}=1.000`} />
          </div>
        </div>

        <div className="softmax-matrix-guide">
          <div className="score-symbol-heading">
            <span>与 <Formula tex="S" /> 使用完全相同的行列坐标</span>
            <b>把归一化结果写入权重矩阵 <Formula tex="A" /></b>
            <p>softmax 只改变每个格子的数值，不改变坐标含义：第 <Formula tex="i" /> 行仍对应 Token <Formula tex="i" /> 的 Query，第 <Formula tex="j" /> 列仍对应 Token <Formula tex="j" />。</p>
          </div>
          <FirstRowAttentionMatrix
            symbol="A"
            values={weights.map((weight) => weight.toFixed(3))}
            ariaLabel="A 权重矩阵与 S 使用相同坐标；第一行依次为 0.328、0.103、0.221、0.348"
          />
          <div className="score-index-notes">
            <span><b><Formula tex="A_{1,2}=0.103" /></b>：Token 1 汇聚 Token 2 的 <Formula tex="v_2" /> 时，使用的权重为 10.3%</span>
            <span><b>同一行权重和为 1</b>：<Formula tex={String.raw`\sum_{j=1}^{4}A_{1,j}=1`} /></span>
            <strong><Formula tex="A_{1,4}=0.348" /> 最大，表示本例中 Token 1 从 Token 4 的 <Formula tex="v_4" /> 汇聚信息时采用的权重最高。</strong>
          </div>
        </div>

        <div className="softmax-derivation">
          <div className="softmax-step">逐列计算 · 四个分数指数化后共享同一个分母 Z</div>
          <div className="softmax-exp-row">
            {scores.map((score, i) => (
              <div key={`e-${i}`}>
                <Formula tex={`e^{S_{1,${i + 1}}}=e^{${score.toFixed(3)}}`} />
                <strong>≈ {exps[i].toFixed(3)}</strong>
              </div>
            ))}
            <div className="total">
              <Formula tex={String.raw`Z_1=\sum_{u=1}^{4}e^{S_{1,u}}`} />
              <strong>= 8.024</strong>
            </div>
          </div>
          <Formula block tex={String.raw`Z_1\approx2.631+0.830+1.773+2.790=8.024`} />
          <div className="softmax-weight-calcs">
            {weights.map((weight, i) => (
              <div key={`weight-calc-${i}`}>
                <span>第 {i + 1} 列 · <Formula tex={`A_{1,${i + 1}}`} /></span>
                <Formula block tex={`A_{1,${i + 1}}=\\dfrac{e^{S_{1,${i + 1}}}}{Z_1}`} />
                <Formula block tex={`=\\dfrac{${exps[i].toFixed(3)}}{8.024}`} />
                <strong>= {weight.toFixed(3)}</strong>
              </div>
            ))}
          </div>
          <Formula block tex={String.raw`A_{1,:}\approx[\,0.328,\ 0.103,\ 0.221,\ 0.348\,]`} />
        </div>

        <div className="softmax-key">
          <b>关键：</b>softmax 改变的是整行的相对分配；任意一个分数变化，四个权重都会重新计算。
        </div>
      </div>

      <div className="fig-cap">
        图 · 向量阶段 ③ softmax：整行四个分数共享同一个分母，归一化为和等于 1 的注意力权重
      </div>
    </div>
  );
}

function FigStageAggregate() {
  const rows = [
    { token: "Token 1", weight: "0.328", value: "[0.40,1.28]", contribution: "[0.13120,0.41984]" },
    { token: "Token 2", weight: "0.103", value: "[1.08,0.60]", contribution: "[0.11124,0.06180]" },
    { token: "Token 3", weight: "0.221", value: "[0.65,1.06]", contribution: "[0.14365,0.23426]" },
    { token: "Token 4", weight: "0.348", value: "[0.92,1.72]", contribution: "[0.32016,0.59856]" },
  ];

  return (
    <div className="fig">
      <div className="aggregate-walkthrough" aria-label="四个注意力权重分别乘对应的 Value，再逐项相加得到 Token 1 的新表示 b1">
        <div className="aggregate-heading">
          <span>向量阶段 ④ · 加权汇聚</span>
          <b><Formula tex="A" /> 的第 1 行，按编号 <Formula tex="j" /> 与 <Formula tex="V" /> 的第 <Formula tex="j" /> 行一一配对</b>
          <p>上一图已经算出 Token 1 的四个权重。每个 <Formula tex="A_{1,j}" /> 只乘编号相同的 <Formula tex="v_j" />；四项分别算完，再把四份贡献相加。</p>
        </div>

        <div className="aggregate-meaning">
          <div className="weight"><Formula block tex="A_{1,j}" /><span>取多少：Token 1 从 Token <Formula tex="j" /> 读取的比例</span></div>
          <div className="value"><Formula block tex="v_j" /><span>取什么：Token <Formula tex="j" /> 真正提供的内容</span></div>
          <div className="output"><Formula block tex="A_{1,j}v_j" /><span>这一项最终写入 <Formula tex="b_1" /> 的实际贡献</span></div>
        </div>

        <div className="aggregate-table-wrap">
          <table className="aggregate-table">
            <colgroup>
              <col className="source-col" />
              <col className="weight-col" />
              <col className="operator-col" />
              <col className="value-col" />
              <col className="arrow-col" />
              <col className="contribution-col" />
            </colgroup>
            <thead>
              <tr>
                <th>信息来源</th>
                <th>① 注意力权重</th>
                <th aria-label="乘以" />
                <th>② Value 内容</th>
                <th aria-label="得到" />
                <th>③ 加权后的实际贡献</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.token}>
                  <th scope="row"><span>第 {i + 1} 项</span><b>{row.token}</b></th>
                  <td className="weight">
                    <Formula block tex={`A_{1,${i + 1}}=${row.weight}`} />
                    <small>Token 1 的读取比例</small>
                  </td>
                  <td className="operator"><Formula tex={String.raw`\times`} /></td>
                  <td className="value">
                    <Formula block tex={`v_${i + 1}=${row.value}`} />
                    <small>{row.token} 提供的内容</small>
                  </td>
                  <td className="arrow">→</td>
                  <td className="contribution">
                    <Formula block tex={`A_{1,${i + 1}}v_${i + 1}=${row.contribution}`} />
                    <small>进入 <Formula tex="b_1" /> 的第 {i + 1} 份贡献</small>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="aggregate-component-sum">
          <div className="aggregate-component-head">
            <span>④ 逐维累加</span>
            <b>四份二维向量相加，结果仍然是一个二维向量</b>
            <p><b>4</b> 表示四个 Token 提供了四份贡献；<b>2</b> 表示每份贡献都有两个分量。求和消掉的是 Token 这一维，不会消掉向量内部的两个分量。</p>
          </div>

          <div className="aggregate-shape-rule" aria-label="四份一乘二向量沿 Token 维求和，得到一份一乘二向量">
            <div><strong>4 份</strong><span>每份都是 <Formula tex={String.raw`1\times2`} /> 行向量</span></div>
            <div className="sum"><Formula block tex={String.raw`\sum_{j=1}^{4}`} /><span>按相同位置分别相加</span></div>
            <div><strong>1 份</strong><span>结果仍是 <Formula tex={String.raw`1\times2`} /> 行向量</span></div>
          </div>

          <div className="aggregate-axis-sums">
            <div>
              <span>第 1 个分量只和第 1 个分量相加</span>
              <Formula block tex={String.raw`0.13120+0.11124+0.14365+0.32016=0.70625`} />
            </div>
            <div>
              <span>第 2 个分量只和第 2 个分量相加</span>
              <Formula block tex={String.raw`0.41984+0.06180+0.23426+0.59856=1.31446`} />
            </div>
          </div>

          <Formula
            block
            className="aggregate-recombine"
            tex={String.raw`b_1=\begin{bmatrix}0.70625&1.31446\end{bmatrix}\approx\begin{bmatrix}0.706&1.314\end{bmatrix}`}
          />
        </div>

        <div className="aggregate-sum-arrow" aria-hidden="true"><span>两个分量重新并排，得到 <Formula tex="b_1" /></span><b>↓</b></div>

        <div className="aggregate-result">
          <div className="aggregate-result-head">
            <span>⑤ 写入输出矩阵</span>
            <b><Formula tex="b_1" /> 是输出矩阵 <Formula tex="O" /> 的第 1 行</b>
          </div>
          <div className="aggregate-matrix-flow">
            <div className="aggregate-matrix-card vector">
              <span>本节实际算出的行向量</span>
              <Formula block tex={String.raw`b_1=\begin{bmatrix}0.706&1.314\end{bmatrix}`} />
              <small><Formula tex={String.raw`b_1\in\mathbb{R}^{1\times2}`} />，已经融合 Token 1～4 的 Value 信息</small>
            </div>

            <div className="aggregate-stack-arrow" aria-hidden="true">
              <span>放入第 1 行</span>
              <b>→</b>
            </div>

            <div className="aggregate-matrix-card matrix">
              <span>四个 Token 最终组成输出矩阵</span>
              <Formula
                block
                tex={String.raw`O=\begin{bmatrix}0.706&1.314\\b_{2,1}&b_{2,2}\\b_{3,1}&b_{3,2}\\b_{4,1}&b_{4,2}\end{bmatrix}\in\mathbb{R}^{4\times2}`}
              />
              <small>再用 <Formula tex={String.raw`q_2,q_3,q_4`} /> 重复同一流程，就能依次填满第 2～4 行</small>
            </div>
          </div>
        </div>

        <div className="aggregate-precision-note">
          表中使用上一图显示的三位小数权重，因此得到近似结果；使用 softmax 未舍入权重时为 <Formula tex={String.raw`[0.706397,1.313975]`} />，保留三位小数仍是 <Formula tex={String.raw`[0.706,1.314]`} />。
        </div>
      </div>
      <div className="fig-cap">图 · 向量阶段 ④ 汇聚：每个权重乘对应 <Formula tex="v_j" />，四份贡献相加得到 <Formula tex="b_1" /></div>
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

/* ---- 统一数据（token₁..token₄，d=2；禁止改值）---- */
const FMS_WORDS = tokenLabels;
const FMS_DATA: { S: Mat; A: Mat; O: Mat } = {
  S:  [[0.967, -0.186, 0.573, 1.026], [1.37, 1.067, 1.287, 2.024], [1.129, 0.258, 0.838, 1.4], [1.693, 0.245, 1.206, 2.04]],
  A:  [[0.328, 0.103, 0.221, 0.348], [0.218, 0.161, 0.201, 0.42], [0.287, 0.12, 0.215, 0.377], [0.306, 0.072, 0.188, 0.433]],
  O:  [[0.706, 1.314], [0.778, 1.311], [0.732, 1.317], [0.721, 1.38]],
};

/* 配色（字面色值，行内 style 用）——与全局 Q/K/V/Att/Out 一致 */
const FMS_PAL: Record<string, { c: string; t: string }> = {
  Q: { c: "#f5b042", t: "rgba(245,176,66,0.10)" },
  K: { c: "#a78bfa", t: "rgba(167,139,250,0.10)" },
  V: { c: "#2dd4bf", t: "rgba(45,212,191,0.10)" },
  S: { c: "#38bdf8", t: "rgba(56,189,248,0.10)" },
  A: { c: "#38bdf8", t: "rgba(56,189,248,0.10)" },
  O: { c: "#f472b6", t: "rgba(244,114,182,0.10)" },
};

const fmt = (v: number, d: number): string => (v === 0 ? 0 : v).toFixed(d);

/* ---- 单个带行列标签的矩阵网格 ---- */
function FmsMatGrid({
  data, name, shape, pal,
  rowLabels, colLabels, cornerLabel,
  heat = false, digits = 2, focusRow,
}: {
  data: Mat;
  name: React.ReactNode;
  shape: React.ReactNode;
  pal: { c: string; t: string };
  rowLabels?: string[];
  colLabels?: string[];
  cornerLabel?: React.ReactNode;
  heat?: boolean;
  digits?: number;
  focusRow?: number;
}) {
  const flat = data.flat();
  const min = heat ? Math.min(...flat) : 0;
  const max = heat ? Math.max(...flat) : 1;
  const range = Math.max(max - min, 1e-9);
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
              <tr key={i} className={i === focusRow ? "fms-focus-row" : ""}>
                {showRowLab && (
                  <th className="fms-rowlab" style={{ color: pal.c }}>
                    {rowLabels ? rowLabels[i] : ""}
                  </th>
                )}
                {row.map((v, j) => {
                  const t = heat ? (v - min) / range : 0;
                  const bg = heat
                    ? `rgba(56,189,248,${(0.1 + t * 0.75).toFixed(3)})`
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

/* ---- 主组件：把向量级第一行扩展成四行并行计算 ---- */
function FigMatrixStage() {
  return (
    <div className="fms-wrap">
      <div className="fms-canvas">
        <div className="fms-banner">
          <span className="fms-banner-tag">从第一行扩展到全部四行</span>
          <Formula block tex={String.raw`\underbrace{Q_{[4\times2]}K^{\mathsf T}_{[2\times4]}/\sqrt{2}}_{S_{[4\times4]}}\ \xrightarrow{\ \text{逐行 softmax}\ }\ A_{[4\times4]}\ \xrightarrow{\ \times V_{[4\times2]}\ }\ O_{[4\times2]}`} />
          <div className="fms-prep">
            <span>上一节已完成投影，不再重复展开：</span>
            <b style={{ color: FMS_PAL.Q.c }}><Formula tex={String.raw`Q\ [4\times2]`} /></b>
            <b style={{ color: FMS_PAL.K.c }}><Formula tex={String.raw`K\ [4\times2]`} /></b>
            <b style={{ color: FMS_PAL.V.c }}><Formula tex={String.raw`V\ [4\times2]`} /></b>
          </div>
        </div>

        <div className="fms-summary-flow" aria-label="Self-Attention 矩阵级三步计算链">
          <div className="fms-summary-card">
            <div className="fms-summary-head">
              <span>1 · 点积并缩放</span>
              <b>softmax 前分数 <Formula tex="S" /></b>
            </div>
            <Formula block tex={String.raw`S=QK^{\mathsf T}/\sqrt{d_k}`} />
            <FmsMatGrid name={<Formula tex="S" />} shape={<Formula tex={String.raw`[4\times4]`} />} pal={FMS_PAL.S} data={FMS_DATA.S} heat focusRow={0}
              rowLabels={FMS_WORDS} colLabels={FMS_WORDS} cornerLabel={<>行 <Formula tex="Q" />＼列 <Formula tex="K" /></>} digits={3} />
            <div className="fms-row-link"><b>第 1 行</b><Formula tex={String.raw`S_{1,:}=[0.967,-0.186,0.573,1.026]`} /></div>
          </div>

          <div className="fms-flow-op"><b>逐行</b><span>softmax</span><i>→</i></div>

          <div className="fms-summary-card">
            <div className="fms-summary-head">
              <span>2 · 每行独立归一化</span>
              <b>softmax 后权重 <Formula tex="A" /></b>
            </div>
            <Formula block tex={String.raw`A_{i,:}=\operatorname{softmax}(S_{i,:})`} />
            <FmsMatGrid name={<Formula tex="A" />} shape={<Formula tex={String.raw`[4\times4]`} />} pal={FMS_PAL.A} data={FMS_DATA.A} heat focusRow={0}
              rowLabels={FMS_WORDS} colLabels={FMS_WORDS} cornerLabel={<>行 <Formula tex="Q" />＼列 <Formula tex="K" /></>} digits={3} />
            <div className="fms-row-link"><b>第 1 行</b><Formula tex={String.raw`A_{1,:}=[0.328,0.103,0.221,0.348]`} /></div>
          </div>

          <div className="fms-flow-op"><b>每行</b><Formula tex={String.raw`\times V_{[4\times2]}`} /><i>→</i></div>

          <div className="fms-summary-card fms-output-card">
            <div className="fms-summary-head">
              <span>3 · 加权汇聚 Value</span>
              <b>新表示 <Formula tex="O" /></b>
            </div>
            <Formula block tex={String.raw`O=AV`} />
            <FmsMatGrid name={<Formula tex="O" />} shape={<Formula tex={String.raw`[4\times2]`} />} pal={FMS_PAL.O} data={FMS_DATA.O} focusRow={0}
              rowLabels={FMS_WORDS} colLabels={["维 1", "维 2"]} cornerLabel="token＼维" digits={3} />
            <div className="fms-row-link"><b>第 1 行</b><Formula tex={String.raw`O_{1,:}=b_1=[0.706,1.314]`} /></div>
          </div>
        </div>

        <div className="fms-legend-row">
          <div className="fms-legend">
            <span className="fms-focus-swatch" />
            <span className="fms-legend-text">橙色框 = 上一节逐项算过的 Token 1 路径</span>
          </div>
          <div className="fms-read">
            矩阵级没有新增算法：高亮的第一行就是上一节的 <Formula tex={String.raw`q_1\rightarrow S_{1,:}\rightarrow A_{1,:}\rightarrow b_1`} />；其余三行遵循同一规则，由 GPU 与第一行并行计算。
          </div>
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
      <text x={x + w / 2} y={y + h / 2 + (sub ? -5 : 5)} textAnchor="middle" fill={lc || "#eef3ff"} fontSize="15" fontWeight="700">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fill={sc || "#6e7aab"} fontSize="11.5">{sub}</text>}
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
        <text x="225" y="26" textAnchor="middle" fill="#38bdf8" fontSize="17" fontWeight="700">Encoder（左）</text>
        <text x="695" y="26" textAnchor="middle" fill="#f472b6" fontSize="17" fontWeight="700">Decoder（右）</text>

        <Box x={155} y={48} w={140} h={38} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Input Embedding" lc="#a9b4dc" />
        <Arrow d="M225,86 V90" />
        <circle cx="225" cy="106" r="14" fill="#070b18" stroke="#a78bfa" />
        <text x="225" y="111" textAnchor="middle" fill="#a78bfa" fontSize="15">+</text>
        <rect x="35" y="84" width="115" height="44" rx="8" fill="#0c1430" stroke="rgba(167,139,250,0.45)" />
        <text x="92.5" y="101" textAnchor="middle" fill="#a78bfa" fontSize="11.5">Positional</text>
        <text x="92.5" y="119" textAnchor="middle" fill="#a78bfa" fontSize="11.5">Encoding</text>
        <Arrow d="M150,106 H207" />
        <Arrow d="M225,120 V146" />
        <rect x="80" y="138" width="290" height="257" rx="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeDasharray="5 5" />
        <g>
          <rect x="91" y="128" width="126" height="20" rx="7" fill="#0c1430" stroke="rgba(56,189,248,0.28)" />
          <text x="154" y="142" textAnchor="middle" fill="#7e8ac0" fontSize="10.8" fontWeight="700">N× Encoder Layer</text>
        </g>
        <Box x={120} y={150} w={210} h={54} fill="rgba(56,189,248,0.14)" stroke="#38bdf8" label="Multi-Head Self-Attention" sub="本节讲的核心算子" lc="#38bdf8" sc="#6e7aab" />
        <Box x={150} y={222} w={150} h={36} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={120} y={278} w={210} h={50} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" label="Feed-Forward Network" sub="两层 MLP（逐位置作用）" lc="#2dd4bf" sc="#6e7aab" />
        <Box x={150} y={346} w={150} h={36} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Arrow d="M225,204 V218" /><Arrow d="M225,258 V274" /><Arrow d="M225,328 V342" /><Arrow d="M225,382 V401" />
        <Arrow d="M120,177 H100 V240 H150" color="#f5b042" dash="4 3" />
        <Arrow d="M120,303 H100 V364 H150" color="#f5b042" dash="4 3" />
        <Box x={150} y={405} w={150} h={34} fill="rgba(244,114,182,0.14)" stroke="#f472b6" label="编码器输出 Memory" lc="#f472b6" />

        <Box x={625} y={48} w={140} h={38} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Output Embedding" lc="#a9b4dc" />
        <Arrow d="M695,86 V90" />
        <circle cx="695" cy="106" r="14" fill="#070b18" stroke="#a78bfa" />
        <text x="695" y="111" textAnchor="middle" fill="#a78bfa" fontSize="15">+</text>
        <rect x="790" y="84" width="115" height="44" rx="8" fill="#0c1430" stroke="rgba(167,139,250,0.45)" />
        <text x="847.5" y="101" textAnchor="middle" fill="#a78bfa" fontSize="11.5">Positional</text>
        <text x="847.5" y="119" textAnchor="middle" fill="#a78bfa" fontSize="11.5">Encoding</text>
        <Arrow d="M790,106 H713" />
        <Arrow d="M695,120 V146" />
        <rect x="540" y="138" width="310" height="332" rx="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeDasharray="5 5" />
        <g>
          <rect x="551" y="128" width="126" height="20" rx="7" fill="#0c1430" stroke="rgba(244,114,182,0.28)" />
          <text x="614" y="142" textAnchor="middle" fill="#7e8ac0" fontSize="10.8" fontWeight="700">N× Decoder Layer</text>
        </g>
        <Box x={575} y={150} w={240} h={50} fill="rgba(245,176,66,0.14)" stroke="#f5b042" label="Masked Multi-Head Attention" sub="只能看过去（屏蔽未来位）" lc="#f5b042" sc="#6e7aab" />
        <Box x={620} y={216} w={150} h={34} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={575} y={268} w={240} h={50} fill="rgba(56,189,248,0.14)" stroke="#38bdf8" label="Cross-Attention" sub="Q: Decoder · K,V: Encoder Memory" lc="#38bdf8" sc="#6e7aab" />
        <Box x={620} y={334} w={150} h={34} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={575} y={386} w={240} h={44} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" label="Feed-Forward Network" lc="#2dd4bf" />
        <Box x={620} y={442} w={150} h={32} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Arrow d="M695,200 V212" /><Arrow d="M695,250 V264" /><Arrow d="M695,318 V330" /><Arrow d="M695,368 V382" /><Arrow d="M695,430 V438" />
        <Arrow d="M575,177 H555 V233 H620" color="#f5b042" dash="4 3" />
        <Arrow d="M575,293 H555 V351 H620" color="#f5b042" dash="4 3" />
        <Arrow d="M575,408 H555 V458 H620" color="#f5b042" dash="4 3" />
        <Arrow d="M300,422 H455 V293 H571" color="#a78bfa" dash="4 3" />
        <text x="312" y="411" fill="#a78bfa" fontSize="11.5" fontWeight="700">Encoder Memory</text>
        <text x="466" y="283" fill="#a78bfa" fontSize="11.5">作为 K、V 来源</text>
        <Arrow d="M695,474 V489" />
        <Box x={585} y={493} w={220} h={46} fill="rgba(167,139,250,0.14)" stroke="#a78bfa" label="Linear + Softmax" sub="输出下一 Token 的概率" lc="#a78bfa" sc="#7e8ac0" />

        <g transform="translate(80,555)">
          <rect x="0" y="0" width="14" height="14" rx="3" fill="rgba(56,189,248,0.14)" stroke="#38bdf8" /><text x="20" y="12" fill="#6e7aab" fontSize="13">Attention</text>
          <rect x="110" y="0" width="14" height="14" rx="3" fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" /><text x="130" y="12" fill="#6e7aab" fontSize="13">FFN</text>
          <rect x="190" y="0" width="14" height="14" rx="3" fill="#0c1430" stroke="rgba(255,255,255,0.08)" /><text x="210" y="12" fill="#6e7aab" fontSize="13">Add&amp;Norm</text>
          <rect x="310" y="0" width="14" height="14" rx="3" fill="#070b18" stroke="#a78bfa" /><text x="330" y="12" fill="#6e7aab" fontSize="13">位置编码</text>
          <line x1="430" y1="7" x2="447" y2="7" stroke="#f5b042" strokeWidth="1.5" strokeDasharray="4 3" /><text x="457" y="12" fill="#6e7aab" fontSize="13">残差连接</text>
        </g>
      </svg>
      <div className="fig-cap">图 · 论文 Figure 1 重绘 — Attention 在 Encoder/Decoder 中共出现三次，是同一算子</div>
    </div>
  );
}

function PositionEncodingFlow() {
  const contentEmbedding = [
    [0.3, 1.0],
    [1.3, 0.2],
    [0.5, 0.8],
    [0.7, 1.2],
  ];
  const positionEmbedding = [
    [0.1, 0.2],
    [0.2, 0.1],
    [0.3, 0.1],
    [0.4, 0.3],
  ];
  const attentionInput = [
    [0.4, 1.2],
    [1.5, 0.3],
    [0.8, 0.9],
    [1.1, 1.5],
  ];
  const rows = ["Token 1", "Token 2", "Token 3", "Token 4"];
  const cols = ["维 1", "维 2"];

  return (
    <div className="position-demo">
      <div className="position-demo-head">
        <span>进入第一层 Attention 之前</span>
        <b>内容矩阵 <Formula tex="E_{\mathrm{seq}}" /> + 位置矩阵 <Formula tex="P_{\mathrm{seq}}" /> = 输入矩阵 <Formula tex="X" /></b>
        <p>Self-Attention 只看向量间的关系，本身分不出 Token 的先后顺序。位置编码先把位置 <Formula tex="i" /> 的向量 <Formula tex="p_i" /> 注入内容 embedding <Formula tex="e_i" />，再把结果 <Formula tex="x_i" /> 送去生成 <Formula tex={String.raw`Q,\ K,\ V`} />。</p>
      </div>

      <div className="position-matrix-flow" role="img" aria-label="四个 Token 的内容向量逐行加上位置向量，得到 Attention 输入矩阵 X">
        <div className="position-matrix-stage">
          <span>① Token 内容</span>
          <FmsMatGrid data={contentEmbedding} name={<Formula tex="E_{\mathrm{seq}}" />} shape={<Formula tex={String.raw`[4\times2]`} />} pal={{ c: "#a9b4dc", t: "rgba(169,180,220,0.08)" }} rowLabels={rows} colLabels={cols} cornerLabel="位置＼维" digits={2} />
          <small>四个 <Formula tex="e_i" /> 按行堆叠；每行来自 <Formula tex="E_{\mathrm{tok}}" /> 的查表结果</small>
        </div>
        <div className="position-matrix-op"><b>+</b><span>逐行、逐维相加</span></div>
        <div className="position-matrix-stage">
          <span>② 位置信息</span>
          <FmsMatGrid data={positionEmbedding} name={<Formula tex="P_{\mathrm{seq}}" />} shape={<Formula tex={String.raw`[4\times2]`} />} pal={FMS_PAL.K} rowLabels={rows} colLabels={cols} cornerLabel="位置＼维" digits={2} />
          <small>第 <Formula tex="i" /> 行只对应序列位置 <Formula tex="i" /></small>
        </div>
        <div className="position-matrix-op"><b>=</b><span>得到当前层输入</span></div>
        <div className="position-matrix-stage result">
          <span>③ Attention 输入</span>
          <FmsMatGrid data={attentionInput} name={<Formula tex="X" />} shape={<Formula tex={String.raw`[4\times2]`} />} pal={FMS_PAL.Q} rowLabels={rows} colLabels={cols} cornerLabel="位置＼维" digits={2} focusRow={0} />
          <small>每行随后分别投影成 <Formula tex={String.raw`q_i,\ k_i,\ v_i`} /></small>
        </div>
      </div>

      <div className="position-token-example">
        <span>把第 1 行单独展开</span>
        <Formula block tex={String.raw`\underbrace{[0.30,\ 1.00]}_{e_1\;\text{内容}}+\underbrace{[0.10,\ 0.20]}_{p_1\;\text{位置}}=\underbrace{[0.40,\ 1.20]}_{x_1\;\text{输入}}`} />
        <p>这就是前文一直使用的 <Formula tex="x_1" />；随后才有 <Formula tex={String.raw`q_1=x_1W^Q=[0.04,\ 1.16]`} />。因此位置编码位于 <Formula tex={String.raw`Q,\ K,\ V`} /> 投影之前。</p>
      </div>

      <div className="position-methods">
        <article>
          <span>为什么位置表能“按位置取一行”</span>
          <b>one-hot 选择位置向量</b>
          <p>若使用可学习位置表 <Formula tex={String.raw`P_{\mathrm{pos}}\in\mathbb{R}^{L_{\max}\times d_{\mathrm{model}}}`} />（<Formula tex="L_{\max}" /> 是可表示的最大位置数），位置 <Formula tex="i" /> 的 one-hot 行向量 <Formula tex="r_i^{\mathsf T}" /> 只会选中 <Formula tex="P_{\mathrm{pos}}" /> 的第 <Formula tex="i" /> 行：</p>
          <Formula block tex={String.raw`r_3^{\mathsf T}=\begin{bmatrix}0&0&1&0&\cdots&0\end{bmatrix},\qquad p_3=r_3^{\mathsf T}P_{\mathrm{pos}}=(P_{\mathrm{pos}})_{3,:}`} />
          <p>把内容和 one-hot 先拼接，再乘下面这个分块矩阵，结果仍是相加；这解释了图里为什么可以直接用 <Formula tex="e_i+p_i" />：</p>
          <Formula block tex={String.raw`\begin{bmatrix}e_i&r_i^{\mathsf T}\end{bmatrix}\begin{bmatrix}I_{d_{\mathrm{model}}}\\P_{\mathrm{pos}}\end{bmatrix}=e_i+r_i^{\mathsf T}P_{\mathrm{pos}}=e_i+p_i=x_i`} />
          <p><Formula tex="I_{d_{\mathrm{model}}}" /> 是 <Formula tex="d_{\mathrm{model}}" /> 阶单位矩阵；它让内容向量 <Formula tex="e_i" /> 原样通过。</p>
        </article>

        <article>
          <span>原始 Transformer 的具体做法</span>
          <b>固定正弦 / 余弦位置编码</b>
          <Formula block tex={String.raw`\begin{aligned}\mathrm{PE}_{(\mathrm{pos},2f)}&=\sin\!\left(\frac{\mathrm{pos}}{10000^{2f/d_{\mathrm{model}}}}\right)\\\mathrm{PE}_{(\mathrm{pos},2f+1)}&=\cos\!\left(\frac{\mathrm{pos}}{10000^{2f/d_{\mathrm{model}}}}\right)\end{aligned}`} />
          <p>这里 <Formula tex="\mathrm{pos}" /> 是序列位置，<Formula tex="f" /> 是频率组索引。例如 <Formula tex={String.raw`d_{\mathrm{model}}=4,\ \mathrm{pos}=1`} /> 时，<Formula tex={String.raw`f=0,1`} /> 两组频率分别使用分母 1 和 100：</p>
          <Formula block tex={String.raw`\begin{aligned}\mathrm{PE}(1)&=\begin{bmatrix}\sin(1)&\cos(1)&\sin(0.01)&\cos(0.01)\end{bmatrix}\\&\approx\begin{bmatrix}0.8415&0.5403&0.0100&0.99995\end{bmatrix}\end{aligned}`} />
          <p>每个位置都由同一公式确定，不参与训练；不同维度使用不同频率，使模型能区分绝对位置与相对间距。</p>
        </article>
      </div>

      <div className="position-demo-note">上方 2 维矩阵沿用全文的可手算数字，用来说明“如何相加并接到 <Formula tex="q_1" />”；它不是原始论文正弦公式的实际输出。真实模型中 <Formula tex="d_{\mathrm{model}}" /> 更大，计算规则不变。</div>
    </div>
  );
}


/* ============================================================
 * SVG 图：普通 Attention vs FlashAttention 数据搬运对照
 * ============================================================ */
function FigFlashCompare() {
  // 每步：label 文字，bad=true 表示这是"写回/读回 HBM"的瓶颈步
  const normal: { t: React.ReactNode; bad?: boolean }[] = [
    { t: <Formula tex={String.raw`QK^{\mathsf T}`} /> },
    { t: <>完整 <Formula tex="S" /> 写入 HBM</>, bad: true },
    { t: <>读回 <Formula tex="S" />，softmax</> },
    { t: <>完整 <Formula tex="A" /> 写入 HBM</>, bad: true },
    { t: <>读回 <Formula tex="A" />，乘 <Formula tex="V" /></> },
    { t: <Formula tex="O" /> },
  ];
  const flash: { t: React.ReactNode; good?: boolean }[] = [
    { t: <>切分 <Formula tex={String.raw`Q,\ K,\ V`} /></> },
    { t: "小块加载进 SRAM", good: true },
    { t: "片上算局部分数" },
    { t: <>在线更新 <Formula tex={String.raw`m,\ \ell,\ o`} /></>, good: true },
    { t: <>处理下一个 <Formula tex={String.raw`K,\ V`} /> 块</> },
    { t: <>写出最终 <Formula tex="O" /> 与行归一化统计量</> },
  ];
  const Step = ({ t, tone }: { t: React.ReactNode; tone?: "bad" | "good" }) => (
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
          <div className="fc-col-h good">FlashAttention · 分块进片上，只写最终 <Formula tex="O" /> 与每行统计量</div>
          {flash.map((s, i) => (
            <div key={i} className="fc-line">
              <Step t={s.t} tone={s.good ? "good" : undefined} />
              {i < flash.length - 1 && <span className="fc-dn">↓</span>}
            </div>
          ))}
        </div>
      </div>
      <div className="fig-cap">图 · 左侧每物化一次 <Formula tex={String.raw`L\times L`} /> 矩阵都要一次 HBM 写与读；右侧在 SRAM 内完成累加，<Formula tex={String.raw`\mathcal O(L^2)`} /> 中间矩阵从不落地显存</div>
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
  const [qIdx, setQIdx] = useState(0);
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
        <div className="sub">Transformer 核心算子</div>
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
            <p className="lead">Attention 根据 Query 与 Key 的匹配程度计算权重，再据此汇聚 Value，使每个位置获得当前最相关的信息。</p>
            <div className="chips">
              <span>从矩阵乘法起步</span>
              <span><b><Formula tex={String.raw`Q,\ K,\ V`} /></b> 全程配色一致</span>
              <span>含 PyTorch 经典代码</span>
            </div>

            <div className="hero-card">
              <div className="tag">CORE EQUATION</div>
              <div className="eq">
                <Formula block tex={String.raw`\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V`} />
              </div>
              <div className="shapes">
                <b className="q"><Formula tex={String.raw`Q\ [B,h,L_q,d_k]`} /></b>
                <b className="k"><Formula tex={String.raw`K\ [B,h,L_k,d_k]`} /></b>
                <b className="v"><Formula tex={String.raw`V\ [B,h,L_k,d_v]`} /></b>
                <b style={{ color: "var(--att)", borderColor: "rgba(56,189,248,.4)" }}><Formula tex={String.raw`S,A\ [B,h,L_q,L_k]`} /></b>
                <b style={{ color: "var(--out)", borderColor: "rgba(244,114,182,.4)" }}><Formula tex={String.raw`O\ [B,h,L_q,d_v]`} /></b>
              </div>
              <div className="shape-note"><Formula tex="B" />：批大小；<Formula tex="h" />：头数；<Formula tex={String.raw`L_q,L_k`} />：Query / Key 序列长度；<Formula tex={String.raw`d_k,d_v`} />：每个头的 Key / Value 维度。这里的 <Formula tex="O" /> 是各头尚未拼接的输出；下文手算例取 <Formula tex={String.raw`B=h=1`} />，因此省略这两个长度为 1 的轴。</div>
            </div>

            <div className="grid2" style={{ marginTop: 28 }}>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>这个算子为什么重要</h3>
                <p className="t3">它是多数主流大模型（GPT / LLaMA / Claude，以及大量文生图、文生视频模型）的共同骨架。掌握它，等于拿到理解生成式 AI 主流路线的钥匙。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>算子视角的一句话</h3>
                <p className="t3">Attention 的核心就是<b style={{ color: "#eef3ff" }}>两次矩阵乘法（<Formula tex={String.raw`QK^{\mathsf T}`} /> 计算相关分数、<Formula tex="AV" /> 加权汇聚）+ 一个 softmax</b>；完整多头还包括 <Formula tex={String.raw`Q,\ K,\ V`} /> 与 <Formula tex="W^O" /> 四个投影，共六次矩阵乘法；QKV 融合后则是四次 GEMM。</p>
              </div>
            </div>
          </section>

          {/* ===== 矩阵乘法 ===== */}
          <section className="section" id="s1">
            <SecHead idx="01" title="热身：矩阵乘法到底怎么乘" />
            <p className="sec-lead">Attention 的主要线性运算由<b style={{ color: "#eef3ff" }}>矩阵乘法</b>完成，中间穿插缩放、mask 和 softmax。先用具体数字把矩阵乘法规则搞明白——<b style={{ color: "#f472b6" }}>点一下右边结果矩阵的任意格子</b>，左边高亮参与计算的行与列。</p>
            <div className="mbox">
              <div className="mcol">
                <div className="mname"><Formula tex={String.raw`U_1\ [2\times3]`} /></div>
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
                <div className="mname"><Formula tex={String.raw`U_2\ [3\times2]`} /></div>
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
                <div className="mname"><Formula tex={String.raw`U_3\ [2\times2]`} /> ← 点击</div>
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
              <Formula tex={`(U_3)_{${mr + 1},${mc + 1}}=${rowA.map((v, i) => `${v}\\times${colB[i]}`).join("+")}=${matrixC[mr][mc]}`} />
            </div>
            <div className="note">规则只有一句：<Formula tex={String.raw`(U_3)_{i,j}=\sum_{r=1}^{3}(U_1)_{i,r}(U_2)_{r,j}`} />，即 <Formula tex="U_1" /> 的第 <Formula tex="i" /> 行与 <Formula tex="U_2" /> 的第 <Formula tex="j" /> 列<b>逐项相乘再相加</b>。这里用 <Formula tex={String.raw`U_1,U_2,U_3`} />，把字母 <Formula tex="A" /> 留给后文的注意力权重矩阵。</div>
          </section>

          {/* ===== 向量级 ===== */}
          <section className="section" id="s2">
            <SecHead idx="02" title="Self-Attention · 向量级（一步一步算）" />
            <p className="sec-lead">先不急着做点积。下面先说明每个符号从哪里来，再只展开 Token 1，完整算出它如何读取 Token 1～4 的信息。没有 causal mask 时，四个位置都能按同样方式并行计算。</p>

            <AttentionSetupGuide />

            <h3>只展开 Token 1：从 <Formula tex="q_1" /> 到新表示 <Formula tex="b_1" /></h3>
            <p className="section-bridge">为了演示一行怎样算，这里只求 Token 1 的输出 <Formula tex="b_1" />，所以取 <Formula tex="Q" /> 的第 1 行 <Formula tex={String.raw`q_1=x_1W^Q`} />。<Formula tex="q_1" /> 并非特殊变量：每个 <Formula tex="q_i" /> 都负责生成分数矩阵 <Formula tex="S" /> 的第 <Formula tex="i" /> 行；本节展开第 1 行，其余三行算法完全相同。</p>
            <div className="legend-row">
              <span><i className="lq" />Query：用于发起匹配</span>
              <span><i className="lk" />Key：用于被 Query 匹配</span>
              <span><i className="lv" />Value：真正被加权汇聚的内容</span>
            </div>
            <div className="calculation-route" aria-label="Token 1 的四步 Attention 计算路线">
              <div><b>① 投影</b><Formula block tex={String.raw`X\rightarrow Q,K,V`} /><span>准备 <Formula tex="q_1" />、全部 <Formula tex="k_j" /> 与全部 <Formula tex="v_j" /></span></div>
              <div><b>② 生成 <Formula tex="S" /> 的第 1 行</b><Formula block tex={String.raw`S_{1,j}=q_1k_j^{\mathsf T}/\sqrt{d_k}`} /><span><Formula tex="q_1" /> 与四个 <Formula tex="k_j" /> 分别比较，得到四个原始分数</span></div>
              <div><b>③ 归一化</b><Formula block tex={String.raw`A_{1,:}=\operatorname{softmax}(S_{1,:})`} /><span>四个分数共同变成和为 1 的权重</span></div>
              <div><b>④ 汇聚</b><Formula block tex={String.raw`b_1=\sum_j A_{1,j}v_j`} /><span>按权重组合四个 Value</span></div>
            </div>

            <FigStageQKV />
            <FigStageScore />
            <FigStageSoftmax />
            <FigStageAggregate />

            <div className="note">换成 <Formula tex={String.raw`q_2,q_3,q_4`} /> 重复同一过程，就分别得到 <Formula tex={String.raw`b_2,b_3,b_4`} />。把四个输出按行堆叠，便是 <Formula tex={String.raw`O=\begin{bmatrix}b_1\\b_2\\b_3\\b_4\end{bmatrix}`} />；下一节用矩阵一次算完这四行。</div>
          </section>

          {/* ===== 矩阵级 ===== */}
          <section className="section detail-section" id="s3">
            <SecHead idx="03" title="Self-Attention · 矩阵级" />
            <p className="sec-lead">上一节只展开输出矩阵的第一行；这里把 <Formula tex={String.raw`q_1,\ldots,q_4`} />、<Formula tex={String.raw`k_1,\ldots,k_4`} />、<Formula tex={String.raw`v_1,\ldots,v_4`} /> 分别按行堆成 <Formula tex={String.raw`Q,K,V`} />，一次并行得到全部四行输出。数学没有变化，只是从“逐个位置理解”切换到“矩阵整体执行”。</p>
            <div className="matrix-level-bridge">
              <div><b>只追踪 Token 1</b><Formula block tex={String.raw`b_1=\operatorname{softmax}\!\left(\frac{q_1K^{\mathsf T}}{\sqrt{d_k}}\right)V`} /></div>
              <span>四个 Query 行同时执行 →</span>
              <div><b>一次得到 Token 1～4</b><Formula block tex={String.raw`O=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V`} /></div>
            </div>
            <ScoreMatrixReadingGuide />
            <div className="note"><b>记号约定（先说清楚，避免和代码对不上）</b>：本文把 Token 按行堆叠，记作 <Formula tex={String.raw`Q_{\mathrm r}=X_{\mathrm r}W_{\mathrm r}^Q`} />，因此分数为 <Formula tex={String.raw`Q_{\mathrm r}K_{\mathrm r}^{\mathsf T}`} />。若把 Token 按列堆叠，则 <Formula tex={String.raw`X_{\mathrm c}=X_{\mathrm r}^{\mathsf T}`} />、<Formula tex={String.raw`W_{\mathrm c}^Q=(W_{\mathrm r}^Q)^{\mathsf T}`} />，于是 <Formula tex={String.raw`Q_{\mathrm c}=W_{\mathrm c}^QX_{\mathrm c}`} />，<Formula tex="K_{\mathrm c}" /> 同理；同一张“Query 行、Key 列”分数矩阵满足 <Formula tex={String.raw`Q_{\mathrm r}K_{\mathrm r}^{\mathsf T}=Q_{\mathrm c}^{\mathsf T}K_{\mathrm c}`} />。代码采用行向量约定，所以需要 <code>key.transpose(-2, -1)</code>。</div>
            <FigMatrixStage />
            <div className="note">现在这句公式对你不再是一串符号：<Formula tex={String.raw`QK^{\mathsf T}`} /> 是「两两算相关度」，softmax 是「分数变权重」，乘 <Formula tex={String.raw`V`} /> 是「按权重取内容」。上面这组数值，正是把向量级 4 步压成矩阵后一次性算出的结果。</div>
          </section>

          {/* ===== Scale 与 Mask ===== */}
          <section className="section detail-section" id="s4">
            <SecHead idx="04" title={<>缩放 <Formula tex={String.raw`\sqrt{d_k}`} /> 与 Mask</>} />
            <p className="sec-lead">公式里多了一个「除以 <Formula tex={String.raw`\sqrt{d_k}`} />」，叫<b style={{ color: "#eef3ff" }}>缩放（Scale）</b>。原因一句：维度 <Formula tex="d_k" /> 越大，点积数值越大，softmax 更易饱和（趋于一个 1、其余 0），梯度可能变得很小。</p>
            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>不缩放会怎样</h3>
                <p className="t3">点积是 <Formula tex="d_k" /> 个乘积之和。<Formula tex="d_k" /> 大 → 点积方差大 → softmax 更易饱和近似 one‑hot → 梯度可能变得很小。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>除以 <Formula tex={String.raw`\sqrt{d_k}`} /> 的效果</h3>
                <p className="t3">把点积方差<b style={{ color: "#eef3ff" }}>拉回 1 附近</b>，让 softmax 处在温和区间，梯度健康。这是算子实现里最易漏、但必须有的细节。</p>
              </div>
            </div>
            <div className="note">
              在各维近似独立、均值 0、方差 1 的假设下：
              <Formula block tex={String.raw`\operatorname{Var}(q_i k_j^{\mathsf T})\approx d_k\quad\Rightarrow\quad \operatorname{Var}\!\left(\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}\right)\approx1`} />
            </div>

            {/* 数值演示：选 Query 看权重 */}
            <h3>数值演示：选一个 Query，看它把注意力分给谁</h3>
            <div className="tabs">
              {tokenLabels.map((w, i) => (
                <button key={w} className={`tab ${qIdx === i ? "active" : ""}`} onClick={() => setQIdx(i)}><Formula tex={`q_${i + 1}`} /> · {w}</button>
              ))}
            </div>
            <div className="card">
              <div style={{ display: "flex", gap: 22, flexWrap: "wrap", justifyContent: "center", alignItems: "center" }}>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">缩放分数 <Formula className="mname-formula" tex={String.raw`{\color{#38bdf8}S_{${qIdx + 1},:}=q_{${qIdx + 1}}K^{\mathsf T}/\sqrt{d_k}}`} /></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.scaled.map((v, i) => (
                      <div key={i} style={{ textAlign: "center" }}>
                        <div className="mcell" style={{ cursor: "default" }}>{v.toFixed(2)}</div>
                        <div className="mname" style={{ marginTop: 4 }}><Formula tex={`k_${i + 1}`} /></div>
                      </div>
                    ))}
                  </div>
                </div>
                <span className="msign">→</span>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">权重 <Formula className="mname-formula" tex={String.raw`{\color{#38bdf8}A_{${qIdx + 1},:}=\operatorname{softmax}(S_{${qIdx + 1},:})}`} /></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.weights.map((v, i) => (
                      <div key={i} style={{ textAlign: "center" }}>
                        <div className="mcell" style={{ cursor: "default", backgroundColor: `rgba(56,189,248,${(0.12 + v * 0.6).toFixed(3)})`, borderColor: "#38bdf8" }}>{(v * 100).toFixed(0)}%</div>
                        <div className="mname" style={{ marginTop: 4 }}><Formula tex={`k_${i + 1}`} /></div>
                      </div>
                    ))}
                  </div>
                </div>
                <span className="msign">→</span>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">输出 <Formula className="mname-formula" tex={String.raw`{\color{#f472b6}b_{${qIdx + 1}}=\sum_j A_{${qIdx + 1},j}v_j}`} /></div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {attn.output.map((v, i) => (
                      <div key={i} className="mcell" style={{ cursor: "default", color: "#f472b6", borderColor: "#f472b6" }}>{v.toFixed(2)}</div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* —— Mask：因果掩码与 padding —— */}
            <h3>Mask：让每个位置「只看到该看的」</h3>
            <p className="sec-lead">生成时每个位置必须「看到过去、看不到未来」。实现上用一个<b style={{ color: "#f5b042" }}>下三角因果掩码（Causal Mask）</b>：第 <Formula tex="i" /> 个位置只允许查看索引 <Formula tex={String.raw`0,\ldots,i`} /> 的 Key；padding 位则用 padding mask 屏蔽。</p>
            <div className="note">为了展示生成序列的起始位置，下面在 Token 1～4 前额外加入一个 <code>&lt;BOS&gt;</code>，所以示意矩阵是 <Formula tex={String.raw`5\times5`} />；它只用于解释 mask，不改变前面 <Formula tex={String.raw`4\times4`} /> 的数值算例。</div>

            <div className="mask-grid">
              <table>
                <tbody>
                  <tr>
                    <th></th>
                    <th><Formula tex="k_0" /><br /><small style={{ color: "#4d577f" }}>&lt;BOS&gt;</small></th>
                    <th><Formula tex="k_1" /><br /><small style={{ color: "#4d577f" }}>Token 1</small></th>
                    <th><Formula tex="k_2" /><br /><small style={{ color: "#4d577f" }}>Token 2</small></th>
                    <th><Formula tex="k_3" /><br /><small style={{ color: "#4d577f" }}>Token 3</small></th>
                    <th><Formula tex="k_4" /><br /><small style={{ color: "#4d577f" }}>Token 4</small></th>
                  </tr>
                  <tr>
                    <th><Formula tex="q_0" /> · &lt;BOS&gt;</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th><Formula tex="q_1" /> · Token 1</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th><Formula tex="q_2" /> · Token 2</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th><Formula tex="q_3" /> · Token 3</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell block">−∞</div></td>
                  </tr>
                  <tr>
                    <th><Formula tex="q_4" /> · Token 4</th>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                    <td><div className="mask-cell allow">0</div></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="fig-cap">因果掩码矩阵：绿色 <b style={{ color: "#34d399" }}><Formula tex="0" /></b> = 允许看，红色 <b style={{ color: "#f472b6" }}><Formula tex="-\infty" /></b> = 屏蔽。第 <Formula tex="i" /> 行只允许查看第 <Formula tex={String.raw`0,\ldots,i`} /> 列</div>

            <h3 style={{ marginTop: 30 }}>Mask 加在哪一步</h3>
            <div className="flow-chain">
              <b><Formula tex={String.raw`QK^{\mathsf T}`} /></b><em>→</em>
              <b>÷ <Formula tex={String.raw`\sqrt{d_k}`} /></b><em>→</em>
              <b className="hi">+ Mask（−∞）</b><em>→</em>
              <b>softmax</b><em>→</em>
              <b><Formula tex={String.raw`\times V`} /></b>
            </div>
            <div className="eq-box">
              <Formula block tex={String.raw`O=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V,\quad M_{i,j}=\begin{cases}0 & i\ge j\\ -\infty & i<j\end{cases}`} />
            </div>
            <div className="note warn"><b>实现陷阱</b>：Mask 必须在 softmax <b>之前</b>加 <code>−∞</code>。若在 softmax 后再乘 0，屏蔽位虽然归零，但剩余权重之和不再为 1，输出尺度会出错。</div>

            <h3 style={{ marginTop: 24 }}>同一个因果约束，两种执行方式</h3>
            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>推理（逐 token，串行）</h3>
                <p className="t3" style={{ fontFamily: "var(--mono)", fontSize: 13 }}>
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt;</b> → 预测 Token 1<br />
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt; Token 1</b> → 预测 Token 2<br />
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt; Token 1 Token 2</b> → 预测 Token 3<br />
                  ……直到 <b style={{ color: "#f5b042" }}>&lt;end&gt;</b>
                </p>
                <p className="t3">每步只能用已生成的内容，天生串行；配合 KV cache 时 Key 本身就只有历史前缀，不一定需要显式的完整三角 mask。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>训练（整句并行）</h3>
                <p className="t3" style={{ fontFamily: "var(--mono)", fontSize: 13 }}>
                  输入 <b style={{ color: "#2dd4bf" }}>&lt;BOS&gt; Token 1 Token 2 Token 3</b>（右移一位）<br />
                  目标 <b style={{ color: "#f472b6" }}>Token 1 Token 2 Token 3 Token 4</b><br />
                  一次前向 + 因果 Mask
                </p>
                <p className="t3">训练靠 causal mask 一次性实现整句并行，每个位置「假装只看到过去」——因果约束由 mask 显式施加。</p>
              </div>
            </div>
          </section>

          {/* ===== 多头 ===== */}
          <section className="section detail-section" id="s5">
            <SecHead idx="05" title="多头注意力（Multi-Head）" />
            <p className="sec-lead">单头 attention 只在一组 <Formula tex={String.raw`Q,\ K,\ V`} /> 投影子空间里建模关系。拆成<b style={{ color: "#eef3ff" }}>多个头</b>，每个头用各自独立的可学习矩阵把输入<b>投影到不同子空间</b>再算 attention，就允许多个子空间并行捕捉不同关系，最后拼回来。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`\operatorname{head}_i=\operatorname{Attention}(XW_i^Q,\,XW_i^K,\,XW_i^V)`} />
              <Formula block tex={String.raw`\operatorname{MHA}=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\,W^O`} />
            </div>
            <div className="note">其中 <Formula tex="h" /> 是头数，<Formula tex={String.raw`W^O\in\mathbb{R}^{(h d_v)\times d_{\mathrm{model}}}`} /> 把拼接结果投影回模型维度。通常取 <Formula tex={String.raw`d_k=d_v=d_{\mathrm{model}}/h`} />，所以主 FLOPs 量级与单头接近、表达能力更强；但投影层、显存占用与调度开销并不为零，并非真的「免费」。</div>

            <h3>不同头可能形成不同的关注模式</h3>
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
                      {tokenLabels.map((w) => <td key={w} style={{ textAlign: "center", fontFamily: "var(--mono)", fontSize: 16, color: "#6e7aab", width: 110 }}>{w}</td>)}
                    </tr>
                    {heads[headIdx].matrix.map((row, i) => (
                      <tr key={i}>
                        <td style={{ textAlign: "right", paddingRight: 16, fontFamily: "var(--mono)", fontSize: 16, color: "#6e7aab" }}>{tokenLabels[i]}</td>
                        {row.map((v, j) => (
                          <td key={j}>
                            <div style={{
                              width: 104, height: 74, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center",
                              fontFamily: "var(--mono)", fontSize: 18, fontWeight: 700,
                              background: `rgba(56,189,248,${(0.08 + v * 0.7).toFixed(3)})`, border: "1px solid rgba(56,189,248,0.3)",
                              color: v > 0.4 ? "#fff" : "#a9b4dc",
                            }}>{v.toFixed(2)}</div>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="fig-cap"><b>人工示意</b>（非真实训练结果）：{heads[headIdx].note} · 行=Query（谁在问）· 列=Key（看谁）</div>
            </div>
          </section>

          {/* ===== FlashAttention ===== */}
          <section className="section detail-section" id="s6">
            <SecHead idx="06" title="FlashAttention：不改变数学，改变算的方式" />
            <p className="sec-lead">前面每一步都算清楚了。现在换个角度：这套 <Formula tex={String.raw`\operatorname{softmax}(QK^{\mathsf T}/\sqrt{d_k})V`} /> 在 GPU 上到底怎么跑？结论一句话——<b style={{ color: "#eef3ff" }}>FlashAttention 没有改变 Attention 的数学定义，改变的是计算顺序和数据搬运方式</b>。</p>

            <div className="eq-box">
              <Formula block tex={String.raw`O=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V`} />
            </div>

            <h3>长序列朴素 Attention 常受 HBM 访存限制</h3>
            <p className="sec-lead">对长度为 <Formula tex="L" /> 的 self-attention（此时 <Formula tex={String.raw`L_q=L_k=L`} />），朴素实现会把整张 <Formula tex="L\times L" /> 的分数矩阵 <Formula tex="S" /> 和权重矩阵 <Formula tex="A" /> 写进 HBM（显存）再读回。序列一长，<b>显存读写</b>就可能成为主要开销——具体瓶颈取决于序列长度、head dimension、硬件和实现。FlashAttention 重点优化这个 IO 瓶颈：<b style={{ color: "#2dd4bf" }}>把 <Formula tex={String.raw`Q,\ K,\ V`} /> 切成小块，分批搬进 SRAM，在片上计算并在线更新，只写回最终输出 <Formula tex="O" /> 与反向所需的行归一化统计量</b>——中间的 <Formula tex="L\times L" /> 矩阵不在 HBM 中物化。</p>

            <FigFlashCompare />

            <h3>在线 softmax：不存全矩阵也能归一化</h3>
            <p className="sec-lead">难点在 softmax 的每行分母 <Formula tex={String.raw`\ell_i=\sum_j e^{S_{i,j}}`} /> 需要「看到整行」。固定一个含 <Formula tex="T_q" /> 行的 Query 块 <Formula tex="Q_{\mathrm{blk}}" />，再按块编号 <Formula tex="t" /> 依次扫描 <Formula tex={String.raw`K_t,\ V_t`} />；<Formula tex="M_{\mathrm{blk},t}" /> 是与当前分数块同形状的 mask。新块中若出现更大的分数，旧累加值必须按新最大值<b style={{ color: "#eef3ff" }}>重新缩放</b>。三个状态都按 Query 行维护：<Formula tex={String.raw`m,\ell\in\mathbb{R}^{T_q}`} /> 是行最大值与归一化系数，<Formula tex={String.raw`o\in\mathbb{R}^{T_q\times d_v}`} /> 是尚未归一化的输出累加矩阵。</p>
            <div className="eq-box">
              <Formula block tex={String.raw`S_t=Q_{\mathrm{blk}}K_t^{\mathsf T}/\sqrt{d_k}+M_{\mathrm{blk},t},\qquad m_0=-\infty,\ \ell_0=0,\ o_0=0`} />
              <Formula block tex={String.raw`m_{\mathrm{new}}=\max\!\left(m,\,\operatorname{rowmax}(S_t)\right)`} />
              <Formula block tex={String.raw`\ell_{\mathrm{new}}=e^{m-m_{\mathrm{new}}}\odot\ell+\operatorname{rowsum}\!\left(e^{S_t-m_{\mathrm{new}}}\right)`} />
              <Formula block tex={String.raw`o_{\mathrm{new}}=e^{m-m_{\mathrm{new}}}\odot o+e^{S_t-m_{\mathrm{new}}}V_t`} />
            </div>
            <div className="note"><Formula tex={String.raw`\operatorname{rowmax},\ \operatorname{rowsum}`} /> 都按行计算，<Formula tex="\odot" /> 表示按 Query 行广播的逐元素缩放。全部 Key / Value 块处理完后，<Formula tex={String.raw`O_{\mathrm{blk}}=\operatorname{diag}(\ell)^{-1}o`} /> 表示把 <Formula tex="o" /> 的每一行除以对应的 <Formula tex="\ell_i" />。当新块出现更大分数时，旧累加按 <Formula tex="e^{m-m_{\mathrm{new}}}" /> 缩小，因此<b>无需保存过去的全部分数</b>。</div>

            <h3>三个要点</h3>
            <div className="grid3">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>精确，不是近似</h3>
                <p className="t3">FlashAttention 与朴素 Attention 的实数数学定义等价——它不是稀疏或低秩近似。只是浮点归约顺序不同，结果在误差容限内一致，不保证逐位相等。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>省的是显存与读写</h3>
                <p className="t3">中间矩阵 <Formula tex={String.raw`S,\ A`} /> 不落地 HBM；固定 head dimension 时，额外中间存储从 <Formula tex={String.raw`\mathcal O(L^2)`} /> 降到 <Formula tex={String.raw`\mathcal O(L)`} />，HBM 访问量也大幅减少。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>渐进复杂度不变</h3>
                <p className="t3">注意力主体仍是 <Formula tex={String.raw`\mathcal O\!\left(L^2(d_k+d_v)\right)`} /> 量级；但在线归一化有额外运算、反向可能靠重算换显存，实际运算条数并非完全不变。省下的是访存与中间存储，收益随 shape / dtype / 硬件 / mask 变化。</p>
              </div>
            </div>
            <div className="note">参考：FlashAttention（Dao 等，2022）、FlashAttention‑2（Dao，2023）在分块与并行划分上进一步优化。下一节看它如何对应到真实的算子调用与测试。</div>
          </section>

          {/* ===== Transformer 收尾定位 ===== */}
          <section className="section detail-section" id="s7">
            <SecHead idx="07" title="代码与算子测试：从原理到真实算子" />
            <p className="sec-lead">原理看懂了，落到代码就两层：一层<b style={{ color: "#eef3ff" }}>透明的参考实现</b>用于对照，一层<b style={{ color: "#2dd4bf" }}>真实算子</b>用于生产。算子测试就围着这两层展开。</p>

            <div className="code-title">① 透明参考实现 — 逐行对应公式</div>
            <pre><code>{`import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

def attention_ref(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)          # QK^T
    scores = scores / math.sqrt(q.size(-1))   # / sqrt(d_k)
    if mask is not None:
        # 这里只支持 additive float mask（屏蔽位 = -inf）。
        # 仅对 SDPA：布尔 mask 的 True=允许参与；这与
        # nn.MultiheadAttention 的 key_padding_mask 语义相反。
        # 若传入布尔 mask，需先转成 additive：
        # mask = torch.where(bool_mask, 0.0, float("-inf"))
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)   # softmax 按行
    return weights @ v                         # · V`}</code></pre>
            <div className="note">这几行就是前面所有图的代码化身。它<b>正确但不够快</b>：会把整张分数矩阵物化到显存，长序列下又慢又费显存。生产里用下面这行替代。</div>

            <div className="code-title">② 真实算子 — PyTorch SDPA（按输入与设备选可用后端）</div>
            <pre><code>{`output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask,     # causal / padding / 自定义
    dropout_p=0.0,
    is_causal=False,
)
# 需要对照不同后端时，可强制选择内核：
# with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION): ...`}</code></pre>
            <div className="note"><b>F.scaled_dot_product_attention</b>（SDPA）根据输入形状、数据类型、设备和可用内核，从 math、Flash、Memory‑Efficient 等后端中选择可用实现——FlashAttention 只是其中之一。某个 fused 后端受限时会尝试其他可用实现，最终才可能回退到 math；它既不保证命中 flash，也不是实测后选择「最快」（后端种类随版本演进，新版还有 cuDNN 等）。可用 <code>sdpa_kernel</code> 强制指定后端做对照验证。它与参考实现实数等价；<b>只有命中 fused 后端时</b>才能避免/减少完整注意力矩阵的显存物化，回退到 math 则和参考实现一样会物化中间量。</div>

            <div className="code-title">③ 多头自注意力 — reshape → transpose → SDPA → concat → <Formula tex="W^O" /></div>
            <pre><code>{`class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        # 与上文 Q=XW^Q 等无偏置公式保持一致
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        # 投影 + 拆头: (B, L, d_model) -> (B, h, L, d_k)
        q = self.wq(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        # 每个头独立做 SDPA（后端由框架选）: (B, h, L, d_k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )
        # 拼头: (B, h, L, d_k) -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.h * self.d_k)
        return self.wo(out)`}</code></pre>
            <div className="note">多头的工程本质就是<b>投影 → reshape/transpose 拆头 → 对每个头调 SDPA → 拼头 → 输出投影</b>。拆头靠 reshape + transpose 改变维度排布，让 <Formula tex="h" /> 个头作为独立 batch 维度并行计算，无需循环。上面设 <code>bias=False</code> 以严格对应无偏置公式；PyTorch 的 <code>nn.Linear</code> 内部计算 <Formula tex={String.raw`xW_{\mathrm{store}}^{\mathsf T}`} />，因此本文行向量公式里的 <Formula tex="W^Q" /> 对应 <Formula tex={String.raw`W_{\mathrm{store}}^{\mathsf T}`} />。</div>

            <h3>算子测试，重点看这五类</h3>
            <div className="grid3">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>① 前向正确性</h3>
                <p className="t3">Flash / SDPA 输出与透明参考实现逐元素比对；覆盖不同 shape、<Formula tex={String.raw`[B,1,1,L_k]`} /> 等可广播到 <Formula tex={String.raw`[B,h,L_q,L_k]`} /> 的 mask、<Formula tex={String.raw`L_q\ne L_k`} /> 的 cross-attention、非连续内存等输入组合。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>② 反向正确性</h3>
                <p className="t3">设训练损失为 <Formula tex="\mathcal L" />，比较 <Formula tex={String.raw`\nabla_Q\mathcal L,\ \nabla_K\mathcal L,\ \nabla_V\mathcal L`} />，不能只验前向——梯度路径才是算子真正容易出错的地方。</p>
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

            <div className="note ok"><b>一句话总结</b>：Attention 把「每个位置该关注谁」变成 <Formula tex={String.raw`QK^{\mathsf T}\!/\sqrt{d_k}`} /> 算分、softmax 变权重、再乘 <Formula tex="V" /> 取内容；多头扩展视角，FlashAttention 换算的方式不换数学，配上位置编码装进 Encoder/Decoder，就是撑起多数主流大模型的 Transformer。</div>
          </section>

          {/* ===== 代码 ===== */}
          <section className="section detail-section" id="s8">
            <SecHead idx="08" title="Transformer 全景：Attention 被装在哪里" />
            <p className="sec-lead">Attention 本身只是一个算子。把它装进完整模型，就是这篇被引用几万次的论文——<b style={{ color: "#38bdf8" }}>左 Encoder</b>、<b style={{ color: "#f472b6" }}>右 Decoder</b>，各堆叠 <Formula tex="N" /> 层。</p>
            <FigTransformer />
            <div className="grid2">
              <div className="note"><b>Encoder</b>：对源序列做 Self‑Attention + FFN，逐层提炼表示，输出的 Memory 作为 Cross‑Attention 的 <Formula tex={String.raw`K,\ V`} /> 来源。</div>
              <div className="note"><b>Decoder</b>：先用 <b>Masked</b> Self‑Attention（屏蔽未来位防作弊），再通过 <b>Cross‑Attention</b> 把编码器 Memory 投影成 <Formula tex={String.raw`K,\ V`} /> 来读取，最后预测下一个词。</div>
            </div>
            <div className="note warn">这张图里 <b>Attention 出现了三次</b>（Encoder 自注意、Decoder 掩码自注意、Decoder 交叉注意）——是同一个算子的三种调用。这一节只为定位：讲清楚 Attention 在整个模型里扮演什么角色。</div>

            <h3>现代大模型怎么取舍</h3>
            <div className="grid3">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>原始 Transformer</h3>
                <p className="t3">Encoder–Decoder 完整结构，用于翻译等 seq2seq 任务。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>BERT（Encoder‑only）</h3>
                <p className="t3">只保留 Encoder，双向自注意力，擅长理解类任务（分类、抽取）。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>GPT / LLaMA（Decoder‑only）</h3>
                <p className="t3">只保留 Decoder，主要用 causal self‑attention 自回归生成——没有独立 Encoder 和 Cross‑Attention。</p>
              </div>
            </div>

            <h3>位置编码：把「顺序」补回去</h3>
            <p className="sec-lead">这里把之前省略的计算链补完整：先说明位置向量怎样与内容向量组成 <Formula tex="X" />，再说明位置表如何取值，最后代入原始 Transformer 的正弦 / 余弦公式。</p>
            <PositionEncodingFlow />
            <div className="note">现代模型未必沿用“输入端直接相加”：例如 <b>RoPE</b> 在每层对 <Formula tex={String.raw`Q,\ K`} /> 做与位置相关的旋转，<b>ALiBi</b> 在注意力分数上加入线性位置偏置。注入位置不同，但目标相同——让 Attention 能感知顺序与距离。</div>
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
