"use client";

import { useEffect, useState } from "react";
import katex from "katex";
import {
  attentionDemo,
  attentionHeads,
  matrixMultiplicationDemo,
  multiHeadDemo,
} from "./attention-demo.js";

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
const { A: matrixA, B: matrixB, C: matrixC } = matrixMultiplicationDemo;

/* ---------- Attention 演示数据（全页统一使用 4 个位置标签，d=2） ---------- */
const tokenLabels = ["token₁", "token₂", "token₃", "token₄"];
const { Q: queries, K: keys, V: values } = attentionDemo;

/* ---------- 多头热力图数据 ---------- */
const heads = attentionHeads;

function formatNumber(value: number, digits = 3) {
  return value.toFixed(digits);
}

function displayNumber(value: number, digits = 3) {
  return formatNumber(value, digits).replace("-", "−");
}

function rowVectorTex(values: number[], digits = 3) {
  return String.raw`\begin{bmatrix}${values.map((value) => formatNumber(value, digits)).join("&")}\end{bmatrix}`;
}

function matrixTex(matrix: number[][], digits = 2) {
  return String.raw`\begin{bmatrix}${matrix
    .map((row) => row.map((value) => formatNumber(value, digits)).join("&"))
    .join(String.raw`\\`)}\end{bmatrix}`;
}

function roundedMatrix(matrix: number[][], digits = 3) {
  return matrix.map((row) =>
    row.map((value) => Number(value.toFixed(digits))),
  );
}


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
            fontSize: 14,
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
                    {Number(v.toFixed(2))}
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
            fontSize: 12,
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
        <b>Token、<Formula tex="x_i" />、<Formula tex="X" /> 与 <Formula tex="q_1" /> 的来源</b>
        <p><code>Token 1～4</code> 表示序列中的四个位置。第 1～3 步说明原始 Transformer 如何准备首层输入 <Formula tex="X" />；第 4 步开始进入完整 Self-Attention 模块，其中缩放点积核心接收投影后的 <Formula tex={String.raw`Q,\ K,\ V`} />。</p>
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
          <p>文本经 tokenizer 转换成 token ID 序列，每个 ID 按其序列索引对应一个位置。</p>
        </article>

        <article className="setup-node">
          <span className="setup-step">2</span>
          <strong>Token ID 变为输入向量 <Formula tex="x_i" /></strong>
          <Formula block tex={String.raw`\begin{aligned}u_i&=E_{\mathrm{tok}}[\operatorname{tokenId}_i]\\e_i&=\sqrt{d_{\mathrm{model}}}\,u_i\\x_i&=e_i+p_i\end{aligned}`} />
          <p>Embedding 表按 ID 取出一行，把离散编号转换成可计算的 <Formula tex="d_{\mathrm{model}}" /> 维内容向量。原始 Transformer 再做缩放并加入位置编码；这些属于模型的首层输入处理，不属于 Attention 算子。</p>
        </article>

        <article className="setup-node">
          <span className="setup-step">3</span>
          <strong>按位置堆叠成输入矩阵 <Formula tex="X" /></strong>
          <Formula block tex={`X=\\begin{bmatrix}x_1\\\\x_2\\\\x_3\\\\x_4\\end{bmatrix}=${matrixTex(attentionDemo.X, 1)}`} />
          <p><Formula tex={String.raw`X\in\mathbb{R}^{4\times2}`} />：4 表示四个位置，2 表示每个位置暂用两个数描述。后续数值演示从这份已经准备好的 <Formula tex="X" /> 开始。</p>
        </article>

        <article className="setup-node">
          <span className="setup-step">4</span>
          <strong>进入 Self-Attention：投影为 <Formula tex={String.raw`Q,\ K,\ V`} /></strong>
          <Formula block tex={String.raw`\begin{aligned}Q&=XW^Q\\K&=XW^K\\V&=XW^V\end{aligned}`} />
          <Formula block tex={String.raw`q_1=x_1W^Q`} />
          <p><Formula tex="q_1" /> 不是额外生成的变量：它就是 <Formula tex="Q" /> 的第一行，对应 Token 1；<Formula tex={String.raw`k_1,\ v_1`} /> 同理。</p>
        </article>
      </div>

      <div className="setup-dimension-note">
        <b>为什么只有二维？</b>
        <span>单头数值算例取 <Formula tex={String.raw`d_{\mathrm{model}}=d_k=d_v=2`} />，可以逐项核算矩阵乘法。真实模型通常使用数百到数千维，计算规则相同。</span>
      </div>

      <div className="setup-table-wrap">
        <table className="setup-table">
          <thead>
            <tr><th>符号</th><th>数学含义</th><th>从哪里来</th></tr>
          </thead>
          <tbody>
            <tr>
              <th>Token i</th>
              <td>序列中的第 i 个位置标签；它本身不是向量</td>
              <td>tokenizer 输出 token ID 序列后，由它在序列中的索引确定</td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`u_i`} /></th>
              <td>从可训练 Embedding 表中查出的内容向量</td>
              <td><Formula tex={String.raw`u_i=E_{\mathrm{tok}}[\operatorname{tokenId}_i]`} />；用离散 ID 选择表中的一行</td>
            </tr>
            <tr>
              <th><Formula tex="x_i" /></th>
              <td>第 i 个位置送入当前 Attention 层的行向量</td>
              <td>原始 Transformer 首层为 <Formula tex={String.raw`x_i=\sqrt{d_{\mathrm{model}}}\,u_i+p_i`} />；后续层来自上一层输出</td>
            </tr>
            <tr>
              <th><Formula tex="X" /></th>
              <td>把 <Formula tex={String.raw`x_1,\ldots,x_4`} /> 按行堆叠后的输入矩阵</td>
              <td>数值算例中 <Formula tex={String.raw`X\in\mathbb{R}^{4\times2}`} /></td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`W^Q,W^K,W^V`} /></th>
              <td>三组独立的线性投影参数：<Formula tex={String.raw`W^Q,W^K\in\mathbb{R}^{d_{\mathrm{model}}\times d_k}`} />，<Formula tex={String.raw`W^V\in\mathbb{R}^{d_{\mathrm{model}}\times d_v}`} />；数值算例中均为 <Formula tex={String.raw`2\times2`} /></td>
              <td>训练开始时初始化，训练中由反向传播学习；推理时保持固定</td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`q_i,k_i,v_i`} /></th>
              <td>位置 <Formula tex="i" /> 的 Query、Key、Value 行向量；<Formula tex={String.raw`q_i,k_i\in\mathbb{R}^{d_k}`} />，<Formula tex={String.raw`v_i\in\mathbb{R}^{d_v}`} /></td>
              <td><Formula tex={String.raw`q_i=x_iW^Q,\ k_i=x_iW^K,\ v_i=x_iW^V`} /></td>
            </tr>
            <tr>
              <th><Formula tex={String.raw`S,A`} /></th>
              <td>缩放后的相关分数矩阵、softmax 后的权重矩阵；数值算例中均属于 <Formula tex={String.raw`\mathbb{R}^{4\times4}`} /></td>
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
        <div><b>可训练参数</b><span>Transformer 输入层包含 Embedding 表 <Formula tex="E_{\mathrm{tok}}" />；Self-Attention 模块包含投影矩阵 <Formula tex={String.raw`W^Q,\ W^K,\ W^V`} />。训练初始通常随机，随后由训练学习；推理时不再重新随机。</span></div>
        <div><b>前向中间量</b><span><Formula tex={String.raw`X,\ Q,\ K,\ V,\ S,\ A,\ O`} />：随输入变化，每次前向重新计算，不是模型单独保存的参数。</span></div>
        <div><b>数值算例</b><span>矩阵使用固定数值，便于逐项核算；这些数值不对应某个已训练模型。</span></div>
      </div>
    </div>
  );
}

function ScoreMatrixReadingGuide() {
  return (
    <div className="score-reading-guide">
      <div className="score-reading-head">
        <span>矩阵坐标</span>
        <b><Formula tex={String.raw`4\times4`} /> 分数矩阵的行、列与单元格</b>
        <p>全部 <Formula tex={String.raw`4\times4`} /> 数值矩阵沿用相同的行列语义。</p>
      </div>
      <div className="score-reading-rules">
        <article className="row-rule">
          <span>① 行坐标</span>
          <b><Formula tex="i" /> → Query <Formula tex="q_i" /></b>
          <p>第 <Formula tex="i" /> 行表示 Token <Formula tex="i" /> 发起查询，回答“谁在看”。</p>
        </article>
        <article className="col-rule">
          <span>② 列坐标</span>
          <b><Formula tex="j" /> → Key <Formula tex="k_j" /></b>
          <p>第 <Formula tex="j" /> 列表示 Token <Formula tex="j" /> 被匹配，回答“正在看谁”。</p>
        </article>
        <article className="cell-rule">
          <span>③ 单元格</span>
          <b><Formula tex="(i,j)" /> → 分数 <Formula tex="S_{i,j}" /></b>
          <Formula block tex={String.raw`S_{i,j}=\dfrac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}`} />
          <p>例如 <Formula tex="S_{1,2}" /> 就是 Token 1 看 Token 2 的原始分数。</p>
        </article>
      </div>
    </div>
  );
}

function FigStageQKV() {
  const WORDS = tokenLabels;
  const { X, WQ, WK, WV, Q, K, V } = attentionDemo;

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
      <div className="qkv-projection-head">
        <div className="qkv-projection-title">
          第 1 步 · 投影：同一份 <span style={{ color: "var(--t2)" }}><Formula tex="X" /></span> 乘三个权重矩阵，准备全部{" "}
          <span style={{ color: CQ }}><Formula tex="Q" /></span> / <span style={{ color: CK }}><Formula tex="K" /></span> /{" "}
          <span style={{ color: CV }}><Formula tex="V" /></span>
        </div>
        <div className="qkv-projection-formulas">
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
        <div className="qkv-projection-branches">
          {rows.map((r) => (
            <div
              key={r.key}
              className="qkv-projection-branch"
              style={{
                background: `${r.color}0d`,
                border: `1px solid ${r.color}40`,
              }}
            >
              <span className="qkv-projection-operator">×</span>
              <QKVMat data={r.W} accent={r.color} label={r.wlabel} sub={<Formula tex={String.raw`[2\times2]`} />} />
              <span className="qkv-projection-operator">=</span>
              <QKVMat data={r.R} accent={r.color} label={<Formula tex={r.key} />} sub={<Formula tex={String.raw`[4\times2]`} />} heroRow={r.key === "Q" ? 0 : -1} heroColor={r.color} />
              <div className="qkv-projection-note">
                <Formula block tex={r.tex} />
                <div>{r.note}</div>
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
          fontSize: 13,
          fontFamily: "var(--mono)",
        }}
      >
        <span>例：<Formula tex={String.raw`x_1=[0.4,\ 1.2]`} /></span>
        <span style={{ color: CQ }}>→ <Formula tex={String.raw`q_1=[0.4\times1.0+1.2\times(-0.3),\ 0.4\times0.5+1.2\times0.8]=[0.04,\ 1.16]`} /> ★</span>
        <span style={{ color: CK }}>→ <Formula tex={String.raw`k_1=[0.4\times0.8+1.2\times0.2,\ 0.4\times(-0.4)+1.2\times1.1]=[0.56,\ 1.16]`} /></span>
        <span style={{ color: CV }}>→ <Formula tex={String.raw`v_1=[0.4\times0.7+1.2\times0.1,\ 0.4\times0.2+1.2\times1.0]=[0.40,\ 1.28]`} /></span>
      </div>
      <div style={{ textAlign: "center", color: "var(--t3)", fontSize: 13, marginTop: 10 }}>
        图 · 每一行都对应同序号 Token。以第 1 行的 <b style={{ color: CQ }}><Formula tex={String.raw`q_1=[0.04,\ 1.16]`} /></b> 为例，
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
  const q1 = attentionDemo.Q[0];
  const comparisons = attentionDemo.K.map((keyVector, index) => ({
    key: `k_${index + 1}`,
    scoreLabel: `S_{1,${index + 1}}`,
    token: `Token ${index + 1}`,
    vector: `[${keyVector.map((value) => formatNumber(value, 2)).join(",")}]`,
    score: displayNumber(attentionDemo.S[0][index]),
    tex: `\\dfrac{${formatNumber(q1[0], 2)}\\times${formatNumber(keyVector[0], 2)}+${formatNumber(q1[1], 2)}\\times${keyVector[1] < 0 ? `(${formatNumber(keyVector[1], 2)})` : formatNumber(keyVector[1], 2)}}{\\sqrt{2}}`,
  }));
  const scoreList = comparisons.map((item) => item.score).join("、");

  return (
    <div className="fig">
      <div className="score-explainer" role="img" aria-label="用 Token 1 的 Query 与四个 Key 分别计算缩放点积，并组成分数矩阵 S 的第一行">
        <div className="score-explainer-head">
          <span>只展开 Token 1 的输出路径</span>
          <b>用 <Formula tex="q_1" /> 生成分数矩阵 <Formula tex="S" /> 的第 1 行</b>
          <p>“用 <Formula tex="q_1" /> 打分”表示取 <Formula tex="Q" /> 的第 1 行，分别衡量 <Formula tex="q_1" /> 与四个 Key 的匹配程度，由此计算 Token 1 的输出 <Formula tex="b_1" />。</p>
        </div>

        <div className="score-symbol-guide">
          <section className="score-symbol-origin">
            <div className="score-symbol-heading">
              <span>单个分数的来源</span>
              <b><Formula tex="S_{1,1}" /> 从哪里来</b>
              <p>同一个 <Formula tex="x_1" /> 经过两组不同的投影参数，得到角色不同的 <Formula tex="q_1" /> 与 <Formula tex="k_1" />；它们做缩放点积，才产生一个分数。</p>
            </div>
            <div className="score-symbol-flow">
              <div className="score-symbol-node input">
                <span>Token 1 当前层输入</span>
                <Formula block tex={`x_1=${rowVectorTex(attentionDemo.X[0], 2)}`} />
                <small><Formula tex="X" /> 的第 1 行</small>
              </div>

              <div className="score-symbol-projections" aria-hidden="true">
                <div><span><Formula tex={String.raw`\times W^Q`} /></span><b>→</b></div>
                <div><span><Formula tex={String.raw`\times W^K`} /></span><b>→</b></div>
              </div>

              <div className="score-symbol-roles">
                <div className="query">
                  <span><Formula tex="Q" /> 的第 1 行 · 发起匹配</span>
                  <Formula block tex={`q_1=${rowVectorTex(q1, 2)}`} />
                </div>
                <div className="key">
                  <span><Formula tex="K" /> 的第 1 行 · 提供匹配</span>
                  <Formula block tex={`k_1=${rowVectorTex(attentionDemo.K[0], 2)}`} />
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
                <strong><Formula tex={`=${formatNumber(attentionDemo.S[0][0])}`} /></strong>
              </div>
            </div>
          </section>

          <section className="score-index-map">
            <div className="score-symbol-heading">
              <span>下标与矩阵坐标</span>
              <b><Formula tex="S_{i,j}" /> 在矩阵中的坐标</b>
            </div>
            <FirstRowAttentionMatrix
              symbol="S"
              values={comparisons.map((item) => item.score)}
              ariaLabel={`S 矩阵行由 q_i 决定，列由 k_j 决定；第一行依次为 ${scoreList}`}
            />
            <div className="score-index-notes">
              <span><b>第一个下标 <Formula tex="i" /></b> → 选择 <Formula tex="q_i" /> → 确定第 <Formula tex="i" /> 行</span>
              <span><b>第二个下标 <Formula tex="j" /></b> → 选择 <Formula tex="k_j" /> → 确定第 <Formula tex="j" /> 列</span>
              <strong>一般地，<Formula tex="S_{i,j}" /> 表示 Token <Formula tex="i" /> 对 Token <Formula tex="j" /> 的原始关联分数；例如 <Formula tex={`S_{1,2}=${formatNumber(attentionDemo.S[0][1])}`} /> 就是 Token 1 对 Token 2 的原始分数。</strong>
            </div>
          </section>
        </div>

        <div className="score-explainer-flow">
          <article className="score-query-card">
            <span>① 取 <Formula tex="Q" /> 的第 1 行</span>
            <b>Token 1 的 Query</b>
            <Formula block tex={`q_1=${rowVectorTex(q1, 2)}`} />
            <p>它是 Token 1 的 Query 向量。计算 <Formula tex="b_1" /> 时使用 <Formula tex="q_1" />；计算 <Formula tex="b_2" /> 时使用 <Formula tex="q_2" />。</p>
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
                  <small><span>{item.token}</span><span>{item.key} = {item.vector}</span></small>
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
            <div className="score-row-matrix" aria-label={`S 的第一行等于 ${scoreList}`}>
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
 * 数据直接来自统一数值链路。
 * ============================================================ */
function FigStageSoftmax() {
  const scores = attentionDemo.S[0];
  const weights = attentionDemo.A[0];
  const exps = scores.map(Math.exp);
  const normalizer = exps.reduce((sum, value) => sum + value, 0);
  const maximumIndex = weights.indexOf(Math.max(...weights));

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
            <Formula block tex={String.raw`S_{1,:}\approx${rowVectorTex(scores)}`} />
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
            <Formula block tex={String.raw`A_{1,:}\approx${rowVectorTex(weights)}`} />
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
            ariaLabel={`A 权重矩阵与 S 使用相同坐标；第一行依次为 ${weights.map((weight) => formatNumber(weight)).join("、")}`}
          />
          <div className="score-index-notes">
            <span><b><Formula tex={`A_{1,2}=${formatNumber(weights[1])}`} /></b>：Token 1 汇聚 Token 2 的 <Formula tex="v_2" /> 时，使用的权重约为 {(weights[1] * 100).toFixed(1)}%</span>
            <span><b>同一行权重和为 1</b>：<Formula tex={String.raw`\sum_{j=1}^{4}A_{1,j}=1`} /></span>
            <strong><Formula tex={`A_{1,${maximumIndex + 1}}=${formatNumber(weights[maximumIndex])}`} /> 最大，表示 Token 1 从 Token {maximumIndex + 1} 的 <Formula tex={`v_${maximumIndex + 1}`} /> 汇聚信息时采用的权重最高。</strong>
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
              <strong>= {formatNumber(normalizer)}</strong>
            </div>
          </div>
          <Formula block tex={String.raw`Z_1\approx${exps.map((value) => formatNumber(value)).join("+")}=${formatNumber(normalizer)}`} />
          <div className="softmax-example">
            <span>以第 2 列为例；其余三列使用同一个分母</span>
            <Formula block tex={String.raw`A_{1,2}=\dfrac{e^{S_{1,2}}}{Z_1}\approx\dfrac{${formatNumber(exps[1])}}{${formatNumber(normalizer)}}=${formatNumber(weights[1])}`} />
          </div>
        </div>

        <div className="softmax-key">
          <b>关键：</b>softmax 改变的是整行的相对分配；任意一个分数变化，四个权重都会重新计算。实际内核通常先减去该行最大值再求指数，数学结果不变，数值更稳定。
        </div>
      </div>

      <div className="fig-cap">
        图 · 向量阶段 ③ softmax：整行四个分数共享同一个分母，归一化为和等于 1 的注意力权重
      </div>
    </div>
  );
}

function FigStageAggregate() {
  const weights = attentionDemo.A[0];
  const output = attentionDemo.O[0];
  const componentFormula = (dimension: number) => weights
    .map((weight, index) => `${formatNumber(weight)}\\times${formatNumber(values[index][dimension], 2)}`)
    .join("+");

  return (
    <div className="fig">
      <div className="aggregate-walkthrough" aria-label="A 的第一行乘完整 V 矩阵，得到 Token 1 的二维输出 b1">
        <div className="aggregate-heading">
          <span>向量阶段 ④ · 加权汇聚</span>
          <b><Formula tex="A" /> 的第 1 行乘完整 <Formula tex="V" /> 矩阵，直接得到 <Formula tex="b_1" /></b>
          <p>四个权重负责在 Token 维上取舍信息；<Formula tex="V" /> 的两列是两个特征维度，所以同一行权重会分别与两列做点积，最终保留两个输出分量。</p>
        </div>

        <div className="aggregate-equation">
          <Formula
            block
            tex={String.raw`\underbrace{${rowVectorTex(weights)}}_{A_{1,:}\;[1\times4]}\;
              \underbrace{${matrixTex(values, 2)}}_{V\;[4\times2]}
              =\underbrace{${rowVectorTex(output)}}_{b_1\;[1\times2]}`}
          />
        </div>

        <div className="aggregate-rule">
          <Formula block tex={String.raw`b_1=A_{1,:}V=\sum_{j=1}^{4}A_{1,j}v_j`} />
          <p><Formula tex="A_{1,j}" /> 与 <Formula tex="V" /> 的第 <Formula tex="j" /> 行 <Formula tex="v_j" /> 一一对应；求和消掉四个 Token 这一维，不会消掉 <Formula tex="V" /> 的两个特征维度。</p>
        </div>

        <div className="aggregate-axis-sums">
          <article>
            <span><Formula tex="V" /> 第 1 列 → <Formula tex="b_{1,1}" /></span>
            <Formula block tex={String.raw`b_{1,1}\approx${componentFormula(0)}=${formatNumber(output[0])}`} />
          </article>
          <article>
            <span><Formula tex="V" /> 第 2 列 → <Formula tex="b_{1,2}" /></span>
            <Formula block tex={String.raw`b_{1,2}\approx${componentFormula(1)}=${formatNumber(output[1])}`} />
          </article>
        </div>

        <div className="aggregate-result">
          <b><Formula tex="b_1" /> 就是输出矩阵 <Formula tex="O=AV" /> 的第 1 行</b>
          <span>换成 <Formula tex={String.raw`A_{2,:},A_{3,:},A_{4,:}`} /> 重复同一矩阵乘法，就依次得到 <Formula tex={String.raw`b_2,b_3,b_4`} />。</span>
        </div>
      </div>
      <div className="fig-cap">图 · 向量阶段 ④ 汇聚：<Formula tex="A_{1,:}" /> 与 <Formula tex="V" /> 的每一列做点积，得到二维输出 <Formula tex="b_1" /></div>
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

/* ---- 统一数据（token₁..token₄，d=2）---- */
const FMS_WORDS = tokenLabels;
const FMS_DATA: { S: Mat; A: Mat; O: Mat } = {
  S: roundedMatrix(attentionDemo.S),
  A: roundedMatrix(attentionDemo.A),
  O: roundedMatrix(attentionDemo.O),
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
            <span><Formula tex={String.raw`Q,\ K,\ V`} /> 已由输入投影得到：</span>
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
            <div className="fms-row-link"><b>第 1 行</b><Formula tex={String.raw`S_{1,:}=${rowVectorTex(attentionDemo.S[0])}`} /></div>
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
            <div className="fms-row-link"><b>第 1 行</b><Formula tex={String.raw`A_{1,:}=${rowVectorTex(attentionDemo.A[0])}`} /></div>
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
            <div className="fms-row-link"><b>第 1 行</b><Formula tex={String.raw`O_{1,:}=b_1=${rowVectorTex(attentionDemo.O[0])}`} /></div>
          </div>
        </div>

        <div className="fms-legend-row">
          <div className="fms-legend">
            <span className="fms-focus-swatch" />
            <span className="fms-legend-text">橙色框 = Token 1 的逐项计算路径</span>
          </div>
          <div className="fms-read">
            高亮第一行对应 <Formula tex={String.raw`q_1\rightarrow S_{1,:}\rightarrow A_{1,:}\rightarrow b_1`} />；其余三行遵循同一规则，并可与第一行并行计算。
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
 * Multi-Head 数值链：投影 → 拆头 → 各头 Attention → 拼接 → W^O
 * ============================================================ */

const MH_CHANNELS = ["维 1", "维 2"];
const MH_HEAD_COUNT = 3;
const MH_HEAD_WIDTH = 2;
const MH_HEAD_CLASSES = ["mh-head-one", "mh-head-two", "mh-head-three"];
const MH_MODEL_CHANNELS = Array.from({ length: 6 }, (_, index) => `维 ${index + 1}`);
const MH_OUTPUT_CHANNELS = Array.from(
  { length: MH_HEAD_COUNT * MH_HEAD_WIDTH },
  (_, index) => `H${Math.floor(index / MH_HEAD_WIDTH) + 1}·${index % MH_HEAD_WIDTH + 1}`,
);

function MhSplitMatrix({
  data,
  symbol,
  accent,
}: {
  data: number[][];
  symbol: string;
  accent: string;
}) {
  return (
    <div className="mh-split-matrix">
      <div className="mh-split-name" style={{ color: accent }}>
        <Formula tex={symbol} />
        <span><Formula tex={String.raw`[4\times6]`} /></span>
      </div>
      <table>
        <thead>
          <tr>
            <th rowSpan={2}>token＼头</th>
            {MH_HEAD_CLASSES.map((className, headIndex) => (
              <th key={className} colSpan={MH_HEAD_WIDTH} className={className}>Head {headIndex + 1}</th>
            ))}
          </tr>
          <tr>
            {MH_HEAD_CLASSES.flatMap((className) => MH_CHANNELS.map((channel, index) => (
              <th key={`${className}-${index}`} className={className}>{channel}</th>
            )))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th>{tokenLabels[rowIndex]}</th>
              {row.map((value, columnIndex) => (
                <td key={columnIndex} className={MH_HEAD_CLASSES[Math.floor(columnIndex / MH_HEAD_WIDTH)]}>
                  {displayNumber(value, 1)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MhHeadInputs({ headIndex }: { headIndex: number }) {
  const headNumber = headIndex + 1;
  const head = multiHeadDemo.heads[headIndex];
  return (
    <article className={`mh-head-input-card mh-head-${headNumber}`}>
      <div className="mh-head-card-title">
        <span>HEAD {headNumber}</span>
        <b>取第 {headIndex * MH_HEAD_WIDTH + 1}～{headIndex * MH_HEAD_WIDTH + MH_HEAD_WIDTH} 列，四个 Token 全部保留</b>
      </div>
      <Formula block tex={String.raw`Q^{(${headNumber})},K^{(${headNumber})},V^{(${headNumber})}\in\mathbb{R}^{4\times2}`} />
      <div className="mh-head-input-grid">
        <FmsMatGrid
          data={head.Q}
          name={<Formula tex={`Q^{(${headNumber})}`} />}
          shape={<Formula tex={String.raw`[4\times2]`} />}
          pal={FMS_PAL.Q}
          rowLabels={tokenLabels}
          colLabels={MH_CHANNELS}
          cornerLabel="token＼维"
          digits={1}
        />
        <FmsMatGrid
          data={head.K}
          name={<Formula tex={`K^{(${headNumber})}`} />}
          shape={<Formula tex={String.raw`[4\times2]`} />}
          pal={FMS_PAL.K}
          rowLabels={tokenLabels}
          colLabels={MH_CHANNELS}
          cornerLabel="token＼维"
          digits={1}
        />
        <FmsMatGrid
          data={head.V}
          name={<Formula tex={`V^{(${headNumber})}`} />}
          shape={<Formula tex={String.raw`[4\times2]`} />}
          pal={FMS_PAL.V}
          rowLabels={tokenLabels}
          colLabels={MH_CHANNELS}
          cornerLabel="token＼维"
          digits={1}
        />
      </div>
      <div className="mh-token-slice">
        <span>以 Token 1 为例</span>
        <Formula tex={`q_1^{(${headNumber})}=${rowVectorTex(head.Q[0], 1)}`} />
        <Formula tex={`k_1^{(${headNumber})}=${rowVectorTex(head.K[0], 1)}`} />
        <Formula tex={`v_1^{(${headNumber})}=${rowVectorTex(head.V[0], 1)}`} />
      </div>
    </article>
  );
}

function MhHeadLane({ headIndex }: { headIndex: number }) {
  const headNumber = headIndex + 1;
  const head = multiHeadDemo.heads[headIndex];
  return (
    <article className={`mh-head-lane mh-head-${headNumber}`}>
      <div className="mh-lane-title">
        <span>HEAD {headNumber}</span>
        <div className="mh-lane-formulas">
          <Formula tex={String.raw`S^{(${headNumber})}=Q^{(${headNumber})}{K^{(${headNumber})}}^{\mathsf T}/\sqrt{2}`} />
          <i>→</i>
          <Formula tex={String.raw`A^{(${headNumber})}=\operatorname{softmax}(S^{(${headNumber})})`} />
          <i>→</i>
          <Formula tex={String.raw`H^{(${headNumber})}=A^{(${headNumber})}V^{(${headNumber})}`} />
        </div>
      </div>
      <div className="mh-lane-flow">
        <div className="mh-lane-step">
          <span>① 点积并缩放</span>
          <FmsMatGrid
            data={head.S}
            name={<Formula tex={`S^{(${headNumber})}`} />}
            shape={<Formula tex={String.raw`[4\times4]`} />}
            pal={FMS_PAL.S}
            rowLabels={tokenLabels}
            colLabels={tokenLabels}
            cornerLabel={<>行 <Formula tex="Q" />＼列 <Formula tex="K" /></>}
            heat
            digits={3}
          />
        </div>
        <div className="mh-lane-op"><span>逐行</span><b>softmax</b><i>→</i></div>
        <div className="mh-lane-step">
          <span>② 得到权重</span>
          <FmsMatGrid
            data={head.A}
            name={<Formula tex={`A^{(${headNumber})}`} />}
            shape={<Formula tex={String.raw`[4\times4]`} />}
            pal={FMS_PAL.A}
            rowLabels={tokenLabels}
            colLabels={tokenLabels}
            cornerLabel={<>行 <Formula tex="Q" />＼列 <Formula tex="K" /></>}
            heat
            digits={3}
          />
        </div>
        <div className="mh-lane-op"><span>乘本头</span><b><Formula tex={`V^{(${headNumber})}`} /></b><i>→</i></div>
        <div className="mh-lane-step">
          <span>③ 汇聚本头 Value</span>
          <FmsMatGrid
            data={head.H}
            name={<Formula tex={`H^{(${headNumber})}`} />}
            shape={<Formula tex={String.raw`[4\times2]`} />}
            pal={FMS_PAL.O}
            rowLabels={tokenLabels}
            colLabels={MH_CHANNELS}
            cornerLabel="token＼维"
            digits={3}
          />
        </div>
      </div>
      <div className="mh-first-row">
        <b>只追踪 Token 1：</b>
        <div className="mh-first-row-chain">
          <Formula tex={String.raw`q_1^{(${headNumber})}=${rowVectorTex(head.Q[0], 1)}`} /><i>→</i>
          <Formula tex={String.raw`S_{1,:}^{(${headNumber})}=${rowVectorTex(head.S[0])}`} /><i>→</i>
          <Formula tex={String.raw`A_{1,:}^{(${headNumber})}=${rowVectorTex(head.A[0])}`} /><i>→</i>
          <Formula tex={String.raw`h_1^{(${headNumber})}=${rowVectorTex(head.H[0])}`} />
        </div>
      </div>
    </article>
  );
}

function FigMultiHeadCalculation() {
  const firstTokenHeadOutputs = multiHeadDemo.heads
    .map((head) => rowVectorTex(head.H[0]))
    .join(",");
  const firstOutputTerms = multiHeadDemo.H[0]
    .map((value, index) => `${formatNumber(value)}\\times${formatNumber(multiHeadDemo.WO[index][0], 2)}`)
    .join("+");

  return (
    <div className="mh-walkthrough">
      <div className="mh-overview">
        <span>缩小版 Transformer 多头层 · Token 1～4</span>
        <b>输入 <Formula tex={String.raw`4\times6`} /> → 每头 <Formula tex={String.raw`4\times2`} /> → 拼接 <Formula tex={String.raw`4\times6`} /> → 输出 <Formula tex={String.raw`4\times6`} /></b>
        <Formula block tex={String.raw`L=4,\quad d_{\mathrm{model}}=6,\quad h=3,\quad d_k=d_v=d_{\mathrm{model}}/h=2`} />
        <p><Formula tex={String.raw`4\times2`} /> 单头算例独立展示 Attention 核心计算。原始 Transformer 与常见实现通常取 <Formula tex={String.raw`d_k=d_v=d_{\mathrm{model}}/h`} />，因而 <Formula tex={String.raw`h\,d_v=d_{\mathrm{model}}`} />；此处取 <Formula tex={String.raw`6=3\times2`} />，原始 Transformer 取 <Formula tex={String.raw`512=8\times64`} />。更一般地，<Formula tex="W^O" /> 负责把拼接宽度 <Formula tex={String.raw`h\,d_v`} /> 映射回 <Formula tex="d_{\mathrm{model}}" />。</p>
      </div>

      <div className="mh-shape-route" aria-label="多头注意力张量形状变化">
        <div><span>输入：4 个 Token，每个 6 维</span><b><Formula tex={String.raw`X\ [4\times6]`} /></b></div>
        <i>→</i>
        <div><span>一次融合生成三头投影</span><b><Formula tex={String.raw`Q_{\mathrm{all}},K_{\mathrm{all}},V_{\mathrm{all}}\ [4\times6]`} /></b></div>
        <i>→</i>
        <div><span>恢复头维，每头保留全部 Token</span><b><Formula tex={String.raw`3\ \mathrm{heads}\times[4\times2]`} /></b></div>
        <i>→</i>
        <div><span>每头一张完整权重图</span><b><Formula tex={String.raw`A^{(r)}\ [4\times4],\ H^{(r)}\ [4\times2]`} /></b></div>
        <i>→</i>
        <div><span>拼接并输出投影</span><b><Formula tex={String.raw`H\ [4\times6]\ \xrightarrow{\,W^O\,}\ Y\ [4\times6]`} /></b></div>
      </div>

      <section className="mh-stage">
        <header>
          <span>1</span>
          <div>
            <b>由同一个 <Formula tex="X" /> 一次融合生成三个头的 <Formula tex={String.raw`Q,K,V`} /></b>
            <p>第 <Formula tex="r" /> 个头的 <Formula tex={String.raw`W_r^Q,W_r^K,W_r^V\in\mathbb{R}^{6\times2}`} /> 把六维输入投影到二维。代码沿列合并三个头的参数，形成 <Formula tex={String.raw`W^Q,W^K,W^V\in\mathbb{R}^{6\times6}`} />，一次得到 <Formula tex={String.raw`Q_{\mathrm{all}},K_{\mathrm{all}},V_{\mathrm{all}}`} />。</p>
          </div>
        </header>
        <div className="mh-projection-flow">
          <div className="mh-projection-source">
            <FmsMatGrid
              data={multiHeadDemo.X}
              name={<Formula tex="X" />}
              shape={<Formula tex={String.raw`[4\times6]`} />}
              pal={{ c: "#a9b4dc", t: "rgba(169,180,220,0.08)" }}
              rowLabels={tokenLabels}
              colLabels={MH_MODEL_CHANNELS}
              cornerLabel="token＼维"
              digits={1}
            />
          </div>
          <div className="mh-projection-op">
            <Formula tex={String.raw`\times W^Q,W^K,W^V`} />
            <i>→</i>
          </div>
          <div className="mh-projection-results">
            <MhSplitMatrix data={multiHeadDemo.Q} symbol="Q=XW^Q" accent={FMS_PAL.Q.c} />
            <MhSplitMatrix data={multiHeadDemo.K} symbol="K=XW^K" accent={FMS_PAL.K.c} />
            <MhSplitMatrix data={multiHeadDemo.V} symbol="V=XW^V" accent={FMS_PAL.V.c} />
          </div>
        </div>
        <details className="mh-parameters">
          <summary>查看融合后的三个投影矩阵 <Formula tex={String.raw`W^Q,W^K,W^V`} /></summary>
          <div className="mh-parameter-grid">
            <FmsMatGrid data={multiHeadDemo.WQ} name={<Formula tex="W^Q" />} shape={<Formula tex={String.raw`[6\times6]`} />} pal={FMS_PAL.Q} digits={1} />
            <FmsMatGrid data={multiHeadDemo.WK} name={<Formula tex="W^K" />} shape={<Formula tex={String.raw`[6\times6]`} />} pal={FMS_PAL.K} digits={1} />
            <FmsMatGrid data={multiHeadDemo.WV} name={<Formula tex="W^V" />} shape={<Formula tex={String.raw`[6\times6]`} />} pal={FMS_PAL.V} digits={1} />
          </div>
          <p>数值算例采用稀疏固定矩阵以便逐项核算；真实模型中的投影矩阵由训练学习，通常是稠密矩阵。</p>
        </details>
      </section>

      <section className="mh-stage">
        <header>
          <span>2</span>
          <div>
            <b>把融合投影结果重排为三个头，不拆 Token</b>
            <p><Formula tex={String.raw`Q_{\mathrm{all}}=[Q^{(1)}\mid Q^{(2)}\mid Q^{(3)}]`} />，<Formula tex="K" />、<Formula tex="V" /> 同理。三个头划分的是投影结果的特征列，而不是原始 <Formula tex="X" /> 的 Token 行；每个头都保留 Token 1～4，并用二维向量表示每个 Token。</p>
          </div>
        </header>
        <div className="mh-head-inputs">
          {multiHeadDemo.heads.map((_, headIndex) => <MhHeadInputs key={headIndex} headIndex={headIndex} />)}
        </div>
      </section>

      <section className="mh-stage">
        <header>
          <span>3</span>
          <div>
            <b>三个头各算一套 <Formula tex={String.raw`QK^{\mathsf T}\rightarrow\operatorname{softmax}\rightarrow AV`} /></b>
            <p>每个头都由 <Formula tex={String.raw`Q^{(r)},K^{(r)},V^{(r)}\ [4\times2]`} /> 计算一张权重矩阵 <Formula tex={String.raw`A^{(r)}\ [4\times4]`} />，再得到本头输出 <Formula tex={String.raw`H^{(r)}=A^{(r)}V^{(r)}\ [4\times2]`} />。</p>
          </div>
        </header>
        <div className="mh-head-lanes">
          {multiHeadDemo.heads.map((_, headIndex) => <MhHeadLane key={headIndex} headIndex={headIndex} />)}
        </div>
      </section>

      <section className="mh-stage">
        <header>
          <span>4</span>
          <div>
            <b>拼接三个头，再用 <Formula tex="W^O" /> 融合头间特征并保持模型宽度</b>
            <p>三个 <Formula tex={String.raw`[4\times2]`} /> 沿特征列拼成 <Formula tex={String.raw`H\ [4\times6]`} />；<Formula tex="W^O" /> 对每个 Token 的六个跨头特征做线性组合，输出仍为 <Formula tex={String.raw`Y\ [4\times6]`} />。</p>
          </div>
        </header>
        <div className="mh-output-flow">
          <div className="mh-output-step">
            <span>拼接</span>
            <FmsMatGrid
              data={multiHeadDemo.H}
              name={<Formula tex={String.raw`H=[H^{(1)}\mid H^{(2)}\mid H^{(3)}]`} />}
              shape={<Formula tex={String.raw`[4\times6]`} />}
              pal={FMS_PAL.O}
              rowLabels={tokenLabels}
              colLabels={MH_OUTPUT_CHANNELS}
              cornerLabel="token＼头·维"
              digits={3}
            />
          </div>
          <div className="mh-output-op"><b>×</b><span>输出投影</span></div>
          <div className="mh-output-step">
            <span>跨头特征融合参数</span>
            <FmsMatGrid
              data={multiHeadDemo.WO}
              name={<Formula tex="W^O" />}
              shape={<Formula tex={String.raw`[6\times6]`} />}
              pal={FMS_PAL.V}
              rowLabels={MH_OUTPUT_CHANNELS}
              colLabels={MH_MODEL_CHANNELS}
              cornerLabel="头·维＼输出维"
              digits={2}
            />
          </div>
          <div className="mh-output-op"><b>=</b><span>保持模型维度</span></div>
          <div className="mh-output-step">
            <span>多头最终输出</span>
            <FmsMatGrid
              data={multiHeadDemo.Y}
              name={<Formula tex="Y=HW^O" />}
              shape={<Formula tex={String.raw`[4\times6]`} />}
              pal={FMS_PAL.S}
              rowLabels={tokenLabels}
              colLabels={MH_MODEL_CHANNELS}
              cornerLabel="token＼维"
              digits={3}
            />
          </div>
        </div>
        <div className="mh-output-row">
          <span>Token 1 的完整合并过程</span>
          <Formula block tex={String.raw`h_1=\operatorname{Concat}\!\left(${firstTokenHeadOutputs}\right)=${rowVectorTex(multiHeadDemo.H[0])}`} />
          <Formula block tex={String.raw`y_1=h_1W^O=${rowVectorTex(multiHeadDemo.Y[0])}`} />
        </div>
        <div className="mh-output-detail">
          <div>
            <b><Formula tex="W^O" />：融合各头特征，并映射回 <Formula tex="d_{\mathrm{model}}" /></b>
            <p>每个头的 <Formula tex={String.raw`A^{(r)}V^{(r)}`} /> 已经完成 Token 之间的信息汇聚；<Formula tex="W^O" /> 随后只沿每个 Token 的特征维工作，把不同头取回的内容进行可学习的线性融合。它不在 Token 维上再次混合信息，同时保证输出宽度适合残差连接和下一层。</p>
          </div>
          <Formula block tex={String.raw`y_{1,1}=\sum_{c=1}^{6}h_{1,c}W^O_{c,1}=${firstOutputTerms}=${formatNumber(multiHeadDemo.Y[0][0])}`} />
        </div>
        <div className="mh-output-purpose">
          <div>
            <b>为什么下一层仍然是 <Formula tex={String.raw`4\times6`} /></b>
            <p>本例中三个二维头拼接为 <Formula tex={String.raw`3\times2=6`} /> 维，<Formula tex={String.raw`W^O\in\mathbb{R}^{6\times6}`} /> 再输出六维。一般情况下，即使 <Formula tex={String.raw`h\,d_v`} /> 与 <Formula tex="d_{\mathrm{model}}" /> 不同，<Formula tex={String.raw`W^O\in\mathbb{R}^{(h d_v)\times d_{\mathrm{model}}}`} /> 也会映射回模型宽度；头数不会在层与层之间累乘维度。</p>
          </div>
          <Formula block tex={String.raw`\begin{aligned}X^{(\ell)}_{[4\times6]}&\xrightarrow{\;3\text{ 个头，各 }[4\times2]\;}H_{[4\times6]}\xrightarrow{\;W^O_{[6\times6]}\;}Y^{(\ell)}_{[4\times6]}\\X^{(\ell)}_{[4\times6]}+Y^{(\ell)}_{[4\times6]}&\xrightarrow{\;\text{归一化、FFN 与残差}\;}X^{(\ell+1)}_{[4\times6]}\end{aligned}`} />
        </div>
      </section>

      <div className="mh-implementation-note">
        <b>论文写法与代码写法是同一件事</b>
        <Formula block tex={String.raw`W^Q=[W_1^Q\mid W_2^Q\mid W_3^Q]`} />
        <Formula block tex={String.raw`Q_{\mathrm{all}}=XW^Q=[Q^{(1)}\mid Q^{(2)}\mid Q^{(3)}]`} />
        <p>论文为每个头分别写 <Formula tex={String.raw`W_i^Q,W_i^K,W_i^V`} />；工程代码把三组投影合并成一次矩阵乘法，再通过 <code>reshape + transpose</code> 恢复头维。代码并没有先算一次单头 Attention 再切开。</p>
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
      <text x={x + w / 2} y={y + h / 2 + (sub ? -5 : 4.5)} textAnchor="middle" fill={lc || "#eef3ff"} fontSize="14" fontWeight="700">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fill={sc || "#6e7aab"} fontSize="11">{sub}</text>}
    </g>
  );
}

function FigTransformer() {
  const Arrow = TfArrow;
  const Box = TfBox;
  return (
    <div className="fig">
      <svg className="transformer-figure" viewBox="0 0 940 630" width="940" role="img" aria-label="经典 Transformer Encoder-Decoder 结构图">
        <defs>
          <marker id="ah-t" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L8,4.5 L0,9 z" fill="#6e7aab" /></marker>
        </defs>
        <text x="225" y="26" textAnchor="middle" fill="#38bdf8" fontSize="16" fontWeight="700">Encoder（左）</text>
        <text x="695" y="26" textAnchor="middle" fill="#f472b6" fontSize="16" fontWeight="700">Decoder（右）</text>

        <Box x={155} y={48} w={140} h={38} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Input Embedding" lc="#a9b4dc" />
        <Arrow d="M225,86 V90" />
        <circle cx="225" cy="106" r="14" fill="#070b18" stroke="#a78bfa" />
        <text x="225" y="111" textAnchor="middle" fill="#a78bfa" fontSize="14">+</text>
        <rect x="35" y="84" width="115" height="44" rx="8" fill="#0c1430" stroke="rgba(167,139,250,0.45)" />
        <text x="92.5" y="101" textAnchor="middle" fill="#a78bfa" fontSize="11">Positional</text>
        <text x="92.5" y="119" textAnchor="middle" fill="#a78bfa" fontSize="11">Encoding</text>
        <Arrow d="M150,106 H207" />
        <Arrow d="M225,120 V170" />
        <rect x="80" y="138" width="290" height="280" rx="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeDasharray="5 5" />
        <g>
          <rect x="91" y="146" width="118" height="20" rx="7" fill="#0c1430" stroke="rgba(56,189,248,0.28)" />
          <text x="150" y="160" textAnchor="middle" fill="#7e8ac0" fontSize="11" fontWeight="700">N× Encoder Layer</text>
        </g>
        <Box x={120} y={174} w={210} h={54} fill="rgba(56,189,248,0.14)" stroke="#38bdf8" label="Multi-Head Self-Attention" sub="全局上下文建模" lc="#38bdf8" sc="#6e7aab" />
        <Box x={150} y={246} w={150} h={36} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={120} y={302} w={210} h={50} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" label="Feed-Forward Network" sub="两层 MLP（逐位置作用）" lc="#2dd4bf" sc="#6e7aab" />
        <Box x={150} y={370} w={150} h={36} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Arrow d="M225,228 V242" /><Arrow d="M225,282 V298" /><Arrow d="M225,352 V366" /><Arrow d="M225,406 V425" />
        <Arrow d="M225,170 H100 V264 H150" color="#f5b042" dash="4 3" />
        <Arrow d="M225,290 H100 V388 H150" color="#f5b042" dash="4 3" />
        <Box x={150} y={429} w={150} h={34} fill="rgba(244,114,182,0.14)" stroke="#f472b6" label="编码器输出 Memory" lc="#f472b6" />

        <Box x={625} y={48} w={140} h={38} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Target Embedding" sub="目标序列右移一位" lc="#a9b4dc" />
        <Arrow d="M695,86 V90" />
        <circle cx="695" cy="106" r="14" fill="#070b18" stroke="#a78bfa" />
        <text x="695" y="111" textAnchor="middle" fill="#a78bfa" fontSize="14">+</text>
        <rect x="790" y="84" width="115" height="44" rx="8" fill="#0c1430" stroke="rgba(167,139,250,0.45)" />
        <text x="847.5" y="101" textAnchor="middle" fill="#a78bfa" fontSize="11">Positional</text>
        <text x="847.5" y="119" textAnchor="middle" fill="#a78bfa" fontSize="11">Encoding</text>
        <Arrow d="M790,106 H713" />
        <Arrow d="M695,120 V170" />
        <rect x="540" y="138" width="310" height="370" rx="14" fill="none" stroke="rgba(255,255,255,0.08)" strokeDasharray="5 5" />
        <g>
          <rect x="551" y="146" width="118" height="20" rx="7" fill="#0c1430" stroke="rgba(244,114,182,0.28)" />
          <text x="610" y="160" textAnchor="middle" fill="#7e8ac0" fontSize="11" fontWeight="700">N× Decoder Layer</text>
        </g>
        <Box x={575} y={174} w={240} h={50} fill="rgba(245,176,66,0.14)" stroke="#f5b042" label="Masked Multi-Head Attention" sub="只能看过去（屏蔽未来位）" lc="#f5b042" sc="#6e7aab" />
        <Box x={620} y={240} w={150} h={34} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={575} y={292} w={240} h={50} fill="rgba(56,189,248,0.14)" stroke="#38bdf8" label="Cross-Attention" sub="Q: Decoder · K,V: Encoder Memory" lc="#38bdf8" sc="#6e7aab" />
        <Box x={620} y={358} w={150} h={34} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Box x={575} y={410} w={240} h={44} fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" label="Feed-Forward Network" lc="#2dd4bf" />
        <Box x={620} y={466} w={150} h={32} fill="#0c1430" stroke="rgba(255,255,255,0.08)" label="Add &amp; Norm" lc="#a9b4dc" />
        <Arrow d="M695,224 V236" /><Arrow d="M695,274 V288" /><Arrow d="M695,342 V354" /><Arrow d="M695,392 V406" /><Arrow d="M695,454 V462" />
        <Arrow d="M695,170 H555 V257 H620" color="#f5b042" dash="4 3" />
        <Arrow d="M695,283 H555 V375 H620" color="#f5b042" dash="4 3" />
        <Arrow d="M695,401 H555 V482 H620" color="#f5b042" dash="4 3" />
        <Arrow d="M300,446 H455 V317 H571" color="#a78bfa" dash="4 3" />
        <rect x="354" y="294" width="82" height="20" rx="7" fill="#0c1430" stroke="rgba(167,139,250,0.34)" />
        <text x="395" y="308" textAnchor="middle" fill="#a78bfa" fontSize="10.5" fontWeight="700">提供 K、V</text>
        <Arrow d="M695,498 V513" />
        <Box x={585} y={517} w={220} h={46} fill="rgba(167,139,250,0.14)" stroke="#a78bfa" label="Linear + Softmax" sub="输出下一 Token 的概率" lc="#a78bfa" sc="#7e8ac0" />

        <g transform="translate(80,590)">
          <rect x="0" y="0" width="14" height="14" rx="3" fill="rgba(56,189,248,0.14)" stroke="#38bdf8" /><text x="20" y="12" fill="#6e7aab" fontSize="12">Attention</text>
          <rect x="110" y="0" width="14" height="14" rx="3" fill="rgba(45,212,191,0.14)" stroke="#2dd4bf" /><text x="130" y="12" fill="#6e7aab" fontSize="12">FFN</text>
          <rect x="190" y="0" width="14" height="14" rx="3" fill="#0c1430" stroke="rgba(255,255,255,0.08)" /><text x="210" y="12" fill="#6e7aab" fontSize="12">Add&amp;Norm</text>
          <rect x="310" y="0" width="14" height="14" rx="3" fill="#070b18" stroke="#a78bfa" /><text x="330" y="12" fill="#6e7aab" fontSize="12">位置编码</text>
          <line x1="430" y1="7" x2="447" y2="7" stroke="#f5b042" strokeWidth="1.5" strokeDasharray="4 3" /><text x="457" y="12" fill="#6e7aab" fontSize="12">残差连接</text>
        </g>
      </svg>
      <div className="fig-cap">图 · 论文 Figure 1 重绘 — 三处 Attention 共享缩放点积核心，但 Query / Key / Value 的来源与因果 Mask 不同</div>
    </div>
  );
}

function PositionEncodingFlow() {
  const positionEmbedding = [
    [0.05, 0.1],
    [0.15, 0.2],
    [0.25, 0.3],
    [0.35, 0.4],
  ];
  const attentionInput = attentionDemo.X;
  const contentEmbedding = attentionInput.map((row, rowIndex) =>
    row.map((value, columnIndex) => value - positionEmbedding[rowIndex][columnIndex]),
  );
  const rows = ["Token 1", "Token 2", "Token 3", "Token 4"];
  const cols = ["维 1", "维 2"];
  const sinusoidalRows = Array.from({ length: 4 }, (_, position) => [
    Math.sin(position),
    Math.cos(position),
    Math.sin(position / 100),
    Math.cos(position / 100),
  ]);

  return (
    <div className="position-demo">
      <div className="position-demo-head">
        <span>进入第一层 Attention 之前</span>
        <b>内容矩阵 <Formula tex="E_{\mathrm{seq}}" /> + 位置矩阵 <Formula tex="P_{\mathrm{seq}}" /> = 输入矩阵 <Formula tex="X" /></b>
        <p>没有位置编码或位置偏置时，双向 Self-Attention 不会自行获得先后顺序。这里用 <Formula tex={String.raw`u_i=E_{\mathrm{tok}}[\operatorname{tokenId}_i]`} /> 表示查表结果，并把原始 Transformer 缩放后的内容向量记为 <Formula tex={String.raw`e_i=\sqrt{d_{\mathrm{model}}}\,u_i`} />；它与同维位置向量 <Formula tex="p_i" /> 逐元素相加得到 <Formula tex="x_i" />。</p>
      </div>

      <div className="position-matrix-flow" role="img" aria-label="四个 Token 的内容向量逐行加上位置向量，得到 Attention 输入矩阵 X">
        <div className="position-matrix-stage">
          <span>① Token 内容</span>
          <FmsMatGrid data={contentEmbedding} name={<Formula tex="E_{\mathrm{seq}}" />} shape={<Formula tex={String.raw`[4\times2]`} />} pal={{ c: "#a9b4dc", t: "rgba(169,180,220,0.08)" }} rowLabels={rows} colLabels={cols} cornerLabel="位置＼维" digits={2} />
          <small>四个已完成输入缩放的 <Formula tex="e_i" /> 按行堆叠</small>
        </div>
        <div className="position-matrix-op"><b>+</b><span>逐行、逐维相加</span></div>
        <div className="position-matrix-stage">
          <span>② 位置信息</span>
          <FmsMatGrid data={positionEmbedding} name={<Formula tex="P_{\mathrm{seq}}" />} shape={<Formula tex={String.raw`[4\times2]`} />} pal={FMS_PAL.K} rowLabels={rows} colLabels={cols} cornerLabel="位置＼维" digits={2} />
          <small>第 <Formula tex="i" /> 行就是 <Formula tex="p_i" />；此处使用二维固定数值</small>
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
        <Formula block tex={String.raw`\underbrace{${rowVectorTex(contentEmbedding[0], 2)}}_{e_1\;\text{内容}}+\underbrace{${rowVectorTex(positionEmbedding[0], 2)}}_{p_1\;\text{位置}}=\underbrace{${rowVectorTex(attentionInput[0], 2)}}_{x_1\;\text{输入}}`} />
        <p><Formula tex="x_1" /> 随后参与投影：<Formula tex={String.raw`q_1=x_1W^Q=${rowVectorTex(attentionDemo.Q[0], 2)}`} />。因此位置编码位于 <Formula tex={String.raw`Q,\ K,\ V`} /> 投影之前。</p>
      </div>

      <div className="position-addition-proof">
        <div className="position-addition-head">
          <span>为什么使用相加，而不是拼接</span>
          <b>拼接输入经分块线性变换可写成内容项与位置项之和</b>
          <p>内容和位置索引拼接后，经过某种特定结构的线性变换，可以写成内容项与位置项之和。行向量记法中，内容向量 <Formula tex={String.raw`e_i\in\mathbb{R}^{1\times d_{\mathrm{model}}}`} /> 与位置 one-hot <Formula tex={String.raw`r_i\in\mathbb{R}^{1\times L}`} /> 拼接后，得到 <Formula tex={String.raw`[e_i\mid r_i]\in\mathbb{R}^{1\times(d_{\mathrm{model}}+L)}`} />。</p>
        </div>

        <div className="position-proof-terms" aria-label="内容行向量、位置 one-hot 行向量与拼接行向量">
          <article>
            <span>内容向量</span>
            <Formula block tex={String.raw`e_3=\begin{bmatrix}0.55&0.60\end{bmatrix}`} />
            <small>Token 3 的内容，shape 为 <Formula tex={String.raw`1\times2`} /></small>
          </article>
          <article className="address">
            <span>位置索引 · one-hot</span>
            <Formula block tex={String.raw`r_3=\begin{bmatrix}0&0&1&0\end{bmatrix}`} />
            <small>第 3 项为 1，表示位置 3</small>
          </article>
          <article className="combined">
            <span>沿特征维拼接</span>
            <Formula block tex={String.raw`[e_3\mid r_3]=\left[\begin{array}{cc|cccc}0.55&0.60&0&0&1&0\end{array}\right]`} />
            <small>拼接后 shape 为 <Formula tex={String.raw`1\times6`} /></small>
          </article>
        </div>

        <div className="position-linear-equation">
          <span>分块线性变换</span>
          <Formula block tex={String.raw`\begin{aligned}
          \underbrace{[e_i\mid r_i]}_{1\times(d_{\mathrm{model}}+L)}
          \underbrace{\vphantom{[e_i\mid r_i]}W}_{(d_{\mathrm{model}}+L)\times d_{\mathrm{model}}}
          &=[e_i\mid r_i]\underbrace{\begin{bmatrix}W^e\\W^p\end{bmatrix}}_{W\text{ 按输入分块}}\\
          &=\underbrace{e_iW^e}_{\text{变换后的内容}}+\underbrace{r_iW^p}_{p_i}\\
          &=e_iW^e+p_i
          \end{aligned}`} />
          <div className="position-linear-reading">
            <div><b>内容分块</b><span><Formula tex={String.raw`W^e\in\mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{model}}}`} /> 负责变换内容，因此一般结果保留为 <Formula tex="e_iW^e" />。</span></div>
            <div><b>位置分块</b><span>令 <Formula tex={String.raw`W^p=P_{\mathrm{seq}}=\begin{bmatrix}p_1\\p_2\\\vdots\\p_L\end{bmatrix}`} />，one-hot 只选中第 <Formula tex="i" /> 行，所以 <Formula tex={String.raw`r_iW^p=p_i`} />。</span></div>
          </div>
          <p className="position-general-result">输出的一般形式为 <Formula tex="e_iW^e+p_i" />；取 <Formula tex={String.raw`W^e=I_{d_{\mathrm{model}}}`} /> 时化为 <Formula tex="e_i+p_i" />。</p>
        </div>

        <div className="position-proof-calculation" role="img" aria-label="取内容分块为二维单位矩阵后，Token 3 的拼接行向量经过分块线性变换得到内容与位置之和">
          <span>单位矩阵数值例 · <Formula tex={String.raw`W^e=I_2`} /></span>
          <Formula block tex={String.raw`\begin{aligned}
          &\underbrace{\left[\begin{array}{cc|cccc}0.55&0.60&0&0&1&0\end{array}\right]}_{[e_3\mid r_3]\;(1\times6)}
          \underbrace{\left[\begin{array}{cc}1&0\\0&1\\\hline0.05&0.10\\0.15&0.20\\0.25&0.30\\0.35&0.40\end{array}\right]}_{W=[I_2;W^p]\;(6\times2)}\\[4pt]
          ={}&0.55\begin{bmatrix}1&0\end{bmatrix}+0.60\begin{bmatrix}0&1\end{bmatrix}\\
          &+0\begin{bmatrix}0.05&0.10\end{bmatrix}+0\begin{bmatrix}0.15&0.20\end{bmatrix}+1\begin{bmatrix}0.25&0.30\end{bmatrix}+0\begin{bmatrix}0.35&0.40\end{bmatrix}\\[4pt]
          ={}&\begin{bmatrix}0.55&0.60\end{bmatrix}+\begin{bmatrix}0.25&0.30\end{bmatrix}
          =\begin{bmatrix}0.80&0.90\end{bmatrix}=x_3
          \end{aligned}`} />
          <div className="position-proof-reading">
            <div><b>内容块 <Formula tex="W^e" /></b><span>取 <Formula tex={String.raw`W^e=I_2`} />，因此 <Formula tex={String.raw`e_3I_2=e_3`} />，内容数值保持不变。</span></div>
            <div><b>位置块 <Formula tex="W^p" /></b><span><Formula tex={String.raw`r_3=[0,0,1,0]`} /> 选中 <Formula tex="W^p" /> 第 3 行，得到 <Formula tex="p_3" />。</span></div>
            <div><b>输出</b><span><Formula tex={String.raw`W^e=I_2`} /> 时，<Formula tex="e_3W^e+p_3" /> 化为 <Formula tex="e_3+p_3=x_3" />。</span></div>
          </div>
        </div>

        <div className="position-proof-boundary">
          <b>与 Transformer 输入相加的关系</b>
          <p><Formula tex={String.raw`[e_i\mid r_i]W=e_iW^e+p_i`} /> 描述拼接输入经过分块线性变换后的结果；取 <Formula tex={String.raw`W^e=I_{d_{\mathrm{model}}}`} /> 即得到 <Formula tex="e_i+p_i" />。原始 Transformer 直接计算 <Formula tex={String.raw`e_i=\sqrt{d_{\mathrm{model}}}\,E_{\mathrm{tok}}[\operatorname{tokenId}_i]`} />，再与位置编码相加，不显式构造 one-hot、拼接向量或矩阵 <Formula tex="W" />。</p>
        </div>
      </div>
      <div className="position-sine-proof">
        <div className="position-sine-head">
          <span>原始 Transformer · 固定正弦 / 余弦位置编码</span>
          <b>sin 不要求单调：位置由整行多频率向量表示</b>
          <p>单一 <Formula tex="\sin(\mathrm{pos})" /> 坐标具有周期性。成对的 sin/cos 与多组不同频率共同构成位置向量；模型使用整行向量区分位置，而不依赖某一维单调递增。</p>
        </div>

        <div className="position-sine-overview">
          <article className="position-sine-formula">
            <span>生成规则</span>
            <Formula block tex={String.raw`\begin{aligned}\mathrm{PE}_{(\mathrm{pos},2f)}&=\sin\!\left(\frac{\mathrm{pos}}{10000^{2f/d_{\mathrm{model}}}}\right)\\\mathrm{PE}_{(\mathrm{pos},2f+1)}&=\cos\!\left(\frac{\mathrm{pos}}{10000^{2f/d_{\mathrm{model}}}}\right)\end{aligned}`} />
            <p><Formula tex="\mathrm{pos}" /> 是 Token 位置，<Formula tex="f" /> 是频率组。偶数 <Formula tex="d_{\mathrm{model}}" /> 包含 <Formula tex={String.raw`d_{\mathrm{model}}/2`} /> 组 sin/cos；取 <Formula tex={String.raw`d_{\mathrm{model}}=4`} /> 时共有两组，分母分别是 1 和 100。</p>
          </article>
          <article className="position-sine-matrix">
            <span>位置 0～3 的完整编码</span>
            <FmsMatGrid data={sinusoidalRows} name={<Formula tex="\mathrm{PE}" />} shape={<Formula tex={String.raw`[4\times4]`} />} pal={FMS_PAL.K} rowLabels={["pos 0", "pos 1", "pos 2", "pos 3"]} colLabels={["高频 sin", "高频 cos", "低频 sin", "低频 cos"]} cornerLabel="位置＼频率" digits={4} />
            <p>“高频 / 低频”描述的是数值随 <Formula tex="\mathrm{pos}" /> 变化的快慢，不是计算速度。四个维度合起来，为每个位置形成不同的多频率向量。</p>
          </article>
        </div>

        <div className="position-sine-reasons">
          <article><b>不负责排序</b><span>位置编码不是把位置压成一个越来越大的标量，而是给每个位置一组可比较的特征。</span></article>
          <article><b>多尺度变化</b><span>高频维度区分邻近位置，低频维度缓慢变化，帮助表示更长跨度；不同周期组合降低短范围内的混淆。</span></article>
          <article><b>无需训练查表</b><span>任意位置都可直接代入公式计算，因此编码函数能够生成训练长度之外的位置；这不等同于模型必然具备长度外推能力。</span></article>
        </div>

        <div className="position-shift-proof">
          <span>最关键的性质：固定相对距离对应固定旋转</span>
          <Formula block tex={String.raw`\begin{bmatrix}\sin((\mathrm{pos}+k)\omega)&\cos((\mathrm{pos}+k)\omega)\end{bmatrix}=\begin{bmatrix}\sin(\mathrm{pos}\,\omega)&\cos(\mathrm{pos}\,\omega)\end{bmatrix}\begin{bmatrix}\cos(k\omega)&-\sin(k\omega)\\\sin(k\omega)&\cos(k\omega)\end{bmatrix}`} />
          <p>对同一频率 <Formula tex="\omega" />，从位置 <Formula tex="\mathrm{pos}" /> 移动 <Formula tex="k" /> 步，相当于乘一个只由距离 <Formula tex="k" /> 决定的二维旋转矩阵。sin/cos 对使固定相对位移可表示为只依赖 <Formula tex="k" /> 的线性旋转；单调性不是这一设计的目标。</p>
        </div>
      </div>

      <div className="position-demo-note">二维数值表从已准备好的内容向量 <Formula tex="e_i" /> 开始，不再重复展示 Embedding 查表与标量缩放；后面的 Attention 数值演示同样从 <Formula tex="X" /> 开始。输入端绝对位置编码在 <Formula tex={String.raw`Q,\ K,\ V`} /> 投影前与内容向量逐元素相加。</div>
    </div>
  );
}


function FigFlashCompare() {
  return (
    <div className="fig flash-memory-compare">
      <div className="flash-memory-lanes">
        <article className="flash-memory-lane normal">
          <header><b>普通分步前向</b><span>完整中间矩阵进入 HBM</span></header>
          <div className="flash-io-flow">
            <span><Formula tex={String.raw`QK^{\mathsf T}`} /></span><b>→</b>
            <span><Formula tex={String.raw`S_{L\times L}`} /></span><em>写 / 读 HBM</em>
            <span><Formula tex={String.raw`\operatorname{softmax}(S)`} /></span><b>→</b>
            <span><Formula tex={String.raw`A_{L\times L}`} /></span><em>写 / 读 HBM</em>
            <span><Formula tex="AV" /></span>
          </div>
          <div className="flash-memory-formula"><span>若分别保留两张中间矩阵</span><Formula tex={String.raw`S+A=2L^2`} /><span>个元素</span></div>
        </article>

        <article className="flash-memory-lane flash">
          <header><b>FlashAttention 前向</b><span>一个 tile 算完立即消费</span></header>
          <div className="flash-io-flow">
            <span><Formula tex={String.raw`Q_{\mathrm{blk}}`} /></span><b>+</b>
            <span><Formula tex={String.raw`K_t,V_t`} /></span><em>从 HBM 载入 tile</em>
            <span><Formula tex={String.raw`S^{(t)}`} /> · 片上</span><b>→</b>
            <span>更新 <Formula tex={String.raw`m,\ell,o`} /></span><em>丢弃当前 score tile</em>
            <span>循环下一个 tile</span><b>→</b>
            <span>写回 <Formula tex="O" /> 与行统计</span>
          </div>
          <Formula block tex={String.raw`\text{HBM 不保存完整 }S,A`} />
        </article>
      </div>

      <div className="flash-memory-number">
        <div>
          <span>例：<Formula tex={String.raw`L=4096`} />，FP16 / BF16 每个元素 2 Byte</span>
          <Formula block tex={String.raw`L^2\times2\ \mathrm{Byte}=4096^2\times2\ \mathrm{Byte}=32\ \mathrm{MiB}`} />
        </div>
        <strong>每张约 32 MiB；两张合计约 64 MiB / batch / head</strong>
        <small>FlashAttention 不把这两张 <Formula tex={String.raw`L\times L`} /> 前向中间矩阵写入 HBM；额外保留输出 <Formula tex="O" /> 与每行归一化统计，其规模随 <Formula tex="L" /> 线性增长。</small>
      </div>

      <div className="flash-speed-reasons">
        <article><b>少写</b><span>不把完整 <Formula tex={String.raw`S,A`} /> 写入 HBM</span></article>
        <article><b>少读</b><span>softmax 与 <Formula tex="AV" /> 直接消费片上 tile</span></article>
        <article><b>融合</b><span>打分、归一化、汇聚在同一执行链完成</span></article>
        <article><b>计算量不变</b><span>仍计算全部分数：O(L²d)</span></article>
      </div>

      <div className="flash-memory-conclusion">
        <Formula tex={String.raw`\text{更少的 HBM 往返}`} />
        <b>→</b>
        <Formula tex={String.raw`\text{更低的临时显存占用}`} />
        <b>+</b>
        <Formula tex={String.raw`\text{更短的数据等待时间}`} />
      </div>
    </div>
  );
}

function FigFlashGpuMap() {
  return (
    <div className="fig flash-gpu-map">
      <div className="flash-gpu-parallel">
        <strong>FlashAttention-2 风格映射 · 一个 block 处理一个 Query 块</strong>
        <div>
          <span>Block 0 · <Formula tex={String.raw`Q_{\mathrm{blk},0}`} /></span>
          <span>Block 1 · <Formula tex={String.raw`Q_{\mathrm{blk},1}`} /></span>
          <span>Block 2 · <Formula tex={String.raw`Q_{\mathrm{blk},2}`} /></span>
          <span>…</span>
        </div>
      </div>

      <div className="flash-gpu-flow" aria-label="一个 thread block 扫描所有 Key Value tile 并写回一个 Query 块输出的流程">
        <div className="flash-gpu-hbm source">
          <b>HBM · 全局显存</b>
          <div><span><Formula tex={String.raw`Q_{\mathrm{blk}}`} /></span><span><Formula tex={String.raw`K_1,V_1`} /></span><span><Formula tex={String.raw`K_2,V_2`} /></span><span>…</span><span><Formula tex={String.raw`K_T,V_T`} /></span></div>
        </div>

        <div className="flash-gpu-transfer"><b>↓</b><span><Formula tex={String.raw`Q_{\mathrm{blk}}`} /> 只加载一次；每轮加载一组 <Formula tex={String.raw`K_t,V_t`} /></span></div>

        <div className="flash-gpu-kernel">
          <header>
            <b>一个 thread block 在 SM 上处理一个 <Formula tex={String.raw`Q_{\mathrm{blk}}`} /></b>
            <span>片上存储容量有限，所以沿 Key / Value 方向逐 tile 扫描</span>
          </header>

          <div className="flash-gpu-init">
            <span><i>1</i><b>准备 Query 块与行状态</b></span>
            <div className="flash-gpu-init-state"><span>片上复用：<Formula tex={String.raw`Q_{\mathrm{blk}}`} /></span><span>registers 行状态：<Formula tex={String.raw`m=-\infty,\ \ell=0,\ o=0`} /></span></div>
          </div>

          <div className="flash-gpu-loop">
            <div className="flash-gpu-loop-title"><b>循环 <Formula tex={String.raw`t=1,2,\ldots,T`} /></b><span>每轮只处理一个 Key / Value tile</span></div>
            <div className="flash-gpu-loop-steps">
              <article><i>2</i><b>载入当前 tile</b><Formula tex={String.raw`K_t,V_t\ \to\ \text{shared memory}`} /></article>
              <em>→</em>
              <article><i>3</i><b>计算分数 tile</b><Formula tex={String.raw`S_t=Q_{\mathrm{blk}}K_t^{\mathsf T}/\sqrt{d_k}+M_t`} /></article>
              <em>→</em>
              <article><i>4</i><b>在线更新行状态</b><Formula tex={String.raw`m,\ell,o\ \leftarrow\ \operatorname{update}(S_t,V_t)`} /></article>
              <em>→</em>
              <article><i>5</i><b>释放临时分数</b><span>丢弃 <Formula tex={String.raw`S_t`} />，不写回 HBM</span></article>
            </div>
            <div className="flash-gpu-loop-back"><span>仍有下一个 tile</span><b>↺</b><Formula tex={String.raw`t\leftarrow t+1`} /></div>
          </div>

          <div className="flash-gpu-finish">
            <article><i>6</i><b>最后一个 tile 完成</b><span>此时 <Formula tex={String.raw`m,\ell,o`} /> 已汇总所有 Key / Value</span></article>
            <em>→</em>
            <article><i>7</i><b>统一归一化</b><Formula tex={String.raw`O_{\mathrm{blk}}[i,:]=o[i,:]/\ell_i`} /></article>
          </div>
        </div>

        <div className="flash-gpu-transfer output"><span>只写回最终输出与行统计</span><b>↓</b></div>

        <div className="flash-gpu-hbm output">
          <b>HBM · 写回</b>
          <div><span><Formula tex={String.raw`O_{\mathrm{blk}}`} /></span><span>每行归一化统计</span></div>
        </div>
      </div>
      <div className="fig-cap">Grid 包含多个 thread block；上图展开其中一个 block：Query 块留在片上，Key / Value tile 依次流过，全部扫描完成后才归一化并写回输出。</div>
    </div>
  );
}

function FigFlashNumericTiles() {
  return (
    <div className="fig flash-exact-demo">
      <div className="flash-example-input">
        <span>同一输入 · 固定一个 Query 行</span>
        <Formula block tex={String.raw`S_1=\left[\begin{array}{cc|cc}\ln1&\ln2&\ln3&\ln4\end{array}\right],\qquad V=\begin{bmatrix}10\\20\\30\\40\end{bmatrix}`} />
        <small>竖线表示切成两个 tile；选择 <Formula tex={String.raw`\ln1,\ldots,\ln4`} />，是因为 <Formula tex={String.raw`e^{\ln n}=n`} />，可以把每一步算成整洁的分数。<Formula tex={String.raw`p_t`} /> 表示当前 tile 的未归一化指数权重。</small>
      </div>

      <div className="flash-exact-columns">
        <article className="flash-exact-lane normal">
          <header><span>普通 Softmax</span><b>一次看完四个位置</b></header>
          <div className="flash-calc-step">
            <i>1 · 取指数</i>
            <Formula block tex={String.raw`e^{S_1}=[1,\ 2,\ 3,\ 4]`} />
          </div>
          <div className="flash-calc-step">
            <i>2 · 求全局分母</i>
            <Formula block tex={String.raw`Z=1+2+3+4=10`} />
          </div>
          <div className="flash-calc-step">
            <i>3 · 得到 Softmax 权重</i>
            <Formula block tex={String.raw`A_1=e^{S_1}/Z=[0.1,\ 0.2,\ 0.3,\ 0.4]`} />
          </div>
          <div className="flash-calc-step final">
            <i>4 · 权重乘 Value</i>
            <Formula block tex={String.raw`N=1(10)+2(20)+3(30)+4(40)=300`} />
            <Formula block tex={String.raw`O_1=N/Z=300/10=30`} />
          </div>
        </article>

        <article className="flash-exact-lane flash">
          <header><span>FlashAttention</span><b>每次只看两个位置</b></header>
          <div className="flash-pass-order">
            <span><small>初始状态</small><Formula tex={String.raw`m_0=-\infty,\ \ell_0=0,\ o_0=0`} /></span>
            <b>→</b>
            <span><small>扫描 Tile 1</small><Formula tex={String.raw`m_1,\ell_1,o_1`} /></span>
            <b>→</b>
            <span><small>扫描 Tile 2</small><Formula tex={String.raw`m_2,\ell_2,o_2`} /></span>
            <b>→</b>
            <span><small>全部 tile 完成</small><Formula tex={String.raw`O_1=o_2/\ell_2`} /></span>
          </div>
          <div className="flash-tile-round first">
            <i>Tile 1 · <Formula tex={String.raw`[\ln1,\ln2]`} /></i>
            <div className="flash-update-step">
              <span>① 当前最大值</span>
              <div><Formula block tex={String.raw`m_1=\max(\ln1,\ln2)=\ln2`} /></div>
            </div>
            <div className="flash-update-step">
              <span>② tile 指数权重</span>
              <div><Formula block tex={String.raw`p_1=e^{[\ln1,\ln2]-\ln2}=[\tfrac12,\ 1]`} /></div>
            </div>
            <div className="flash-update-step">
              <span>③ 累计分母</span>
              <div><Formula block tex={String.raw`\ell_1=\tfrac12+1=\tfrac32`} /></div>
            </div>
            <div className="flash-update-step">
              <span>④ 累计分子</span>
              <div><Formula block tex={String.raw`o_1=[\tfrac12,1]\begin{bmatrix}10\\20\end{bmatrix}=25`} /></div>
            </div>
          </div>

          <div className="flash-tile-round second">
            <i>Tile 2 · <Formula tex={String.raw`[\ln3,\ln4]`} /></i>
            <strong className="flash-update-rule">旧结果先乘换算比例 <Formula tex="\rho" />，再分别与新 tile 的分子贡献、分母贡献相加</strong>
            <div className="flash-update-step">
              <span>① 更新最大值与比例</span>
              <div>
                <Formula block tex={String.raw`m_2=\max(\ln2,\ln3,\ln4)=\ln4`} />
                <Formula block tex={String.raw`\rho=e^{m_1-m_2}=e^{\ln2-\ln4}=\tfrac12`} />
              </div>
            </div>
            <div className="flash-update-step">
              <span>② 旧结果先乘 <Formula tex="\rho" /></span>
              <div>
                <Formula block tex={String.raw`\ell_{\mathrm{old}}=\rho\ell_1=\tfrac12\times\tfrac32=\tfrac34`} />
                <Formula block tex={String.raw`o_{\mathrm{old}}=\rho o_1=\tfrac12\times25=\tfrac{25}{2}`} />
              </div>
            </div>
            <div className="flash-update-step">
              <span>③ 新 tile 的贡献</span>
              <div>
                <Formula block tex={String.raw`p_2=e^{[\ln3,\ln4]-\ln4}=[\tfrac34,\ 1]`} />
                <Formula block tex={String.raw`\Delta\ell_2=\tfrac34+1=\tfrac74`} />
                <Formula block tex={String.raw`\Delta o_2=[\tfrac34,1]\begin{bmatrix}30\\40\end{bmatrix}=\tfrac{125}{2}`} />
              </div>
            </div>
            <div className="flash-update-step combine">
              <span>④ 旧结果 + 新 tile</span>
              <div>
                <Formula block tex={String.raw`\ell_2=\ell_{\mathrm{old}}+\Delta\ell_2=\tfrac34+\tfrac74=\tfrac52`} />
                <Formula block tex={String.raw`o_2=o_{\mathrm{old}}+\Delta o_2=\tfrac{25}{2}+\tfrac{125}{2}=75`} />
              </div>
            </div>
          </div>

          <div className="flash-calc-step final">
            <i>收尾 · 全部 tile 扫描完成后统一归一化</i>
            <Formula block tex={String.raw`O_1=o_2/\ell_2=75/(\tfrac52)=30`} />
          </div>
        </article>
      </div>

      <div className="flash-exact-result">
        <span>实数精确计算下，两种路径对应的全局权重完全相同</span>
        <div className="flash-equivalence-chain">
          <div className="flash-equivalence-step">
            <i>相同的整体缩放</i>
            <Formula block tex={String.raw`e^{S_1-\ln4}=\tfrac14e^{S_1}`} />
          </div>
          <div className="flash-equivalence-step">
            <i>分母与分子同时缩小四倍</i>
            <Formula block tex={String.raw`\ell_2=Z/4=10/4=\tfrac52,\qquad o_2=N/4=300/4=75`} />
          </div>
          <div className="flash-equivalence-step">
            <i>归一化后得到相同权重</i>
            <Formula block tex={String.raw`A_1^{\mathrm{Flash}}=\frac1{\ell_2}[\tfrac14,\ \tfrac12,\ \tfrac34,\ 1]=[0.1,\ 0.2,\ 0.3,\ 0.4]=A_1^{\mathrm{普通}}`} />
          </div>
          <div className="flash-equivalence-step final">
            <i>最终输出相同</i>
            <Formula block tex={String.raw`O_1^{\mathrm{Flash}}=o_2/\ell_2=75/(\tfrac52)=30=O_1^{\mathrm{普通}}`} />
          </div>
        </div>
      </div>

      <div className="flash-exact-key">
        <Formula tex={String.raw`\rho=e^{m_{\mathrm{old}}-m_{\mathrm{new}}}`} />
        <span>把旧 tile 的分子、分母换算到新最大值的尺度</span>
        <b>→</b>
        <span>加入新 tile</span>
        <b>→</b>
        <span>最后统一归一化</span>
      </div>
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

  const attn = {
    scaled: attentionDemo.S[qIdx],
    weights: attentionDemo.A[qIdx],
    output: attentionDemo.O[qIdx],
  };

  const [mr, mc] = selCell;

  const navItems = [
    ["s0", "为什么重要"],
    ["s1", "矩阵乘法"],
    ["s2", "向量级 Attention"],
    ["s3", "矩阵级 Attention"],
    ["s4", "缩放与 Mask"],
    ["s5", "多头注意力"],
    ["s6", "FlashAttention"],
    ["s7", "代码实现"],
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
              <div className="shape-note"><Formula tex="B" />：批大小；<Formula tex="h" />：头数；<Formula tex={String.raw`L_q,L_k`} />：Query / Key 序列长度；<Formula tex={String.raw`d_k,d_v`} />：每个头的 Key / Value 维度。<Formula tex="O" /> 表示各头尚未拼接的输出；单头数值算例取 <Formula tex={String.raw`B=h=1`} />，因此省略这两个长度为 1 的轴。</div>
            </div>

            <div className="grid2" style={{ marginTop: 28 }}>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>这个算子为什么重要</h3>
                <p className="t3">它是 GPT、LLaMA 等语言模型以及许多视觉生成模型的核心模块。掌握 Attention，才能继续理解 Transformer 与现代生成模型的主要计算路径。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>算子计算结构</h3>
                <p className="t3">Attention 的核心就是<b style={{ color: "#eef3ff" }}>两次矩阵乘法（<Formula tex={String.raw`QK^{\mathsf T}`} /> 计算相关分数、<Formula tex="AV" /> 加权汇聚）+ 一个 softmax</b>；完整多头还包括 <Formula tex={String.raw`Q,\ K,\ V`} /> 与 <Formula tex="W^O" /> 四个投影，共六次矩阵乘法；自注意力把 QKV 合并为一次投影时则是四次 GEMM。</p>
              </div>
            </div>
          </section>

          {/* ===== 矩阵乘法 ===== */}
          <section className="section" id="s1">
            <SecHead idx="01" title="热身：矩阵乘法到底怎么乘" />
            <p className="sec-lead">Attention 的主要线性运算由<b style={{ color: "#eef3ff" }}>矩阵乘法</b>完成，中间穿插缩放、mask 和 softmax。右侧结果矩阵支持逐格查看：<b style={{ color: "#f472b6" }}>点击任意格子</b>，左侧将高亮参与计算的行与列。</p>
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
              {matrixC.flatMap((row, rowIndex) => row.map((value, columnIndex) => (
                <span
                  key={`${rowIndex}-${columnIndex}`}
                  className={`matrix-calc-option ${rowIndex === mr && columnIndex === mc ? "active" : ""}`}
                  data-matrix-index={rowIndex * row.length + columnIndex}
                >
                  <Formula tex={`(U_3)_{${rowIndex + 1},${columnIndex + 1}}=${matrixA[rowIndex].map((entry, i) => `${entry}\\times${matrixB[i][columnIndex]}`).join("+")}=${value}`} />
                </span>
              )))}
            </div>
            <div className="note"><Formula tex={String.raw`(U_3)_{i,j}=\sum_{r=1}^{3}(U_1)_{i,r}(U_2)_{r,j}`} />：<Formula tex="U_1" /> 的第 <Formula tex="i" /> 行与 <Formula tex="U_2" /> 的第 <Formula tex="j" /> 列<b>逐项相乘再相加</b>。记号 <Formula tex={String.raw`U_1,U_2,U_3`} /> 用于矩阵乘法示例；<Formula tex="A" /> 专指注意力权重矩阵。</div>
          </section>

          {/* ===== 向量级 ===== */}
          <section className="section" id="s2">
            <SecHead idx="02" title="Self-Attention · 向量级（一步一步算）" />
            <p className="sec-lead">每个符号的来源明确后，可以沿 Token 1 展开完整计算，观察它如何读取 Token 1～4 的信息。没有 causal mask 时，四个位置都按同样方式并行计算。</p>

            <AttentionSetupGuide />

            <h3>只展开 Token 1：从 <Formula tex="q_1" /> 到新表示 <Formula tex="b_1" /></h3>
            <p className="section-bridge">Token 1 的输出 <Formula tex="b_1" /> 由 <Formula tex="Q" /> 的第 1 行 <Formula tex={String.raw`q_1=x_1W^Q`} /> 生成。一般地，<Formula tex="q_i" /> 生成分数矩阵 <Formula tex="S" /> 的第 <Formula tex="i" /> 行，并最终得到 <Formula tex="b_i" />；四行遵循相同算法。</p>
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

            <div className="note"><Formula tex={String.raw`q_2,q_3,q_4`} /> 遵循相同过程，分别得到 <Formula tex={String.raw`b_2,b_3,b_4`} />。四个输出按行堆叠为 <Formula tex={String.raw`O=\begin{bmatrix}b_1\\b_2\\b_3\\b_4\end{bmatrix}`} />；矩阵形式可并行计算全部四行。</div>
          </section>

          {/* ===== 矩阵级 ===== */}
          <section className="section detail-section" id="s3">
            <SecHead idx="03" title="Self-Attention · 矩阵级" />
            <p className="sec-lead"><Formula tex={String.raw`q_1,\ldots,q_4`} />、<Formula tex={String.raw`k_1,\ldots,k_4`} />、<Formula tex={String.raw`v_1,\ldots,v_4`} /> 分别按行堆成 <Formula tex={String.raw`Q,K,V`} /> 后，一次矩阵运算即可并行得到全部四行输出；它与逐个位置计算使用同一数学定义。</p>
            <div className="matrix-level-bridge">
              <div><b>只追踪 Token 1</b><Formula block tex={String.raw`b_1=\operatorname{softmax}\!\left(\frac{q_1K^{\mathsf T}}{\sqrt{d_k}}\right)V`} /></div>
              <span>四个 Query 行同时执行 →</span>
              <div><b>一次得到 Token 1～4</b><Formula block tex={String.raw`O=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V`} /></div>
            </div>
            <ScoreMatrixReadingGuide />
            <div className="note"><b>记号约定</b>：Token 按行堆叠时，<Formula tex={String.raw`Q_{\mathrm r}=X_{\mathrm r}W_{\mathrm r}^Q`} />，分数为 <Formula tex={String.raw`Q_{\mathrm r}K_{\mathrm r}^{\mathsf T}`} />。Token 按列堆叠时，<Formula tex={String.raw`X_{\mathrm c}=X_{\mathrm r}^{\mathsf T}`} />、<Formula tex={String.raw`W_{\mathrm c}^Q=(W_{\mathrm r}^Q)^{\mathsf T}`} />，于是 <Formula tex={String.raw`Q_{\mathrm c}=W_{\mathrm c}^QX_{\mathrm c}`} />，<Formula tex="K_{\mathrm c}" /> 同理；同一张“Query 行、Key 列”分数矩阵满足 <Formula tex={String.raw`Q_{\mathrm r}K_{\mathrm r}^{\mathsf T}=Q_{\mathrm c}^{\mathsf T}K_{\mathrm c}`} />。代码采用行向量约定，因此使用 <code>key.transpose(-2, -1)</code>。</div>
            <FigMatrixStage />
            <div className="note"><Formula tex={String.raw`QK^{\mathsf T}`} /> 对所有 Query–Key 组合计算相关分数，softmax 将每行分数归一化为权重，乘 <Formula tex={String.raw`V`} /> 后得到每个 Query 的加权内容。向量级四步由此合并为矩阵级计算。</div>
          </section>

          {/* ===== Scale 与 Mask ===== */}
          <section className="section detail-section" id="s4">
            <SecHead idx="04" title={<>缩放 <Formula tex={String.raw`\sqrt{d_k}`} /> 与 Mask</>} />
            <p className="sec-lead">公式里多了一个「除以 <Formula tex={String.raw`\sqrt{d_k}`} />」，叫<b style={{ color: "#eef3ff" }}>缩放（Scale）</b>。维度 <Formula tex="d_k" /> 增大时，未缩放点积的分布通常更分散，softmax 更易饱和（趋于一个 1、其余 0），梯度可能变得很小。</p>
            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>不缩放会怎样</h3>
                <p className="t3">点积是 <Formula tex="d_k" /> 个乘积之和。<Formula tex="d_k" /> 大 → 点积方差大 → softmax 更易饱和近似 one‑hot → 梯度可能变得很小。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>除以 <Formula tex={String.raw`\sqrt{d_k}`} /> 的效果</h3>
                <p className="t3">在常用独立同分布假设下，把点积方差<b style={{ color: "#eef3ff" }}>拉回 1 附近</b>，降低 softmax 过早饱和的风险。这是缩放点积注意力定义中的必要步骤。</p>
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
            <div
              className="card attention-demo-card"
              data-queries={JSON.stringify(queries)}
              data-keys={JSON.stringify(keys)}
              data-values={JSON.stringify(values)}
            >
              <div style={{ display: "flex", gap: 22, flexWrap: "wrap", justifyContent: "center", alignItems: "center" }}>
                <div style={{ textAlign: "center" }}>
                  <div className="mname">缩放分数 {tokenLabels.map((_, index) => (
                    <span key={index} className={`query-title-option ${qIdx === index ? "active" : ""}`} data-query-index={index}>
                      <Formula className="mname-formula" tex={String.raw`{\color{#38bdf8}S_{${index + 1},:}=q_{${index + 1}}K^{\mathsf T}/\sqrt{d_k}}`} />
                    </span>
                  ))}</div>
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
                  <div className="mname">权重 {tokenLabels.map((_, index) => (
                    <span key={index} className={`query-title-option ${qIdx === index ? "active" : ""}`} data-query-index={index}>
                      <Formula className="mname-formula" tex={String.raw`{\color{#38bdf8}A_{${index + 1},:}=\operatorname{softmax}(S_{${index + 1},:})}`} />
                    </span>
                  ))}</div>
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
                  <div className="mname">输出 {tokenLabels.map((_, index) => (
                    <span key={index} className={`query-title-option ${qIdx === index ? "active" : ""}`} data-query-index={index}>
                      <Formula className="mname-formula" tex={String.raw`{\color{#f472b6}b_{${index + 1}}=\sum_j A_{${index + 1},j}v_j}`} />
                    </span>
                  ))}</div>
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
            <p className="sec-lead">自回归生成时，每个位置必须「看到自身与过去、看不到未来」。实现上使用<b style={{ color: "#f5b042" }}>下三角因果掩码（Causal Mask）</b>：Token <Formula tex="i" /> 只允许查看 Token <Formula tex={String.raw`1,\ldots,i`} /> 对应的 Key；padding 位则由 padding mask 另行屏蔽。</p>

            <div className="mask-grid">
              <table>
                <tbody>
                  <tr>
                    <th></th>
                    <th><Formula tex="k_1" /><br /><small style={{ color: "#4d577f" }}>Token 1</small></th>
                    <th><Formula tex="k_2" /><br /><small style={{ color: "#4d577f" }}>Token 2</small></th>
                    <th><Formula tex="k_3" /><br /><small style={{ color: "#4d577f" }}>Token 3</small></th>
                    <th><Formula tex="k_4" /><br /><small style={{ color: "#4d577f" }}>Token 4</small></th>
                  </tr>
                  <tr>
                    <th><Formula tex="q_1" /> · Token 1</th>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell block"><Formula tex="-\infty" /></div></td>
                    <td><div className="mask-cell block"><Formula tex="-\infty" /></div></td>
                    <td><div className="mask-cell block"><Formula tex="-\infty" /></div></td>
                  </tr>
                  <tr>
                    <th><Formula tex="q_2" /> · Token 2</th>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell block"><Formula tex="-\infty" /></div></td>
                    <td><div className="mask-cell block"><Formula tex="-\infty" /></div></td>
                  </tr>
                  <tr>
                    <th><Formula tex="q_3" /> · Token 3</th>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell block"><Formula tex="-\infty" /></div></td>
                  </tr>
                  <tr>
                    <th><Formula tex="q_4" /> · Token 4</th>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                    <td><div className="mask-cell allow"><Formula tex="0" /></div></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="fig-cap">与前文 <Formula tex={String.raw`S,A\in\mathbb{R}^{4\times4}`} /> 使用同一坐标：绿色 <b style={{ color: "#34d399" }}><Formula tex="0" /></b> = 允许看，红色 <b style={{ color: "#f472b6" }}><Formula tex="-\infty" /></b> = 屏蔽；第 <Formula tex="i" /> 行只允许查看第 <Formula tex={String.raw`1,\ldots,i`} /> 列</div>

            <h3 style={{ marginTop: 30 }}>Mask 加在哪一步</h3>
            <div className="flow-chain">
              <b><Formula tex={String.raw`QK^{\mathsf T}`} /></b><em>→</em>
              <b>÷ <Formula tex={String.raw`\sqrt{d_k}`} /></b><em>→</em>
              <b className="hi">+ Mask（<Formula tex="-\infty" />）</b><em>→</em>
              <b>softmax</b><em>→</em>
              <b><Formula tex={String.raw`\times V`} /></b>
            </div>
            <div className="eq-box">
              <Formula block tex={String.raw`O=\operatorname{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V,\quad M_{i,j}=\begin{cases}0 & i\ge j\\ -\infty & i<j\end{cases}`} />
            </div>
            <div className="note warn"><b>实现陷阱</b>：Mask 必须在 softmax <b>之前</b>加 <code>−∞</code>。若在 softmax 后再乘 0，屏蔽位虽然归零，但剩余权重之和不再为 1，输出尺度会出错。</div>
            <div className="note"><b>位置机制与 Mask 的职责不同</b>：位置编码或位置偏置向模型提供顺序与距离信息；Mask 则直接规定某条 Query–Key 连接是否允许参与 softmax。加入位置编码后，因果生成仍然需要 Mask 阻断未来信息。</div>

            <h3 style={{ marginTop: 24 }}>同一个因果约束，两种执行方式</h3>
            <div className="grid2">
              <div className="card">
                <h3 style={{ marginTop: 0 }}>推理（逐 token，串行）</h3>
                <p className="t3" style={{ fontFamily: "var(--mono)", fontSize: 13 }}>
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt;</b> → 预测 Token 1<br />
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt; Token 1</b> → 预测 Token 2<br />
                  输入 <b style={{ color: "#f5b042" }}>&lt;BOS&gt; Token 1 Token 2</b> → 预测 Token 3<br />
                  ……直到 <b style={{ color: "#f5b042" }}>&lt;EOS&gt;</b>
                </p>
                <p className="t3">每一步依赖此前已生成的内容，因此解码按步串行；配合 KV cache 时 Key 本身就只有历史前缀，不一定需要显式的完整三角 mask。</p>
              </div>
              <div className="card">
                <h3 style={{ marginTop: 0 }}>训练（整句并行）</h3>
                <p className="t3" style={{ fontFamily: "var(--mono)", fontSize: 13 }}>
                  输入 <b style={{ color: "#2dd4bf" }}>&lt;BOS&gt; Token 1 Token 2 Token 3</b>（右移一位）<br />
                  目标 <b style={{ color: "#f472b6" }}>Token 1 Token 2 Token 3 Token 4</b><br />
                  一次前向 + 因果 Mask
                </p>
                <p className="t3">训练通过 causal mask 并行处理整句，同时让每个位置只接收允许的历史前缀；因果约束由 mask 显式施加。</p>
              </div>
            </div>
            <div className="note"><code>&lt;BOS&gt;</code> 是序列开始标记，只负责为“预测第一个 Token”提供首个输入；它不是因果 Mask 的组成部分，也不改变下三角约束的定义。<code>&lt;EOS&gt;</code> 则表示序列结束。</div>
          </section>

          {/* ===== 多头 ===== */}
          <section className="section detail-section" id="s5">
            <SecHead idx="05" title="多头注意力（Multi-Head）" />
            <p className="sec-lead">多头注意力让同一输入经过多组独立投影，每个头分别计算完整的 Token 关联，再把各头输出拼接并通过 <Formula tex="W^O" /> 融合。实现通常先用融合矩阵完成投影，再通过 reshape 恢复头维；数值链采用四个 Token、六维模型空间和三个二维头。</p>

            <FigMultiHeadCalculation />

            <h3>把完整流程压回论文公式</h3>
            <div className="eq-box">
              <Formula block tex={String.raw`\operatorname{head}_i=\operatorname{Attention}(XW_i^Q,\,XW_i^K,\,XW_i^V)`} />
              <Formula block tex={String.raw`\operatorname{MHA}=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\,W^O`} />
            </div>
            <div className="note">其中 <Formula tex="h" /> 是头数，<Formula tex={String.raw`W^O\in\mathbb{R}^{(h d_v)\times d_{\mathrm{model}}}`} /> 把拼接结果投影回模型维度。原始 Transformer 与常见实现取 <Formula tex={String.raw`d_k=d_v=d_{\mathrm{model}}/h`} />，所以主 FLOPs 量级与同模型宽度的单头接近、表达能力更强；但 <Formula tex={String.raw`h\,d_v=d_{\mathrm{model}}`} /> 不是多头注意力在数学上的必要条件。</div>

            <h3>同一输入下，三个头得到不同的注意力权重</h3>
            <div className="tabs">
              {heads.map((h, i) => (
                <button key={h.name} className={`tab ${headIdx === i ? "active" : ""}`} onClick={() => setHeadIdx(i)}>{h.name}</button>
              ))}
            </div>
            <div className="fig head-demo" data-heads={JSON.stringify(heads)}>
              <div style={{ display: "flex", justifyContent: "center" }}>
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
              <div className="fig-cap"><b>{heads[headIdx].name} 的数值链计算结果</b> · 行=Query（谁在问）· 列=Key（看谁）</div>
            </div>
          </section>

          {/* ===== FlashAttention ===== */}
          <section className="section detail-section" id="s6">
            <SecHead idx="06" title="FlashAttention：不改变数学，改变算的方式" />
            <p className="sec-lead"><b style={{ color: "#eef3ff" }}>SDPA</b> 是 Scaled Dot-Product Attention（缩放点积注意力）；FlashAttention 将同一个 SDPA 分块执行。它在实数精确计算下与普通实现完全等价，不是近似注意力；实际浮点内核会因归约顺序不同产生微小舍入差异。</p>

            <h3>数学没有变化：仍然是打分、归一化、汇聚</h3>
            <div className="eq-box">
              <Formula block tex={String.raw`S=\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M`} />
              <Formula block tex={String.raw`A=\operatorname{softmax}(S)`} />
              <Formula block tex={String.raw`O=AV`} />
            </div>

            <h3>同一组数字：普通 Softmax 与 FlashAttention 逐步对照</h3>
            <FigFlashNumericTiles />

            <h3>从数值例收束成通用更新公式</h3>
            <div className="eq-box">
              <Formula block tex={String.raw`m_{\mathrm{new}}=\max\!\left(m,\,\operatorname{rowmax}(S_t)\right)`} />
              <Formula block tex={String.raw`\rho=e^{m-m_{\mathrm{new}}}`} />
              <Formula block tex={String.raw`\ell_{\mathrm{new}}=\underbrace{\rho\odot\ell}_{\text{旧分母乘换算比例}}+\underbrace{\operatorname{rowsum}\!\left(e^{S_t-m_{\mathrm{new}}}\right)}_{\text{新 tile 分母}}`} />
              <Formula block tex={String.raw`o_{\mathrm{new}}=\underbrace{\rho\odot o}_{\text{旧分子乘换算比例}}+\underbrace{e^{S_t-m_{\mathrm{new}}}V_t}_{\text{新 tile 分子}}`} />
              <Formula block tex={String.raw`O_{\mathrm{blk}}[i,:]=\frac{o[i,:]}{\ell_i}\qquad\Longleftrightarrow\qquad O_{\mathrm{blk}}=\operatorname{diag}(\ell)^{-1}o`} />
            </div>
            <div className="flash-symbol-row"><span><Formula tex="m" />：已扫描分数的最大值</span><span><Formula tex="\ell" />：Softmax 分母</span><span><Formula tex="o" />：未归一化的 <Formula tex="AV" /> 分子</span></div>
            <div className="flash-diag-explainer"><Formula tex={String.raw`\operatorname{diag}(\ell)`} /><span>把每行分母 <Formula tex={String.raw`\ell_i`} /> 放在对角线上；左乘其逆矩阵，等价于让 <Formula tex="o" /> 的第 <Formula tex="i" /> 行除以 <Formula tex={String.raw`\ell_i`} />。</span></div>

            <h3>为什么能够节省显存并加速</h3>
            <FigFlashCompare />

            <h3>tile 在 GPU 上怎样循环</h3>
            <FigFlashGpuMap />

            <div className="flash-summary-row"><span>数学定义精确等价；浮点结果按容差比较</span><span>HBM 不物化完整 S、A</span><span>算术量仍为 O(L²d)，减少的是数据搬运</span></div>
          </section>

          {/* ===== 代码实现 ===== */}
          <section className="section detail-section" id="s7">
            <SecHead idx="07" title="代码实现：从公式到 PyTorch" />
            <p className="sec-lead"><b style={{ color: "#eef3ff" }}>透明参考实现</b>逐项对应缩放点积、Mask、softmax 与 Value 汇聚；<b style={{ color: "#2dd4bf" }}>PyTorch SDPA</b> 执行相同数学定义。多头代码还包括投影、拆头、拼接与 <Formula tex="W^O" />。</p>

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
        # 与 SDPA 一致：bool mask 中 True=允许；float mask 直接相加。
        scores = (scores.masked_fill(~mask, float("-inf"))
                  if mask.dtype == torch.bool else scores + mask)
    weights = torch.softmax(scores, dim=-1)   # softmax 按行
    return weights @ v                         # · V`}</code></pre>
            <div className="note">该参考实现接受与 SDPA 相同的两类 mask：布尔值 <code>True</code> 表示允许参与，浮点 mask 直接加到分数上；这与 <code>nn.MultiheadAttention</code> 的布尔 <code>key_padding_mask</code> 语义相反。参考实现会物化完整分数矩阵，长序列下通常使用融合算子减少中间存储与显存读写。</div>

            <div className="code-title">② 真实算子 — PyTorch SDPA（按输入与设备选可用后端）</div>
            <pre><code>{`output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=mask,     # bool: True=允许；float: 加到分数上
    dropout_p=0.0,
    is_causal=False,
)
# 需要对照不同后端时，可强制选择内核：
# with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION): ...`}</code></pre>
            <div className="note"><b>F.scaled_dot_product_attention</b>（SDPA）会根据设备、数据类型、shape 与后端可用性进行调度；常见实现包括 PyTorch math、FlashAttention 和 Memory‑Efficient Attention，具体后端集合会随版本与构建变化。它不会在每次调用时逐个实测再挑最快，因此既不能只凭 API 名称断言命中 FlashAttention，也不应假设一定回退到某个固定后端。可用 <code>sdpa_kernel</code> 限定后端做对照验证。各实现与参考公式在实数下等价；浮点归约顺序不同，结果应在合理容差内比较。</div>

            <div className="code-title">③ 多头自注意力 — reshape → transpose → SDPA → concat → <Formula tex="W^O" /></div>
            <pre><code>{`class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        # 使用无偏置投影，对应 Q=XW^Q 等公式
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
            <div className="note">多头实现依次执行<b>投影 → reshape/transpose 拆头 → SDPA → 拼头 → 输出投影</b>。reshape + transpose 将 <Formula tex="h" /> 个头变成独立批次维度并行计算，无需 Python 循环。<code>bias=False</code> 对应无偏置公式；PyTorch 的 <code>nn.Linear</code> 计算 <Formula tex={String.raw`xW_{\mathrm{store}}^{\mathsf T}`} />，因此行向量记法中的 <Formula tex="W^Q" /> 对应 <Formula tex={String.raw`W_{\mathrm{store}}^{\mathsf T}`} />。</div>

            <div className="note ok"><b>核心结论</b>：Attention 用 <Formula tex={String.raw`QK^{\mathsf T}\!/\sqrt{d_k}`} /> 计算位置间的匹配分数，经 softmax 得到权重，再乘 <Formula tex="V" /> 汇聚内容；多头并行学习多组关系，FlashAttention 在保持数学定义不变的前提下优化计算与数据搬运。</div>
          </section>

          {/* ===== 代码 ===== */}
          <section className="section detail-section" id="s8">
            <SecHead idx="08" title="Transformer 全景：Attention 在模型中的位置" />
            <p className="sec-lead">完整 Transformer 由 <b style={{ color: "#38bdf8" }}>Encoder</b> 与 <b style={{ color: "#f472b6" }}>Decoder</b> 组成，两侧分别堆叠 <Formula tex="N" /> 层，并在不同位置调用 Attention。</p>
            <FigTransformer />
            <div className="grid2">
              <div className="note"><b>Encoder</b>：对源序列做 Self‑Attention + FFN，逐层提炼表示，输出的 Memory 作为 Cross‑Attention 的 <Formula tex={String.raw`K,\ V`} /> 来源。</div>
              <div className="note"><b>Decoder</b>：先用 <b>Masked</b> Self‑Attention（屏蔽未来位以保证自回归因果性），再通过 <b>Cross‑Attention</b> 把编码器 Memory 投影成 <Formula tex={String.raw`K,\ V`} /> 来读取，最后预测下一个词。</div>
            </div>
            <div className="note warn"><b>Attention 的三种调用</b>：Encoder Self‑Attention 建模源序列内部关系；Decoder Masked Self‑Attention 建模已生成内容；Cross‑Attention 读取 Encoder Memory。</div>

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
                <p className="t3">采用只含 causal self‑attention 的 Decoder-style 堆叠做自回归生成，不含独立 Encoder 与 Cross‑Attention。</p>
              </div>
            </div>

            <h3>位置编码：把「顺序」补回去</h3>
            <p className="sec-lead">位置向量与内容向量共同组成 <Formula tex="X" />；可学习位置表通过索引取值，原始 Transformer 则直接使用固定的正弦 / 余弦公式。</p>
            <PositionEncodingFlow />
            <div className="note">现代模型未必沿用“输入端直接相加”：例如 <b>RoPE</b> 在每层对 <Formula tex={String.raw`Q,\ K`} /> 做与位置相关的旋转，<b>ALiBi</b> 在注意力分数上加入线性位置偏置。注入位置不同，但目标相同——让 Attention 能感知顺序与距离。</div>
          </section>

          <div className="foot">
            参考：Vaswani et al. <i>Attention Is All You Need</i> · Dao et al. <i>FlashAttention</i> / <i>FlashAttention-2</i> · PyTorch SDPA 文档。
          </div>

        </div>
      </main>
    </>
  );
}
