"use client";

import { useState } from "react";
import { Formula } from "./formula";
import { LessonNavigation } from "./lesson-navigation";
import { AttentionSetupGuide, ScoreMatrixReadingGuide, FigStageQKV, FigStageScore, FigStageSoftmax, FigStageAggregate, FigMatrixStage, FigMultiHeadCalculation, FigTransformer, PositionEncodingFlow, FigFlashCompare, FigFlashGpuMap, FigFlashNumericTiles } from "./teaching-figures";
import {
  attentionDemo,
  attentionHeads,
  matrixMultiplicationDemo,
} from "./attention-demo.js";

/* ---------- 矩阵乘法演示数据 ---------- */
const { A: matrixA, B: matrixB, C: matrixC } = matrixMultiplicationDemo;

/* ---------- Attention 演示数据（全页统一使用 4 个位置标签，d=2） ---------- */
const tokenLabels = ["token₁", "token₂", "token₃", "token₄"];

/* ---------- 多头热力图数据 ---------- */
const heads = attentionHeads;

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
  const [selCell, setSelCell] = useState<[number, number]>([0, 0]);
  const [qIdx, setQIdx] = useState(0);
  const [headIdx, setHeadIdx] = useState(0);
  const attn = {
    scaled: attentionDemo.S[qIdx],
    weights: attentionDemo.A[qIdx],
    output: attentionDemo.O[qIdx],
  };

  const [mr, mc] = selCell;


  return (
    <>
      <LessonNavigation />

      <main className="main" id="main-content" tabIndex={-1}>
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
                <p className="t3">Attention 的核心就是<b style={{ color: "#eef3ff" }}>两次矩阵乘法（<Formula tex={String.raw`QK^{\mathsf T}`} /> 计算相关分数、<Formula tex="AV" /> 加权汇聚）+ 一个 softmax</b>；完整多头还包括 <Formula tex={String.raw`Q,\ K,\ V`} /> 与 <Formula tex="W^O" /> 四个投影，即六类矩阵乘法运算。QKV 投影可以合并，多个头可以批量计算；这些数学步骤不等于 GPU 内核的启动次数。</p>
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
                            type="button"
                            aria-label={`查看结果矩阵第 ${r + 1} 行、第 ${c + 1} 列的计算`}
                            aria-pressed={mr === r && mc === c}
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
          <section className="section" id="s3">
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
          <section className="section" id="s4">
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
                <button key={w} className={`tab ${qIdx === i ? "active" : ""}`} type="button" aria-pressed={qIdx === i} onClick={() => setQIdx(i)}><Formula tex={`q_${i + 1}`} /> · {w}</button>
              ))}
            </div>
            <div
              className="card attention-demo-card"
              data-attention={JSON.stringify({ S: attentionDemo.S, A: attentionDemo.A, O: attentionDemo.O })}
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
          <section className="section" id="s5">
            <SecHead idx="05" title="多头注意力（Multi-Head）" />
            <p className="sec-lead">多头注意力让同一输入经过多组独立投影，每个头分别计算完整的 Token 关联，再把各头输出拼接并通过 <Formula tex="W^O" /> 融合。实现通常先用融合矩阵完成投影，再通过 reshape 恢复头维；数值链采用四个 Token、六维模型空间和三个二维头。</p>

            <FigMultiHeadCalculation />

            <h3>把完整流程压回论文公式</h3>
            <div className="eq-box">
              <Formula block tex={String.raw`\operatorname{head}_i=\operatorname{Attention}(XW_i^Q,\,XW_i^K,\,XW_i^V)`} />
              <Formula block tex={String.raw`\operatorname{MHA}=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\,W^O`} />
            </div>
            <div className="note">其中 <Formula tex="h" /> 是头数，<Formula tex={String.raw`W^O\in\mathbb{R}^{(h d_v)\times d_{\mathrm{model}}}`} /> 把拼接结果投影回模型维度。原始 Transformer 与常见实现取 <Formula tex={String.raw`d_k=d_v=d_{\mathrm{model}}/h`} />，使主要矩阵乘法的 FLOPs 与同模型宽度的单头接近，同时允许各头学习不同的关注关系；但 <Formula tex={String.raw`h\,d_v=d_{\mathrm{model}}`} /> 不是多头注意力在数学上的必要条件，也不保证每个头都会形成不同分工。</div>

            <h3>同一输入下，三个头得到不同的注意力权重</h3>
            <div className="tabs">
              {heads.map((h, i) => (
                <button key={h.name} className={`tab ${headIdx === i ? "active" : ""}`} type="button" aria-pressed={headIdx === i} onClick={() => setHeadIdx(i)}>{h.name}</button>
              ))}
            </div>
            <div className="fig head-demo" data-heads={JSON.stringify(heads)}>
              <div className="head-demo-table">
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
          <section className="section" id="s6">
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

            <div className="note">最大值、分母和换算比例都按 Query 行保存；旧分子的每一行整体乘该行比例，包含全部 Value 特征。因果 Mask 下完全不可见的 tile 不贡献分子或分母，可跳过；通用更新公式中的最大值需来自有限的可见分数。</div>

            <h3>为什么能够节省显存并加速</h3>
            <FigFlashCompare />

            <h3>tile 在 GPU 上怎样循环</h3>
            <FigFlashGpuMap />

            <div className="flash-summary-row"><span>数学定义精确等价；浮点结果按容差比较</span><span>HBM 不物化完整 S、A</span><span>算术量仍为 O(L²d)，减少的是数据搬运</span></div>
          </section>

          {/* ===== 代码实现 ===== */}
          <section className="section" id="s7">
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
            <div className="note">该参考实现展示 SDPA 的两类 mask 语义：布尔值 <code>True</code> 表示允许参与，浮点 mask 直接加到分数上；这与 <code>nn.MultiheadAttention</code> 的布尔 <code>key_padding_mask</code> 语义相反。这里假设每个 Query 至少有一个可见 Key；整行都是 <code>−∞</code> 时，直接 softmax 会得到 NaN，不能用这段参考代码代表融合内核的边界行为。参考实现会物化完整分数矩阵，长序列下通常使用融合算子减少中间存储与显存读写。</div>

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
          <section className="section" id="s8">
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
