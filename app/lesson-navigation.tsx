"use client";

import { memo, useEffect } from "react";
import { observeReadingProgress } from "./reading-progress.js";

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

export const LessonNavigation = memo(function LessonNavigation() {
  useEffect(observeReadingProgress, []);
  return (
    <>
      <a className="skip-link" href="#main-content">跳到正文</a>
      <div className="progress" aria-hidden="true" />
      <nav className="sidenav" aria-label="章节目录">
        <div className="brand">
          <span className="glyph">A</span>
          <b>Attention</b>
        </div>
        <div className="sub">Transformer 核心算子</div>
        <ol>
          {navItems.map(([id, label]) => (
            <li key={id}>
              <a href={`#${id}`}>{label}</a>
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
    </>
  );
});
