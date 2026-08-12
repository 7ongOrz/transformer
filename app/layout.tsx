import type { Metadata } from "next";
import "katex/dist/katex.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Attention｜Transformer 核心算子详解",
  description:
    "Attention 技术讲义：矩阵乘法、QKV、Scaled Dot-Product Attention、多头注意力、FlashAttention、PyTorch 实现与 Transformer 全景。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
