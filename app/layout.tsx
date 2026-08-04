import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Attention Lab｜从零理解 Transformer 核心算子",
  description:
    "面向初学者的 Attention 互动讲义：矩阵乘法、QKV、自注意力、多头注意力、Transformer 架构、经典 PyTorch 代码与算子测试。",
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
