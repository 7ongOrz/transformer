# Attention Operator Guide

面向技术分享的中文单页讲义，从矩阵乘法、Q/K/V 投影和 Scaled Dot-Product Attention，讲到多头注意力、FlashAttention、算子测试与经典 Transformer 结构。

## 直接使用

双击仓库根目录的 `attention.html` 即可离线打开，不需要安装 Node.js 或 pnpm。

参考资料包括：

- `Vision Transformer 超详细解读 (原理分析+代码解读) (一) - 知乎.pdf`
- `reference/vision-transformer-pdf-readable.html`：便于模型读取的完整转换版本

## 本地开发

需要 Node.js 22 和 pnpm。

```bash
pnpm install
pnpm dev
```

常用检查：

```bash
pnpm lint
pnpm test
pnpm export:standalone
```

`pnpm export:standalone` 会重新构建并更新根目录的 `attention.html`。

## 主要文件

- `app/page.tsx`：页面内容与交互
- `app/globals.css`：页面样式
- `app/attention-demo.js`：全文统一使用的教学数值
- `scripts/export-standalone.mjs`：离线单文件导出
- `tests/rendered-html.test.mjs`：渲染、公式、数值和离线文件检查
