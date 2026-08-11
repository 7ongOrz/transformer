import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";
import { attentionDemo } from "../app/attention-demo.js";

const rounded = (matrix, digits = 3) =>
  matrix.map((row) => row.map((value) => Number(value.toFixed(digits))));

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the Attention teaching page", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="zh-CN">/i);
  assert.match(html, /<title>Attention｜Transformer 核心算子详解<\/title>/i);
  assert.match(html, /Attention 算子/);
  assert.match(html, /热身：矩阵乘法到底怎么乘/);
  assert.match(html, /Self-Attention · 向量级/);
  assert.match(html, /Self-Attention · 矩阵级/);
  assert.match(html, /与 Mask/);
  assert.match(html, /多头注意力（Multi-Head）/);
  assert.match(html, /FlashAttention：不改变数学/);
  assert.match(html, /代码与算子测试：从原理到真实算子/);
  assert.match(html, /Transformer 全景：Attention 被装在哪里/);
  assert.match(html, /生成分数矩阵/);
  assert.match(html, /内容矩阵/);
  assert.match(html, /位置矩阵/);
  assert.match(html, /输入矩阵/);
  assert.match(html, /one-hot 选择位置向量/);
  assert.match(html, /A.*第 1 行乘完整.*V.*矩阵/);
  assert.match(html, /data-queries=/);
  assert.match(html, /<svg\b/i);
  assert.match(html, /katex/);
  assert.doesNotMatch(html, /class="[^"]*\bmath-error\b/);
  assert.doesNotMatch(html, /20\s*(?:分钟|MIN)|TOTAL\s*·\s*20:00|20:00/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("keeps the teaching matrices on one verified numerical chain", () => {
  assert.deepEqual(rounded(attentionDemo.Q, 2), [
    [0.04, 1.16],
    [1.41, 0.99],
    [0.53, 1.12],
    [0.65, 1.75],
  ]);
  assert.deepEqual(rounded(attentionDemo.K, 2), [
    [0.56, 1.16],
    [1.26, -0.27],
    [0.82, 0.67],
    [1.18, 1.21],
  ]);
  assert.deepEqual(rounded(attentionDemo.V, 2), [
    [0.4, 1.28],
    [1.08, 0.6],
    [0.65, 1.06],
    [0.92, 1.72],
  ]);
  assert.deepEqual(rounded(attentionDemo.S), [
    [0.967, -0.186, 0.573, 1.026],
    [1.37, 1.067, 1.287, 2.024],
    [1.129, 0.258, 0.838, 1.4],
    [1.693, 0.245, 1.206, 2.04],
  ]);
  assert.deepEqual(rounded(attentionDemo.A), [
    [0.328, 0.103, 0.221, 0.348],
    [0.218, 0.161, 0.201, 0.42],
    [0.287, 0.12, 0.215, 0.377],
    [0.306, 0.072, 0.188, 0.433],
  ]);
  assert.deepEqual(rounded(attentionDemo.O), [
    [0.706, 1.314],
    [0.778, 1.311],
    [0.732, 1.317],
    [0.721, 1.38],
  ]);
});

test("keeps project metadata and generated assets clean", async () => {
  const [page, layout, packageJson, gitignore] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../.gitignore", import.meta.url), "utf8"),
  ]);

  assert.match(page, /katex\.renderToString/);
  assert.match(page, /def attention_ref/);
  assert.match(page, /scaled_dot_product_attention/);
  assert.match(page, /function FigTransformer/);
  assert.match(page, /function FigFlashCompare/);
  assert.match(page, /代码与算子测试：从原理到真实算子/);
  assert.match(page, /E_\{\\mathrm\{tok\}\}/);
  assert.match(page, /\\mathcal O/);
  assert.doesNotMatch(page, /d_\{model\}|L_\{max\}|W[ᵠᵏᵛ]|tex="(?:Q\/K\/V|S\/P|dQ,dK,dV)"/);
  assert.match(layout, /Transformer 核心算子详解/);
  assert.doesNotMatch(`${page}\n${layout}`, /20\s*(?:分钟|MIN)|20:00/i);
  assert.match(packageJson, /"name": "attention-operator-lab"/);
  assert.match(packageJson, /"katex":/);
  assert.doesNotMatch(packageJson, /drizzle|react-loading-skeleton/);
  assert.match(gitignore, /^\.DS_Store$/m);
  await assert.rejects(access(new URL("../.DS_Store", import.meta.url)));
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});

test("keeps the PDF conversion complete and machine-readable", async () => {
  const readableHtml = await readFile(
    new URL("../reference/vision-transformer-pdf-readable.html", import.meta.url),
    "utf8",
  );

  for (let figure = 1; figure <= 40; figure += 1) {
    assert.match(readableHtml, new RegExp(`id="figure-${figure}"`));
  }
  for (let page = 1; page <= 5; page += 1) {
    assert.match(readableHtml, new RegExp(`===== PDF PAGE ${page} =====`));
  }

  assert.match(readableHtml, /PDF column convention: Q=W\^Q I/);
  assert.match(readableHtml, /Modern row-major: S=QK\^T/);
  assert.match(readableHtml, /masked_logits = Q @ K\.T \/ sqrt\(d_k\) \+ M/);
  assert.match(readableHtml, /Object Queries: \[100,B,256\]/);
  assert.match(readableHtml, /Hungarian Matching 计算过程/);
  assert.doesNotMatch(readableHtml, /<(?:img|canvas|svg)\b/i);
});

test("ships the current teaching page as an offline standalone HTML file", async () => {
  const html = await readFile(new URL("../attention.html", import.meta.url), "utf8");

  assert.match(html, /Offline standalone export/);
  assert.match(html, /Self-Attention · 向量级/);
  assert.match(html, /从第一行扩展到全部四行/);
  assert.match(html, /softmax 后权重/);
  assert.match(html, /生成分数矩阵/);
  assert.match(html, /内容矩阵/);
  assert.match(html, /位置矩阵/);
  assert.match(html, /输入矩阵/);
  assert.match(html, /one-hot 选择位置向量/);
  assert.match(html, /FlashAttention：不改变数学/);
  assert.match(html, /data-standalone="attention"/);
  assert.match(html, /data-queries=/);
  assert.match(html, /JSON\.parse\(attentionDemo\.dataset\.queries\)/);
  assert.match(html, /data:font\/woff2;base64,/);
  assert.doesNotMatch(html, /class="[^"]*\bmath-error\b/);
  assert.doesNotMatch(html, /<(?:script|link|img)\b[^>]*(?:src|href)="https?:\/\//i);
  assert.doesNotMatch(html, /(?:href|src)="\/assets\//i);
  assert.doesNotMatch(html, /(?:href|src)="\/(?!\/)/i);
});
