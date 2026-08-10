import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

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
  assert.match(html, /<svg\b/i);
  assert.match(html, /katex/);
  assert.doesNotMatch(html, /20\s*(?:分钟|MIN)|TOTAL\s*·\s*20:00|20:00/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
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
  assert.match(layout, /Transformer 核心算子详解/);
  assert.doesNotMatch(`${page}\n${layout}`, /20\s*(?:分钟|MIN)|20:00/i);
  assert.match(packageJson, /"name": "attention-operator-lab"/);
  assert.match(packageJson, /"katex":/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
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
  assert.match(html, /本图不含 Wᴼ/);
  assert.match(html, /FlashAttention：不改变数学/);
  assert.match(html, /data-standalone="attention"/);
  assert.match(html, /data:font\/woff2;base64,/);
  assert.doesNotMatch(html, /<(?:script|link|img)\b[^>]*(?:src|href)="https?:\/\//i);
  assert.doesNotMatch(html, /(?:href|src)="\/assets\//i);
});
