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
  assert.match(html, /为什么需要 Self-Attention/);
  assert.match(html, /只需要一个矩阵乘法规则/);
  assert.match(html, /Scaled Dot-Product Attention/);
  assert.match(html, /一条完整的数据流：X 如何变成 Z/);
  assert.match(html, /多头不是重复计算：它把特征维拆成多个子空间/);
  assert.match(html, /经典 Transformer Encoder–Decoder 全结构/);
  assert.match(html, /Encoder–Decoder Attention/);
  assert.match(html, /经典代码与算子测试/);
  assert.match(html, /从数学原理到经典实现与算子测试/);
  assert.match(html, /katex/);
  assert.doesNotMatch(html, /20\s*(?:分钟|MIN)|TOTAL\s*·\s*20:00|20:00/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("removes disposable starter assets and keeps project metadata specific", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /katex\.renderToString/);
  assert.match(page, /def attention\(q, k, v, mask=None\)/);
  assert.match(page, /class EncoderLayer\(nn\.Module\)/);
  assert.match(page, /FIG 03 · ORIGINAL TRANSFORMER/);
  assert.match(page, /OPERATOR CHECKLIST/);
  assert.match(layout, /Transformer 核心算子详解/);
  assert.doesNotMatch(`${page}\n${layout}`, /20\s*(?:分钟|MIN)|20:00/i);
  assert.match(packageJson, /"name": "attention-operator-lab"/);
  assert.match(packageJson, /"katex":/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});
