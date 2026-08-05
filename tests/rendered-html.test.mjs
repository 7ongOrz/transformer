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
  assert.match(html, /零基础热身：矩阵乘法到底怎么乘/);
  assert.match(html, /从 RNN 的痛点说起/);
  assert.match(html, /Self-Attention · 向量级/);
  assert.match(html, /Self-Attention · 矩阵级/);
  assert.match(html, /多头注意力（Multi-Head）/);
  assert.match(html, /经典 Transformer 全景图（论文 Figure 1）/);
  assert.match(html, /经典代码 \+ 算子测试要点/);
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
  assert.match(page, /def attention\(query, key, value, mask=None\)/);
  assert.match(page, /class MultiHeadedAttention\(nn\.Module\)/);
  assert.match(page, /function FigTransformer/);
  assert.match(page, /算子测试要点（leader 关心的）/);
  assert.match(layout, /Transformer 核心算子详解/);
  assert.doesNotMatch(`${page}\n${layout}`, /20\s*(?:分钟|MIN)|20:00/i);
  assert.match(packageJson, /"name": "attention-operator-lab"/);
  assert.match(packageJson, /"katex":/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(gitignore, /^\.DS_Store$/m);
  await assert.rejects(access(new URL("../.DS_Store", import.meta.url)));
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});
