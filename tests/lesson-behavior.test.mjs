import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { runInNewContext } from "node:vm";
import { attentionDemo } from "../app/attention-demo.js";
import { observeReadingProgress } from "../app/reading-progress.js";

test("offline export carries the exact shared attention results and accessible controls", async () => {
  const html = await readFile(new URL("../attention.html", import.meta.url), "utf8");
  const data = JSON.parse(html.match(/data-attention="([^"]+)"/)[1].replaceAll("&quot;", '"'));
  assert.deepEqual(data, { S: attentionDemo.S, A: attentionDemo.A, O: attentionDemo.O });
  assert.equal([...html.matchAll(/aria-pressed="(?:true|false)"/g)].length, 11);
  assert.equal([...html.matchAll(/aria-pressed="true"/g)].length, 3);
  assert.match(html, /aria-label="章节目录"/);
  assert.match(html, /href="#main-content"/);
  assert.ok([...html.matchAll(/encoding="application\/x-tex"/g)].length > 400);
  assert.doesNotMatch(html, /katex-error/);
});

test("reading progress coalesces scroll events, updates on resize, and cleans up", () => {
  const handlers = new Map();
  const frames = new Map();
  const progress = { style: {} };
  const links = Array.from({ length: 3 }, (_, index) => ({
    hash: `#s${index}`,
    active: false,
    attributes: {},
    classList: { toggle(_, value) { links[index].active = value; } },
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this.attributes[name]; },
  }));
  const context = {
    innerHeight: 800,
    scrollY: 0,
    document: {
      documentElement: { scrollHeight: 2400 },
      querySelectorAll: () => links,
      querySelector: selector => selector === ".progress" ? progress : {
        getBoundingClientRect: () => ({ top: Number(selector.slice(2)) * 800 - context.scrollY }),
      },
    },
    requestAnimationFrame(callback) { frames.set(1, callback); return 1; },
    cancelAnimationFrame(id) { frames.delete(id); },
    addEventListener(name, handler) { handlers.set(name, handler); },
    removeEventListener(name) { handlers.delete(name); },
  };
  const cleanup = runInNewContext(`(${observeReadingProgress.toString()})()`, context);
  assert.equal(progress.style.width, "0%");
  assert.equal(links[0].attributes["aria-current"], "location");
  context.scrollY = 900;
  for (let event = 0; event < 100; event++) handlers.get("scroll")();
  assert.equal(frames.size, 1);
  const callback = frames.get(1);
  frames.clear();
  callback();
  assert.equal(progress.style.width, "56.25%");
  assert.deepEqual(links.map(link => link.active), [false, true, false]);
  assert.equal(links[0].attributes["aria-current"], undefined);
  context.innerHeight = 2400;
  handlers.get("resize")();
  frames.get(1)();
  assert.equal(progress.style.width, "0%");
  handlers.get("scroll")();
  cleanup();
  assert.equal(handlers.size, 0);
  assert.equal(frames.size, 0);
});

test("online softmax preserves the global result for different tile sizes and large scores", () => {
  const scores = [1001, 1000, 999, 1003, 995];
  const values = [[1, -2], [3, 4], [-5, 6], [7, -8], [9, 10]];
  const maximum = Math.max(...scores);
  const exponents = scores.map(score => Math.exp(score - maximum));
  const denominator = exponents.reduce((sum, x) => sum + x, 0);
  const expected = [0, 1].map(column => exponents.reduce((sum, weight, row) => sum + weight * values[row][column], 0) / denominator);
  for (const tileSize of [1, 2, 3, 5]) {
    let m = -Infinity, l = 0;
    let o = [0, 0];
    for (let start = 0; start < scores.length; start += tileSize) {
      const tile = scores.slice(start, start + tileSize);
      const next = Math.max(m, ...tile);
      const ratio = Math.exp(m - next);
      const weights = tile.map(score => Math.exp(score - next));
      l = ratio * l + weights.reduce((sum, x) => sum + x, 0);
      o = o.map((value, column) => ratio * value + weights.reduce((sum, weight, row) => sum + weight * values[start + row][column], 0));
      m = next;
    }
    o.forEach((value, column) => assert.ok(Math.abs(value / l - expected[column]) < 1e-12));
  }
});
