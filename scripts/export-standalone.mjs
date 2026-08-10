import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const clientDir = resolve(root, "dist/client");
const assetsDir = resolve(clientDir, "assets");

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("standalone", Date.now().toString());
const { default: worker } = await import(workerUrl.href);
const response = await worker.fetch(
  new Request("http://localhost/", { headers: { accept: "text/html" } }),
  { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
  { waitUntil() {}, passThroughOnException() {} },
);

if (!response.ok) throw new Error(`Render failed: ${response.status}`);

let html = await response.text();
const cssName = (await readdir(assetsDir)).find((name) => /^index-.*\.css$/.test(name));
if (!cssName) throw new Error("Built CSS was not found");
let css = await readFile(resolve(assetsDir, cssName), "utf8");

for (const match of [...css.matchAll(/url\(\/assets\/([^)]*\.woff2)\)/g)]) {
  const font = await readFile(resolve(assetsDir, match[1]));
  css = css.replaceAll(match[0], `url(data:font/woff2;base64,${font.toString("base64")})`);
}
css = css.replace(/,url\(\/assets\/[^)]*\.(?:woff|ttf)\)format\("(?:woff|truetype)"\)/g, "");

const standaloneJs = String.raw`
(() => {
  const sections = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"];
  const navLinks = [...document.querySelectorAll(".sidenav a")];
  const progress = document.querySelector(".progress");

  const syncScroll = () => {
    const height = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = (height > 0 ? scrollY / height * 100 : 0) + "%";
    const y = scrollY + 130;
    let active = sections[0];
    for (const id of sections) {
      const section = document.getElementById(id);
      if (section && section.offsetTop <= y) active = id;
    }
    for (const link of navLinks) link.classList.toggle("active", link.hash === "#" + active);
  };
  addEventListener("scroll", syncScroll, { passive: true });
  syncScroll();

  const matrixA = [[1, 2, 0], [0, 1, 3]];
  const matrixB = [[2, 1], [1, 0], [0, 2]];
  const matrixC = [[4, 1], [1, 6]];
  const matrixCols = [...document.querySelectorAll("#s1 .mcol")];
  const resultButtons = [...document.querySelectorAll("#s1 button.mcell.res")];
  const calculation = document.querySelector("#s1 .mcalc");
  const selectCell = (row, col) => {
    [...matrixCols[0].querySelectorAll(".mcell")].forEach((cell, index) => cell.classList.toggle("hl-row", Math.floor(index / 3) === row));
    [...matrixCols[1].querySelectorAll(".mcell")].forEach((cell, index) => cell.classList.toggle("hl-col", index % 2 === col));
    resultButtons.forEach((cell, index) => cell.classList.toggle("hl-res", index === row * 2 + col));
    const terms = matrixA[row].map((value, index) => "(" + value + "×" + matrixB[index][col] + ")").join(" + ");
    calculation.innerHTML = "C[" + row + "][" + col + "] = " + terms + " = <b>" + matrixC[row][col] + "</b>";
  };
  resultButtons.forEach((button, index) => button.addEventListener("click", () => selectCell(Math.floor(index / 2), index % 2)));

  const queries = [[0.04, 1.16], [1.41, 0.99], [0.53, 1.12], [0.65, 1.75]];
  const keys = [[0.56, 1.16], [1.26, -0.27], [0.82, 0.67], [1.18, 1.21]];
  const values = [[0.4, 1.28], [1.08, 0.6], [0.65, 1.06], [0.92, 1.72]];
  const queryTabs = [...document.querySelectorAll("#s4 .tabs .tab")];
  const queryCells = [...document.querySelectorAll("#s4 .tabs + .card .mcell")];
  const softmax = (items) => {
    const max = Math.max(...items);
    const exponents = items.map((value) => Math.exp(value - max));
    const sum = exponents.reduce((a, b) => a + b, 0);
    return exponents.map((value) => value / sum);
  };
  const selectQuery = (index) => {
    queryTabs.forEach((tab, tabIndex) => tab.classList.toggle("active", tabIndex === index));
    const q = queries[index];
    const scaled = keys.map((key) => (q[0] * key[0] + q[1] * key[1]) / Math.sqrt(q.length));
    const weights = softmax(scaled);
    const output = values[0].map((_, dimension) => weights.reduce((sum, weight, row) => sum + weight * values[row][dimension], 0));
    scaled.forEach((value, cell) => { queryCells[cell].textContent = value.toFixed(2); });
    weights.forEach((value, cell) => {
      const target = queryCells[cell + 4];
      target.textContent = (value * 100).toFixed(0) + "%";
      target.style.backgroundColor = "rgba(56,189,248," + (0.12 + value * 0.6).toFixed(3) + ")";
      target.style.borderColor = "#38bdf8";
    });
    output.forEach((value, cell) => { queryCells[cell + 8].textContent = value.toFixed(2); });
  };
  queryTabs.forEach((button, index) => button.addEventListener("click", () => selectQuery(index)));

  const heads = [
    { note: "权重大量出现在非对角线位置 → 可能形成长距离关注。", matrix: [[0.58,0.12,0.22,0.08],[0.18,0.16,0.12,0.54],[0.66,0.08,0.20,0.06],[0.14,0.46,0.09,0.31]] },
    { note: "权重集中在主对角线附近 → 可能形成局部关注。", matrix: [[0.62,0.27,0.07,0.04],[0.24,0.48,0.22,0.06],[0.07,0.24,0.48,0.21],[0.03,0.08,0.30,0.59]] },
    { note: "权重分布较平均 → 可能形成全局关注。", matrix: [[0.28,0.24,0.25,0.23],[0.23,0.29,0.22,0.26],[0.27,0.20,0.30,0.23],[0.22,0.27,0.21,0.30]] },
  ];
  const headTabs = [...document.querySelectorAll("#s5 .tabs .tab")];
  const headCells = [...document.querySelectorAll("#s5 .fig table tr:not(:first-child) td div")];
  const headCaption = document.querySelector("#s5 .fig .fig-cap");
  const selectHead = (index) => {
    headTabs.forEach((tab, tabIndex) => tab.classList.toggle("active", tabIndex === index));
    heads[index].matrix.flat().forEach((value, cell) => {
      headCells[cell].textContent = value.toFixed(2);
      headCells[cell].style.background = "rgba(56,189,248," + (0.08 + value * 0.7).toFixed(3) + ")";
      headCells[cell].style.color = value > 0.4 ? "#fff" : "#a9b4dc";
    });
    headCaption.innerHTML = "<b>人工示意</b>（非真实训练结果）：" + heads[index].note + " · 行=Query（谁在问）· 列=Key（看谁）";
  };
  headTabs.forEach((button, index) => button.addEventListener("click", () => selectHead(index)));
})();
`;

html = html
  .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, "")
  .replace(/<link\b(?=[^>]*rel="(?:stylesheet|modulepreload)")[^>]*\/?\s*>/gi, "");
html = html.slice(0, html.indexOf("</html>") + 7);
html = html.replace("</head>", `<style>${css}</style></head>`);
html = html.replace("</body>", `<script data-standalone="attention">${standaloneJs}</script></body>`);
html = html.replace("<!DOCTYPE html>", "<!DOCTYPE html><!-- Offline standalone export: double-click to open -->");

await writeFile(resolve(root, "attention.html"), html);
console.log(`Wrote attention.html (${Math.round(Buffer.byteLength(html) / 1024)} KiB)`);
