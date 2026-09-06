import { readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { observeReadingProgress } from "../app/reading-progress.js";

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
  (${observeReadingProgress.toString()})();

  const matrixCols = [...document.querySelectorAll("#s1 .mcol")];
  const resultButtons = [...document.querySelectorAll("#s1 button.mcell.res")];
  const matrixCalculations = [...document.querySelectorAll("#s1 .matrix-calc-option")];
  const selectCell = (row, col) => {
    [...matrixCols[0].querySelectorAll(".mcell")].forEach((cell, index) => cell.classList.toggle("hl-row", Math.floor(index / 3) === row));
    [...matrixCols[1].querySelectorAll(".mcell")].forEach((cell, index) => cell.classList.toggle("hl-col", index % 2 === col));
    resultButtons.forEach((cell, index) => { cell.classList.toggle("hl-res", index === row * 2 + col); cell.setAttribute("aria-pressed", String(index === row * 2 + col)); });
    matrixCalculations.forEach((formula, index) => formula.classList.toggle("active", index === row * 2 + col));
  };
  resultButtons.forEach((button, index) => button.addEventListener("click", () => selectCell(Math.floor(index / 2), index % 2)));

  const attentionDemo = document.querySelector("#s4 .attention-demo-card");
  const { S, A, O } = JSON.parse(attentionDemo.dataset.attention);
  const queryTabs = [...document.querySelectorAll("#s4 .tabs .tab")];
  const queryCells = [...attentionDemo.querySelectorAll(".mcell")];
  const queryTitles = [...document.querySelectorAll("#s4 .query-title-option")];
  const selectQuery = (index) => {
    queryTabs.forEach((tab, tabIndex) => { tab.classList.toggle("active", tabIndex === index); tab.setAttribute("aria-pressed", String(tabIndex === index)); });
    queryTitles.forEach((title) => title.classList.toggle("active", Number(title.dataset.queryIndex) === index));
    const scaled = S[index], weights = A[index], output = O[index];
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

  const headDemo = document.querySelector("#s5 .head-demo");
  const heads = JSON.parse(headDemo.dataset.heads);
  const headTabs = [...document.querySelectorAll("#s5 .tabs .tab")];
  const headCells = [...headDemo.querySelectorAll("table tr:not(:first-child) td div")];
  const headCaption = headDemo.querySelector(".fig-cap b");
  const selectHead = (index) => {
    headTabs.forEach((tab, tabIndex) => { tab.classList.toggle("active", tabIndex === index); tab.setAttribute("aria-pressed", String(tabIndex === index)); });
    heads[index].matrix.flat().forEach((value, cell) => {
      headCells[cell].textContent = value.toFixed(2);
      headCells[cell].style.background = "rgba(56,189,248," + (0.08 + value * 0.7).toFixed(3) + ")";
      headCells[cell].style.color = value > 0.4 ? "#fff" : "#a9b4dc";
    });
    headCaption.textContent = heads[index].name + " 的数值链计算结果";
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
