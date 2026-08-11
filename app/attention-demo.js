/** @typedef {number[][]} Matrix */

/** @param {Matrix} left @param {Matrix} right @returns {Matrix} */
function matMul(left, right) {
  return left.map((row) =>
    right[0].map((_, column) =>
      row.reduce((sum, value, index) => sum + value * right[index][column], 0),
    ),
  );
}

/** @param {Matrix} matrix @returns {Matrix} */
function transpose(matrix) {
  return matrix[0].map((_, column) => matrix.map((row) => row[column]));
}

/** @param {number[]} values @returns {number[]} */
function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

/** @param {Matrix} matrix @returns {Matrix} */
function cleanMatrix(matrix) {
  return matrix.map((row) =>
    row.map((value) => Number(value.toFixed(12))),
  );
}

/** @param {Matrix} matrix @param {number} start @param {number} end @returns {Matrix} */
function sliceColumns(matrix, start, end) {
  return matrix.map((row) => row.slice(start, end));
}

const X = [
  [0.4, 1.2],
  [1.5, 0.3],
  [0.8, 0.9],
  [1.1, 1.5],
];
const WQ = [
  [1, 0.5],
  [-0.3, 0.8],
];
const WK = [
  [0.8, -0.4],
  [0.2, 1.1],
];
const WV = [
  [0.7, 0.2],
  [0.1, 1],
];

const Q = cleanMatrix(matMul(X, WQ));
const K = cleanMatrix(matMul(X, WK));
const V = cleanMatrix(matMul(X, WV));
const scale = 1 / Math.sqrt(K[0].length);
const S = cleanMatrix(matMul(Q, transpose(K)).map((row) =>
  row.map((value) => value * scale),
));
const A = cleanMatrix(S.map(softmax));
const O = cleanMatrix(matMul(A, V));

export const attentionDemo = { X, WQ, WK, WV, Q, K, V, S, A, O };

/*
 * Multi-Head teaching example: four tokens, d_model=6, three heads, d_k=d_v=2.
 * The weights are deliberately small fixed values so every intermediate matrix
 * remains readable; real models learn these matrices during training.
 */
const multiHeadX = [
  [2, 0, 2, 0, 1, 0],
  [0, 1, 1, 1, 1, 1],
  [2, 0, 0, 2, 1, 2],
  [0, 1, -1, 1, 1, -1],
];
const multiHeadWQ = [
  [1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1],
];
const multiHeadWK = [
  [1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 1, 0],
];
const multiHeadWV = [
  [1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1],
];
const multiHeadWO = [
  [1, 0, 0.25, 0, 0.1, 0],
  [0, 1, 0, 0.25, 0, 0.1],
  [0.5, 0, 1, 0, 0.25, 0],
  [0, 0.5, 0, 1, 0, 0.25],
  [0.25, 0, 0.5, 0, 1, 0],
  [0, 0.25, 0, 0.5, 0, 1],
];

const multiHeadQ = cleanMatrix(matMul(multiHeadX, multiHeadWQ));
const multiHeadK = cleanMatrix(matMul(multiHeadX, multiHeadWK));
const multiHeadV = cleanMatrix(matMul(multiHeadX, multiHeadWV));

function calculateHead(headIndex) {
  const start = headIndex * 2;
  const query = sliceColumns(multiHeadQ, start, start + 2);
  const key = sliceColumns(multiHeadK, start, start + 2);
  const value = sliceColumns(multiHeadV, start, start + 2);
  const score = cleanMatrix(matMul(query, transpose(key)).map((row) =>
    row.map((entry) => entry / Math.sqrt(2)),
  ));
  const weight = cleanMatrix(score.map(softmax));
  const output = cleanMatrix(matMul(weight, value));
  return { Q: query, K: key, V: value, S: score, A: weight, H: output };
}

const multiHeadHeads = Array.from({ length: 3 }, (_, index) => calculateHead(index));
const multiHeadConcat = cleanMatrix(multiHeadX.map((_, rowIndex) =>
  multiHeadHeads.flatMap((head) => head.H[rowIndex]),
));
const multiHeadOutput = cleanMatrix(matMul(multiHeadConcat, multiHeadWO));

export const multiHeadDemo = {
  X: multiHeadX,
  WQ: multiHeadWQ,
  WK: multiHeadWK,
  WV: multiHeadWV,
  WO: multiHeadWO,
  Q: multiHeadQ,
  K: multiHeadK,
  V: multiHeadV,
  heads: multiHeadHeads,
  H: multiHeadConcat,
  Y: multiHeadOutput,
};

const warmupA = [
  [1, 2, 0],
  [0, 1, 3],
];
const warmupB = [
  [2, 1],
  [1, 0],
  [0, 2],
];

export const matrixMultiplicationDemo = {
  A: warmupA,
  B: warmupB,
  C: matMul(warmupA, warmupB),
};

export const attentionHeads = [
  {
    name: "Head 1 · 分组",
    note: "Token 1/3 与 Token 2/4 分成两组，体现按特征相似性建立联系。",
    matrix: multiHeadHeads[0].A,
  },
  {
    name: "Head 2 · 方向",
    note: "不同 Query 根据二维方向匹配不同 Key，四行呈现不同的关注分布。",
    matrix: multiHeadHeads[1].A,
  },
  {
    name: "Head 3 · 汇聚",
    note: "四个 Query 都更关注 Token 3，体现一个头可以把信息汇聚到共同位置。",
    matrix: multiHeadHeads[2].A,
  },
];
