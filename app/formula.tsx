import { memo } from "react";
import katex from "katex";

type FormulaProps = {
  tex: string;
  block?: boolean;
  className?: string;
};

// Authored formulas must fail validation at build time, not silently show broken TeX.
export const Formula = memo(function Formula({ tex, block = false, className = "" }: FormulaProps) {
  const html = katex.renderToString(tex, {
    displayMode: block,
    throwOnError: true,
    strict: "error",
    output: "htmlAndMathml",
  });
  const Tag = block ? "div" : "span";
  return (
    <Tag
      className={`math ${block ? "math-block" : "math-inline"} ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
});
