import { describe, expect, it } from "vitest";
import { type Block, type Inline, parseMarkdown, parseInline, safeHref } from "./parse.js";

const text = (value: string): Inline => ({ type: "text", value });
const para = (...children: Inline[]): Block => ({ type: "paragraph", children });

describe("blocks", () => {
  it("splits paragraphs on blank lines and keeps soft line breaks", () => {
    expect(parseMarkdown("one\ntwo\n\nthree")).toEqual([para(text("one"), { type: "break" }, text("two")), para(text("three"))]);
  });

  it("parses headings, rules and quotes", () => {
    expect(parseMarkdown("## Costs\n---\n> careful\n> here")).toEqual([
      { type: "heading", level: 2, children: [text("Costs")] },
      { type: "rule" },
      { type: "quote", children: [para(text("careful"), { type: "break" }, text("here"))] },
    ]);
  });

  it("keeps fenced code exactly, with its language, and parses nothing inside", () => {
    expect(parseMarkdown("```ts\nconst a = **1**;\n<script>x</script>\n```")).toEqual([
      { type: "code", lang: "ts", value: "const a = **1**;\n<script>x</script>" },
    ]);
  });

  it("treats an unclosed fence as code to the end, not as text", () => {
    expect(parseMarkdown("```\nstill code")).toEqual([{ type: "code", lang: "", value: "still code" }]);
  });

  it("parses bullet and numbered lists, with nesting by indent", () => {
    expect(parseMarkdown("- a\n- b\n  - b1\n3. x\n4. y")).toEqual([
      {
        type: "list",
        ordered: false,
        start: 1,
        items: [[para(text("a"))], [para(text("b")), { type: "list", ordered: false, start: 1, items: [[para(text("b1"))]] }]],
      },
      { type: "list", ordered: true, start: 3, items: [[para(text("x"))], [para(text("y"))]] },
    ]);
  });

  it("parses a table with a header row", () => {
    expect(parseMarkdown("| Item | Cost |\n|---|---:|\n| Oven | 900 |")).toEqual([
      { type: "table", head: [[text("Item")], [text("Cost")]], rows: [[[text("Oven")], [text("900")]]] },
    ]);
  });
});

describe("inline", () => {
  it("parses strong, emphasis and code, and leaves code spans literal", () => {
    expect(parseInline("a **b** _c_ `**d**`")).toEqual([
      text("a "),
      { type: "strong", children: [text("b")] },
      text(" "),
      { type: "em", children: [text("c")] },
      text(" "),
      { type: "code", value: "**d**" },
    ]);
  });

  it("makes links only to http, https and mailto, and shows anything else as text", () => {
    expect(parseInline("[site](https://example.com) and [x](javascript:alert(1))")).toEqual([
      { type: "link", href: "https://example.com", children: [text("site")] },
      text(" and [x](javascript:alert(1))"),
    ]);
  });

  it("links bare URLs, without the trailing punctuation", () => {
    expect(parseInline("see https://example.com/a.")).toEqual([text("see "), { type: "link", href: "https://example.com/a", children: [text("https://example.com/a")] }, text(".")]);
  });

  it("leaves HTML as plain text: nothing in Markdown becomes markup", () => {
    expect(parseInline('<img src=x onerror="alert(1)">')).toEqual([text('<img src=x onerror="alert(1)">')]);
  });

  it("leaves unmatched markers alone", () => {
    expect(parseInline("2 * 3 and a_b_c")).toEqual([text("2 * 3 and a_b_c")]);
  });
});

describe("safeHref", () => {
  it("accepts only absolute http, https and mailto", () => {
    expect(safeHref("https://a.example/x")).toBe("https://a.example/x");
    expect(safeHref("mailto:a@b.example")).toBe("mailto:a@b.example");
    for (const bad of ["javascript:alert(1)", "JaVaScRiPt:alert(1)", "data:text/html,x", "vbscript:x", "/relative", "//evil.example", " java\tscript:x"]) {
      expect(safeHref(bad), bad).toBeUndefined();
    }
  });
});

describe("robustness", () => {
  it("never throws or hangs on random Markdown-like input", () => {
    const pieces = ["*", "**", "_", "`", "```", "[", "](", ")", "(", "|", "---", "> ", "- ", "1. ", "  ", "\n", "\n\n", "#", "a", "b c", "https://x.example", "javascript:", "<b>"];
    let seed = 42;
    const random = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
    const started = Date.now();
    for (let n = 0; n < 3000; n++) {
      let input = "";
      const length = Math.floor(random() * 40);
      for (let k = 0; k < length; k++) input += pieces[Math.floor(random() * pieces.length)];
      expect(() => parseMarkdown(input), JSON.stringify(input)).not.toThrow();
    }
    expect(Date.now() - started).toBeLessThan(5000);
  });
});
