import { describe, expect, it } from "vitest";
import { sseData } from "./sse.js";

const enc = new TextEncoder();

async function* chunks(...parts: string[]): AsyncIterable<Uint8Array> {
  for (const part of parts) yield enc.encode(part);
}

async function collect(source: AsyncIterable<Uint8Array>): Promise<string[]> {
  const out: string[] = [];
  for await (const data of sseData(source)) out.push(data);
  return out;
}

describe("sseData", () => {
  it("yields each event's data", async () => {
    expect(await collect(chunks('data: {"a":1}\n\ndata: {"a":2}\n\n'))).toEqual(['{"a":1}', '{"a":2}']);
  });

  it("handles chunks split mid-line and mid-field", async () => {
    expect(await collect(chunks("da", 'ta: {"x"', ':1}\n', "\ndata: [DO", "NE]\n\n"))).toEqual(['{"x":1}', "[DONE]"]);
  });

  it("handles a multi-byte character split across chunks", async () => {
    const bytes = enc.encode("data: café\n\n");
    async function* split(): AsyncIterable<Uint8Array> {
      yield bytes.slice(0, 10);
      yield bytes.slice(10);
    }
    expect(await collect(split())).toEqual(["café"]);
  });

  it("accepts CRLF and lone CR line endings", async () => {
    expect(await collect(chunks("data: a\r\n\r\ndata: b\r\rdata: c\n\n"))).toEqual(["a", "b", "c"]);
  });

  it("joins multi-line data with newlines", async () => {
    expect(await collect(chunks("data: line one\ndata: line two\n\n"))).toEqual(["line one\nline two"]);
  });

  it("ignores comments, other fields and empty events", async () => {
    expect(await collect(chunks(": keep-alive\n\nevent: delta\nid: 7\ndata: x\n\n\n\n"))).toEqual(["x"]);
  });

  it("removes only one space after the colon, and accepts none", async () => {
    expect(await collect(chunks("data:x\n\ndata:  y\n\n"))).toEqual(["x", " y"]);
  });

  it("delivers a last event that ends without a blank line", async () => {
    expect(await collect(chunks("data: first\n\ndata: last"))).toEqual(["first", "last"]);
  });

  it("strips a leading byte-order mark", async () => {
    expect(await collect(chunks("﻿data: x\n\n"))).toEqual(["x"]);
  });
});
