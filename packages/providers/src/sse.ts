// Server-sent events: bytes in, each event's data out. Follows the WHATWG
// event-stream rules that matter for model APIs: lines end in LF, CRLF or CR;
// "data:" lines are joined with "\n"; one space after the colon is dropped;
// comment lines (":") and other fields are ignored; a blank line ends an event.
// Unlike the browser, a final event without a closing blank line is still
// delivered, because some providers end the stream that way.

export async function* sseData(source: AsyncIterable<Uint8Array>): AsyncIterable<string> {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let data: string[] = [];
  let first = true;

  function* takeLines(final: boolean): Iterable<string> {
    for (;;) {
      const match = /\r\n|\r|\n/.exec(buffer);
      // A trailing "\r" may be the first half of "\r\n": wait for more input.
      if (match === null || (!final && match[0] === "\r" && match.index === buffer.length - 1)) break;
      yield buffer.slice(0, match.index);
      buffer = buffer.slice(match.index + match[0].length);
    }
    if (final && buffer !== "") {
      yield buffer;
      buffer = "";
    }
  }

  function* feed(final: boolean): Iterable<string> {
    for (const line of takeLines(final)) {
      if (line === "") {
        if (data.length > 0) yield data.join("\n");
        data = [];
      } else if (line.startsWith("data:")) {
        const value = line.slice(5);
        data.push(value.startsWith(" ") ? value.slice(1) : value);
      }
    }
    if (final && data.length > 0) {
      yield data.join("\n");
      data = [];
    }
  }

  for await (const chunk of source) {
    buffer += decoder.decode(chunk, { stream: true });
    if (first && buffer !== "") {
      if (buffer.charCodeAt(0) === 0xfeff) buffer = buffer.slice(1);
      first = false;
    }
    yield* feed(false);
  }
  buffer += decoder.decode();
  yield* feed(true);
}
