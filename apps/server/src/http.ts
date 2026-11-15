// Small HTTP helpers shared by the daemon's routes.
import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

const MAX_BODY = 1_000_000;

export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export function sameSecret(given: string | string[] | undefined, expected: Buffer): boolean {
  if (typeof given !== "string") return false;
  const buffer = Buffer.from(given);
  return buffer.length === expected.length && timingSafeEqual(buffer, expected);
}

export function securityHeaders(res: ServerResponse, scriptHashes: string[] = []): void {
  res.setHeader(
    "Content-Security-Policy",
    [
      "default-src 'self'",
      `script-src 'self'${scriptHashes.map((h) => ` ${h}`).join("")}`,
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data:",
      "font-src 'self'",
      // ipc: is how the Tauri window reaches the shell's native dialogs.
      "connect-src 'self' ipc: http://ipc.localhost",
      "object-src 'none'",
      "base-uri 'none'",
      "form-action 'self'",
      "frame-ancestors 'none'",
    ].join("; "),
  );
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("Cache-Control", "no-store");
}

export function sendJson(res: ServerResponse, status: number, body: unknown): void {
  securityHeaders(res);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

export async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > MAX_BODY) throw new HttpError(413, "request body too large");
    chunks.push(chunk as Buffer);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new HttpError(400, "body is not JSON");
  }
}

