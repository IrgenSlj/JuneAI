#!/usr/bin/env node
/**
 * Rasterise the SVG brand assets to the fixed-size PNGs some surfaces require.
 *
 * GitHub's social preview must be a raster image at 1280x640, and app icons
 * must be PNG at exact sizes, so the SVG sources in assets/ need a renderer.
 * This uses the Playwright Chromium that already backs the e2e suite rather
 * than adding an image library — the working agreement is deliberately hostile
 * to dependencies that an existing tool can cover.
 *
 * Usage:
 *   node tools/render-assets.mjs
 */

import { readFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// ESM resolves bare imports relative to *this file*, and tools/ has no
// node_modules of its own under pnpm's strict layout. Resolve the browser
// driver from the web workspace that actually declares it, so the script can be
// run from anywhere without a wrapper.
// `@playwright/test` is the declared dependency, but the browser driver lives in
// the `playwright` package it depends on, so hop through it: resolve the test
// package from the web workspace, then resolve `playwright` from there.
const requireFromWeb = createRequire(resolve(ROOT, "apps/web/package.json"));
const requireFromTest = createRequire(requireFromWeb.resolve("@playwright/test"));
const playwright = await import(
  pathToFileURL(requireFromTest.resolve("playwright")).href
);
const { chromium } = playwright.default ?? playwright;

/** @type {{src: string, out: string, width: number, height: number}[]} */
const TARGETS = [
  {
    src: "assets/social-preview.svg",
    out: "assets/social-preview.png",
    width: 1280,
    height: 640,
  },
  // The hero renders as SVG in the README; these PNGs exist for surfaces that
  // reject SVG (release banners, link unfurls that are not the social card).
  { src: "assets/hero-dark.svg", out: "assets/hero-dark.png", width: 1600, height: 420 },
  { src: "assets/hero-light.svg", out: "assets/hero-light.png", width: 1600, height: 420 },
];

async function render(browser, target) {
  const svg = await readFile(resolve(ROOT, target.src), "utf8");
  const page = await browser.newPage({
    viewport: { width: target.width, height: target.height },
    deviceScaleFactor: 1,
  });
  // The SVG is inlined into a page sized exactly to the output so the
  // screenshot is pixel-exact rather than scaled after the fact.
  await page.setContent(
    `<!doctype html><meta charset="utf-8">
     <style>
       html,body{margin:0;padding:0;background:transparent}
       svg{display:block;width:${target.width}px;height:${target.height}px}
     </style>
     ${svg}`,
    { waitUntil: "load" },
  );
  await mkdir(dirname(resolve(ROOT, target.out)), { recursive: true });
  await page.screenshot({ path: resolve(ROOT, target.out) });
  await page.close();
  console.log(`rendered ${target.out} (${target.width}x${target.height})`);
}

const browser = await chromium.launch();
try {
  for (const target of TARGETS) {
    await render(browser, target);
  }
} finally {
  await browser.close();
}
