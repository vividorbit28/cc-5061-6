// Browser smoke test: load the studio, pick a clip, inspect the player, run an
// inversion, and verify the result tabs are populated. Saves screenshots.
import pw from '/workspace/.tools/pw/node_modules/playwright/index.js';
const { chromium } = pw;
import fs from 'node:fs';

process.env.LD_LIBRARY_PATH = [
  '/workspace/.tools/syslibs/root/usr/lib/aarch64-linux-gnu',
  '/workspace/.tools/syslibs/root/lib/aarch64-linux-gnu',
  process.env.LD_LIBRARY_PATH || '',
].join(':');
const BASE = process.env.BASE || 'http://localhost:5173';
const OUT = '/workspace/test-output';
fs.mkdirSync(OUT, { recursive: true });

const errors = [];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1680, height: 1050 } });
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));

await page.goto(BASE, { waitUntil: 'networkidle' });
await page.waitForSelector('.clip-item');
await page.screenshot({ path: `${OUT}/01-load.png`, fullPage: true });

// Pick the diagonal clip.
await page.click('.clip-item[data-id="diag"]');
await page.waitForTimeout(600);
await page.screenshot({ path: `${OUT}/02-diag-selected.png`, fullPage: true });

// Play a bit, then pause.
await page.click('#playToggle');
await page.waitForTimeout(900);
await page.click('#playToggle');

// Show the clean reference in the player (truth, player-only).
await page.check('#showClean');
await page.waitForTimeout(250);
await page.screenshot({ path: `${OUT}/03-player-clean.png`, fullPage: true });
await page.uncheck('#showClean');

// Registration readout.
const reg = await page.textContent('#regCandidate');
console.log('registration readout:', reg.trim());

// Set an assumption and invert.
await page.selectOption('#kind', 'linear');
await page.fill('#vx', '6.5');
await page.dispatchEvent('#vx', 'input');
await page.fill('#vy', '3.8');
await page.dispatchEvent('#vy', 'input');
await page.check('#compare');
await page.click('#btnInvert');
await page.waitForFunction(() => {
  const t = document.querySelector('#jobStage');
  return t && (t.textContent === 'done' || t.textContent === 'failed');
}, { timeout: 120000 });
await page.waitForSelector('#resultWrap:not(.hidden)', { timeout: 30000 });
await page.waitForTimeout(500);
await page.screenshot({ path: `${OUT}/04-result-estimate.png`, fullPage: true });

// Residual tab.
await page.click('.tab[data-tab="residual"]');
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/05-residual.png`, fullPage: true });
const residMetrics = await page.textContent('#residualMetrics');
console.log('residual metrics:', residMetrics.replace(/\s+/g, ' ').trim());

// Support tab.
await page.click('.tab[data-tab="support"]');
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/06-support.png`, fullPage: true });

// Kernel tab.
await page.click('.tab[data-tab="kernel"]');
await page.waitForTimeout(300);
await page.screenshot({ path: `${OUT}/07-kernel.png`, fullPage: true });

// Hypotheses tab.
await page.click('.tab[data-tab="hyp"]');
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/08-hypotheses.png`, fullPage: true });
const hypText = await page.textContent('#hypBody');
console.log('hypotheses:', hypText.replace(/\s+/g, ' ').slice(0, 400));

// Evaluation tab.
await page.click('.tab[data-tab="eval"]');
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/09-eval.png`, fullPage: true });

// Failure path: a wildly wrong velocity should still run but show a large
// forward residual; a broken request should surface an error.
await page.selectOption('#kind', 'linear');
await page.fill('#vx', '0');
await page.dispatchEvent('#vx', 'input');
await page.fill('#vy', '0');
await page.dispatchEvent('#vy', 'input');
await page.selectOption('#kind', 'stationary');
await page.click('#btnInvert');
await page.waitForFunction(() => {
  const t = document.querySelector('#jobStage');
  return t && (t.textContent === 'done' || t.textContent === 'failed');
}, { timeout: 120000 });
await page.click('.tab[data-tab="residual"]');
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/10-wrong-assumption.png`, fullPage: true });
const wrong = await page.textContent('#residualNote');
console.log('wrong-assumption note:', wrong.replace(/\s+/g, ' ').slice(0, 300));

// Mobile / narrow layout.
await page.setViewportSize({ width: 900, height: 1100 });
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/11-narrow.png`, fullPage: true });

console.log('console errors:', errors.length ? errors : 'none');
await browser.close();
if (errors.length) process.exit(2);
