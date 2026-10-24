// Browser-level editor regressions. Run: node tests/editor_browser.cjs
const { chromium } = require('playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { spawnSync } = require('node:child_process');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1050 }, acceptDownloads: true });
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  await page.goto(pathToFileURL(path.resolve('coherent_diffusion/editor/index.html')).href);
  await page.waitForSelector('#layerList .layer');
  assert.equal(await page.locator('#layerList .layer').count(), 5, 'initial overlapping 5 regions');
  for(const region of ['sky','sun','ocean','beach','clouds']){
    assert.ok(await page.locator('#overlay [data-region="'+region+'"]').count(), region+' remains visible');
  }
  await page.locator('#addRegionBtn').click();
  assert.equal(await page.locator('#layerList .layer').count(), 6);
  await page.locator('#regionPrompt').fill('an isolated test prompt');
  await page.locator('#regionPrompt').blur();
  const download = page.waitForEvent('download');
  await page.locator('#exportBtn').click();
  const d = await download;
  assert.ok(d.suggestedFilename().endsWith('.json'));
  const exported = JSON.parse(fs.readFileSync(await d.path(),'utf8'));
  assert.equal(exported.regions.length,6);
  assert.ok(exported.regions.some(r=>r.prompt==='an isolated test prompt'));
  assert.ok(exported.regions.every(r=>!Object.hasOwn(r,'ui')));
  assert.ok(exported.regions.every(r=>r.extensions?.['cd.color']));
  const python = spawnSync('python',['-c',
    'import sys; from coherent_diffusion.crf import parse_validate; parse_validate(open(sys.argv[1],encoding="utf-8").read()); print("Python CRF validator PASS")',
    await d.path()],{encoding:'utf8'});
  assert.equal(python.status,0,python.stderr);
  const baseline = JSON.stringify(exported.regions.map(r=>r.id));
  await page.reload();
  await page.waitForSelector('#layerList .layer');
  assert.equal(await page.locator('#layerList .layer').count(),6,'draft survives reload');
  await page.locator('#importBtn').click();
  await page.locator('#importTextarea').fill(JSON.stringify(exported));
  await page.locator('#validateBtn').click();
  assert.equal(await page.locator('#applyImportBtn').isDisabled(),false);
  await page.locator('#applyImportBtn').click();
  assert.equal(await page.locator('#layerList .layer').count(),6,'replace preserves region count');
  await page.locator('#importBtn').click();
  const corrupt = JSON.stringify(exported).replace('"format":"coherent-regions"', '"format":"wrong"');
  await page.locator('#importTextarea').fill(corrupt);
  await page.locator('#validateBtn').click();
  assert.equal(await page.locator('#applyImportBtn').isDisabled(),true,'wrong version/format blocked');
  await page.locator('#modalClose').click();
  assert.equal(await page.locator('#layerList .layer').count(),6,'invalid import is non-destructive');

  // Brush must export real image bytes in a .crf.zip, not invalid "stroke" CRF geometry.
  await page.locator('[data-tool="brush"]').click();
  const rect = await page.locator('#surface').boundingBox();
  const x=rect.x+rect.width*.50, y=rect.y+rect.height*.50;
  await page.mouse.move(x,y);await page.mouse.down();await page.mouse.move(x+65,y+40,{steps:8});await page.mouse.up();
  assert.equal(await page.locator('#layerList .layer').count(),7);
  const zipDownload = page.waitForEvent('download');
  await page.locator('#exportBtn').click();
  const zip = await zipDownload;
  assert.ok(zip.suggestedFilename().endsWith('.zip'));
  const bytes = fs.readFileSync(await zip.path());
  const archiveTest = spawnSync('python',['-c',
    'import sys; from coherent_diffusion.crf.assets import load_crf_zip; a=load_crf_zip(open(sys.argv[1],"rb").read()); assert len(a.assets)==1; print("verified ZIP mask PASS")',
    await zip.path()],{encoding:'utf8'});
  assert.equal(archiveTest.status,0,archiveTest.stderr);
  await page.reload();await page.waitForSelector('#layerList .layer');
  assert.equal(await page.locator('#layerList .layer').count(),7,'brush persists across reload');
  await page.locator('#importBtn').click();
  await page.locator('#importFile').setInputFiles({name:'scene.crf.zip',mimeType:'application/zip',buffer:bytes});
  await page.locator('#validateBtn').click();
  await page.waitForTimeout(600);
  assert.equal(await page.locator('#applyImportBtn').isDisabled(),false,'ZIP verified before apply');
  await page.locator('#applyImportBtn').click();
  assert.equal(await page.locator('#layerList .layer').count(),7,'masked CRF round-trip');
  assert.deepEqual(errors,[],'no uncaught browser script errors');
  await browser.close();
  console.log('P02 browser editor: PASS (5 regions, edit, export, Python conformance, reload, bad import, brush ZIP)');
})().catch(e=>{console.error(e);process.exit(1)});
