// Browser-level editor regressions. Run: node tests/editor_browser.cjs
const { chromium } = require('playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const http = require('node:http');
const { spawnSync } = require('node:child_process');

(async () => {
  const root=fs.readFileSync(path.resolve('coherent_diffusion/editor/index.html'));
  const server=http.createServer((req,res)=>{res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});res.end(root)});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const url='http://127.0.0.1:'+server.address().port+'/editor';
  const browser = await chromium.launch({ headless: true });
  // A srcdoc mount approximates the transport used by the Forge/Gradio wrapper.
  const iframeHost=await browser.newPage({viewport:{width:1500,height:1050}});
  const escaped=root.toString('utf8').replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  await iframeHost.goto(url);
  await iframeHost.setContent('<iframe title="P02 editor" style="width:100%;height:950px" srcdoc="'+escaped+'"></iframe>');
  await iframeHost.frameLocator('iframe').locator('#layerList .layer').first().waitFor();
  assert.equal(await iframeHost.frameLocator('iframe').locator('#layerList .layer').count(),5,'srcdoc iframe loads executable editor');
  await iframeHost.close();
  const page = await browser.newPage({ viewport: { width: 1500, height: 1050 }, acceptDownloads: true });
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  await page.goto(url);
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
  // A damaged mask must not replace the current scene.
  await page.locator('#importBtn').click();
  const corrupted=Buffer.from(bytes);
  const pngStart=corrupted.indexOf(Buffer.from([137,80,78,71,13,10,26,10]));
  assert.ok(pngStart>0,'PNG signature present in archive');
  corrupted[pngStart+40]^=0xff;
  await page.locator('#importFile').setInputFiles({name:'bad.crf.zip',mimeType:'application/zip',buffer:corrupted});
  await page.locator('#validateBtn').click();
  assert.equal(await page.locator('#applyImportBtn').isDisabled(),true,'corrupted ZIP cannot apply');
  await page.locator('#modalClose').click();
  assert.equal(await page.locator('#layerList .layer').count(),7,'ZIP corruption keeps previous scene');


  // Effective overlay is computed from the canonical regions, not a static illustration.
  await page.locator('[data-view="influence"]').click();
  const weightedAlpha=await page.evaluate(()=>document.getElementById('weightMap').getContext('2d').getImageData(64,10,1,1).data[3]);
  assert.ok(weightedAlpha>0,'weighted influence paints sampled mask weights');
  await page.locator('[data-view="regions"]').click();

  // Edit the re-imported editable brush mask twice; none of the other six are replaced.
  await page.locator('[data-tool="brush"]').click();
  let surface=await page.locator('#surface').boundingBox();
  await page.mouse.move(surface.x+surface.width*.6,surface.y+surface.height*.3);
  await page.mouse.down();await page.mouse.move(surface.x+surface.width*.63,surface.y+surface.height*.33,{steps:4});await page.mouse.up();
  assert.equal(await page.locator('#layerList .layer').count(),7,'painting active stroke does not add/delete siblings');
  await page.locator('[data-tool="eraser"]').click();
  await page.mouse.move(surface.x+surface.width*.55,surface.y+surface.height*.5);
  await page.mouse.down();await page.mouse.move(surface.x+surface.width*.56,surface.y+surface.height*.51,{steps:3});await page.mouse.up();
  assert.equal(await page.locator('#layerList .layer').count(),7,'eraser affects selected mask only');

  // 25 mixed, independently undoable actions with redo and no entity/LoRA corruption.
  const initialIds=await page.locator('#layerList .layer').evaluateAll(nodes=>nodes.map(n=>n.dataset.id));
  for(let step=0;step<5;step++){
    await page.locator('#addRegionBtn').click();
    await page.locator('#duplicateBtn').click();
    await page.locator('#deleteBtn').click();
    await page.locator('#regionPrompt').fill('cycle-'+step);
    await page.locator('#regionPrompt').blur();
    await page.locator('#inspector [data-toggle="visible"]').click();
  }
  assert.equal(await page.locator('#layerList .layer').count(),12);
  for(let step=0;step<25;step++)await page.locator('#undoBtn').click();
  assert.deepEqual(await page.locator('#layerList .layer').evaluateAll(nodes=>nodes.map(n=>n.dataset.id)),initialIds,'25 mixed edits undo exactly');
  for(let step=0;step<25;step++)await page.locator('#redoBtn').click();
  assert.equal(await page.locator('#layerList .layer').count(),12,'25 redo steps restore scene');
  const dims=await page.locator('#surface').boundingBox();
  assert.ok(dims.width>320&&dims.height>200,'canvas has usable geometry at desktop viewport');

  // Duplicate ID and orphan mask inputs must fail without modifying existing layers.
  await page.locator('#importBtn').click();
  const duplicate=JSON.parse(JSON.stringify(exported));duplicate.regions[1].id=duplicate.regions[0].id;
  await page.locator('#importTextarea').fill(JSON.stringify(duplicate));await page.locator('#validateBtn').click();
  assert.equal(await page.locator('#applyImportBtn').isDisabled(),true,'duplicate IDs rejected');
  await page.locator('#modalClose').click();
  assert.equal(await page.locator('#layerList .layer').count(),12,'invalid input preserved current project');


  // Check UI validator against the authoritative Python/P01 conformance corpus.
  const fixtures=JSON.parse(fs.readFileSync('tests/fixtures/crf_conformance.json','utf8'));
  await page.locator('#importBtn').click();
  for(const item of fixtures.valid){
    await page.locator('#importTextarea').fill(JSON.stringify(item.document));
    await page.locator('#validateBtn').click();
    assert.equal(await page.locator('#applyImportBtn').isDisabled(),Boolean(item.document.assets?.length),'valid fixture/paste safety disagreement: '+item.name);
  }
  for(const item of fixtures.invalid){
    await page.locator('#importTextarea').fill(JSON.stringify(item.document));
    await page.locator('#validateBtn').click();
    assert.equal(await page.locator('#applyImportBtn').isDisabled(),true,'invalid fixture accepted: '+item.name);
  }
  await page.locator('#modalClose').click();
  assert.equal(await page.locator('#layerList .layer').count(),12,'conformance previews never mutate scene');

  assert.deepEqual(errors,[],'no uncaught browser script errors');
  await browser.close();await new Promise(resolve=>server.close(resolve));
  console.log('P02 browser editor: PASS (5 regions, edit, export, Python conformance, reload, bad import, brush ZIP)');
})().catch(e=>{console.error(e);process.exit(1)});
