// Browser verification of real Gradio 4 HTML rendering (not a Forge model test).
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const assert=require('node:assert/strict');
const fs=require('node:fs');

(async()=>{
 const server=spawn('python',['tests/gradio_mount_smoke.py'],{stdio:['ignore','pipe','pipe']});
 let log='';server.stdout.on('data',x=>log+=x.toString());server.stderr.on('data',x=>log+=x.toString());
 try{
  const url='http://127.0.0.1:18746';
  let ready=false;
  for(let i=0;i<80;i++){
   if(server.exitCode!==null)break;
   try{const r=await fetch(url,{signal:AbortSignal.timeout(800)});if(r.ok){ready=true;break}}catch{}
   await new Promise(resolve=>setTimeout(resolve,350));
  }
  assert.ok(ready,'Gradio host did not start: '+log.slice(-4000));
  const browser=await chromium.launch({headless:true});
  try{
   const page=await browser.newPage({viewport:{width:1536,height:1100},acceptDownloads:true});
   const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(url,{waitUntil:'domcontentloaded'});
   await page.getByText('Coherent Diffusion', {exact:true}).first().click();
   const frame=page.frameLocator('iframe[title="Coherent Diffusion region editor"]');
   await frame.locator('#layerList .layer').first().waitFor({timeout:20000});
   assert.equal(await frame.locator('#layerList .layer').count(),5,'P02 editor mounted under Gradio');
   await frame.locator('#addRegionBtn').click();
   assert.equal(await frame.locator('#layerList .layer').count(),6,'Gradio-mounted iframe accepts interactions');
   // Real Gradio numbers must receive the edited HTML controls, not defaults.
   await frame.locator('#modelSelect').selectOption('anima');
   await frame.locator('#widthInput').fill('2048');await frame.locator('#widthInput').blur();
   await frame.locator('#heightInput').fill('1536');await frame.locator('#heightInput').blur();
   await frame.locator('[data-view="tiles"]').click();
   assert.match(await frame.locator('#tileStats').innerText(),/Tiles: 20 \\(5 × 4\\)/);
   await frame.locator('#advancedBtn').click();
   await frame.locator('#advanced_steps').fill('9');
   await frame.locator('#advanced_cfg').fill('1.25');
   await frame.locator('#advanced_overlap').fill('192');
   await frame.locator('#advanced_halo').fill('128');
   await frame.locator('#advanced_seed').fill('666');
   await frame.locator('#advanced_cpu_mib').fill('2048');
   await frame.locator('#advanced_window_mib').fill('256');
   await frame.locator('#closeAdvanced').click();
   assert.match(await frame.locator('#tileStats').innerText(),/Tiles: 30 \\(6 × 5\\)/);
   // Actual editor-to-Gradio bridge, not a simulated frame-alone editor event.
   await frame.locator('#generateBtn').click();
   try {
     await page.waitForFunction(() => {
       const input=document.querySelector('#cd_scene_json textarea');
       return Boolean(input && input.value.includes('"coherent-regions"'));
     }, null, {timeout:7000});
   } catch(error) {
     const state = await page.evaluate(() => ({
       bridgeInstalled: Boolean(window.__coherentDiffusionBridgeInstalled),
       inputs: [...document.querySelectorAll('[id^="cd_"]')].map(e => ({
         id:e.id, tag:e.tagName, textareaCount:e.querySelectorAll('textarea').length,
         value:(e.querySelector('textarea,input')?.value||'').slice(0,100)
       })).slice(0,12)
     }));
     const toasts = await frame.locator('#toasts').innerText();
     console.error('P07 bridge diagnostics',JSON.stringify({state,toasts,errors}));
     throw error;
   }
   assert.equal(await page.locator('button#cd_submit, #cd_submit button').count(),1,
     'one native Gradio Generate control is registered');
   // The prior browser smoke never actually submitted the Python event.
   // Drive the editor transfer through real Gradio queue transport. A
   // backend_fn:false dependency causes "function has no backend method".
   const queueFailures=[];
   const deps=await page.evaluate(async()=>{
     const cfg=await (await fetch('/config')).json();
     return cfg.dependencies.filter(d=>d.backend_fn===true)
        .map(d=>({id:d.id,targets:d.targets,api_name:d.api_name}));
   });
   assert.ok(deps.length>=3, 'Generate, Stop, Inspect server handlers registered');
   try{
     await page.waitForFunction(() => {
       const result=document.querySelector('#cd_execution_evidence textarea');
       return Boolean(result && result.value.includes('SMOKE_BACKEND_CALLED'));
     }, null, {timeout:20000});
   }catch(e){
     const state=await page.evaluate(()=>({
       response:document.querySelector('#cd_execution_evidence textarea')?.value||'',
       active:document.querySelector('#cd_submit')?.outerHTML.slice(0,300)
     }));
     throw new Error('Gradio editor-to-Python Generate failed: '+JSON.stringify({state,log:log.slice(-3000),deps,errors}));
   }
   const evidence=JSON.parse(await page.locator('#cd_execution_evidence textarea').inputValue());
   assert.equal(evidence.type,'SMOKE_BACKEND_CALLED');
   assert.equal(evidence.input_count,17,'all UI inputs transferred to backend');
   for(const [key,expected] of Object.entries({
     family:'anima',width:2048,height:1536,tile:512,overlap:192,
     halo:128,steps:9,cfg:1.25,seed:666,cpu_mib:2048,window_mib:256
   })){
     assert.equal(evidence[key],expected,'real Gradio backend received editor '+key);
   }
   // Direct Generate, independent of the iframe bridge, must hit exactly
   // the same Python callback and remain usable after a completed request.
   await page.locator('button#cd_submit, #cd_submit button').click();
   await page.waitForTimeout(1200);
   assert.equal(JSON.parse(await page.locator('#cd_execution_evidence textarea').inputValue()).type,
     'SMOKE_BACKEND_CALLED');
   await page.locator('button#cd_stop, #cd_stop button').click();
   await page.waitForFunction(()=>
     (document.querySelector('#cd_cancel_state textarea, #cd_cancel_state input')?.value||'').includes('Cancellation requested'),
     null,{timeout:10000});
   await page.getByText('txt2img',{exact:true}).first().click();
   // A legal Forge-style client-only listener must never be routed
   // to blocks.call_function (it has no Python fn).
   await page.locator('button#cd_host_js_only, #cd_host_js_only button').click();
   await page.waitForTimeout(400);
   await page.getByText('Coherent Diffusion — diagnostics',{exact:true}).first().click();
   await page.locator('button#cd_probe_button, #cd_probe_button button').click();
   await page.waitForFunction(()=>
     (document.querySelector('#cd_probe_evidence')?.innerText||'').includes('SMOKE_INSPECT_BACKEND_CALLED'),
     null,{timeout:10000});
   // Exercise the EXACT distributed JS observer and one-click export.
   await page.addScriptTag({path:'javascript/coherent_diffusion_diagnostics.js'});
   await page.waitForSelector('#cd-debug-toggle');
   await page.evaluate(()=>window.__cdDiagnosticsV1.record('browser_unit_check',{
     fn_index:12, route:'/queue/join', secret:'never-log',
     error:'dummy token with spaces'}));
   await page.locator('#cd-debug-toggle').click();
   await page.locator('#cd-debug-panel button').filter({hasText:'Download logs'}).waitFor();
   const downloadPromise=page.waitForEvent('download');
   await page.locator('#cd-debug-panel button').filter({hasText:'Download logs'}).click();
   const saved=await downloadPromise;
   const exported=JSON.parse(fs.readFileSync(await saved.path(),'utf8'));
   assert.equal(exported.format,'coherent-diffusion-diagnostics-v1');
   assert.ok(exported.browser_events.some(x=>x.kind==='browser_unit_check'&&x.fn_index===12));
   assert.ok(!JSON.stringify(exported).includes('never-log'),'request fields are allowlisted');
   assert.ok(exported.server_log_jsonl.includes('diagnostics_ready'),
      'live rotating Forge-side journal is visible to browser export');
   assert.ok(!log.includes('function has no backend method'),
      'missing backend function invoked by Gradio 4 queue: '+log.slice(-3000));
   assert.deepEqual(errors,[],'no parent page errors');
   console.log('P07 Gradio mount/editor transfer PASS (not model inference)');
  }finally{await browser.close()}
 }finally{server.kill('SIGTERM')}
})().catch(e=>{console.error(e);process.exit(1)});
