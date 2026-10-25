// Browser verification of real Gradio 4 HTML rendering (not a Forge model test).
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const assert=require('node:assert/strict');

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
   const page=await browser.newPage({viewport:{width:1536,height:1100}});
   const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(url,{waitUntil:'domcontentloaded'});
   await page.getByText('Coherent Diffusion', {exact:true}).first().click();
   const frame=page.frameLocator('iframe[title="Coherent Diffusion region editor"]');
   await frame.locator('#layerList .layer').first().waitFor({timeout:20000});
   assert.equal(await frame.locator('#layerList .layer').count(),5,'P02 editor mounted under Gradio');
   await frame.locator('#addRegionBtn').click();
   assert.equal(await frame.locator('#layerList .layer').count(),6,'Gradio-mounted iframe accepts interactions');
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
   await page.locator('button#cd_probe_button, #cd_probe_button button').click();
   await page.waitForFunction(()=>
     (document.querySelector('#cd_probe_evidence')?.innerText||'').includes('SMOKE_INSPECT_BACKEND_CALLED'),
     null,{timeout:10000});
   assert.ok(!log.includes('function has no backend method'),
      'missing backend function invoked by Gradio 4 queue: '+log.slice(-3000));
   assert.deepEqual(errors,[],'no parent page errors');
   console.log('P07 Gradio mount/editor transfer PASS (not model inference)');
  }finally{await browser.close()}
 }finally{server.kill('SIGTERM')}
})().catch(e=>{console.error(e);process.exit(1)});
