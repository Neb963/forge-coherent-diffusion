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
   await page.waitForFunction(() => {
     const input=document.querySelector('#cd_scene_json textarea');
     return Boolean(input && input.value.includes('"coherent-regions"'));
   }, null, {timeout:20000});
   assert.equal(await page.locator('#cd_submit button').count(),1,
     'one native Gradio Generate control is registered');
   assert.deepEqual(errors,[],'no parent page errors');
   console.log('P07 Gradio mount/editor transfer PASS (not model inference)');
  }finally{await browser.close()}
 }finally{server.kill('SIGTERM')}
})().catch(e=>{console.error(e);process.exit(1)});
