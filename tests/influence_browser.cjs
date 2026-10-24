// P02 numerical browser conformance: planned UI coefficients, never inference results.
// Run: node tests/influence_browser.cjs
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const KEY='coherent-diffusion-editor-p02-v1';
function region(id, overrides={}){
  const blend={weight:1,feather_px:0,z_index:1,...overrides.blend};
  return {
    id,name:id,geometry:{type:'bbox',bbox:[0,0,1,1]},prompt:'',
    negative_prompt:'',inherit_style:true,visible:true,enabled:true,
    locked:false,display_opacity:.27,blend,loras:[],color:'#4488ff',
    ...overrides,blend
  };
}
function scene(regions,policy='weighted',base=1,extra={}){
  return {
    canvas:{width:4096,height:3072,coordinate_system:'normalized_top_left',aspect_policy:'preserve_relative'},
    global:{scene_prompt:'Test scene',style_prompt:'Test style',negative_prompt:'',base_weight:base},
    blend:{policy},entities:[],regions,...extra
  };
}
async function mount(page,document){
  // pagehide autosave would overwrite a direct localStorage write on reload.
  // The single addInitScript below restores this fixture before the editor boot script.
  await page.evaluate(data=>{window.name='__P02_TEST_SCENE__'+JSON.stringify(data)},document);
  await page.reload();
  await page.locator('[data-view="influence"]').click();
}
async function readout(page){
  return page.locator('#influenceReadout').getAttribute('title');
}
async function expectShares(page,shares,absent=[]){
  const title=await readout(page);
  for(const [name,value] of Object.entries(shares)){
    assert.ok(title.includes(name+' '+value.toFixed(1)+'%'),name+' expected '+value.toFixed(1)+'%, saw: '+title);
  }
  for(const name of absent)assert.ok(!title.includes(name+' '),name+' must be absent: '+title);
}
async function hover(page,x,y){
  const surface=page.locator('#surface');
  const rect=await surface.boundingBox();
  await surface.hover({position:{x:rect.width*x,y:rect.height*y}});
}
function setLoRA(name,mode='approximate'){
  return [{id:'adapter-'+name,name,weight:.7,scope:'regional',requested_isolation:mode}];
}

(async()=>{
  const html=fs.readFileSync(path.resolve('coherent_diffusion/editor/index.html'));
  const server=http.createServer((_req,res)=>{
    res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});res.end(html);
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try{
    browser=await chromium.launch({headless:true});
    const page=await browser.newPage({viewport:{width:1500,height:1050}});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.addInitScript(key=>{
      const prefix='__P02_TEST_SCENE__';
      if(window.name.startsWith(prefix)){
        const document=JSON.parse(window.name.slice(prefix.length));
        localStorage.setItem(key,JSON.stringify({scene:document,settings:{model:'sdxl',strategy:'progressive',tile:'auto'}}));
      }
    },KEY);
    await page.goto('http://127.0.0.1:'+server.address().port+'/');

    const bottom=region('bottom',{blend:{weight:1,feather_px:0,z_index:1},color:'#0000ff'});
    const top=region('top',{blend:{weight:2,feather_px:0,z_index:2},color:'#ff0000'});
    await mount(page,scene([bottom,top]));
    await expectShares(page,{Base:25,top:50,bottom:25});
    let rgba=await page.evaluate(()=>[...document.querySelector('#weightMap').getContext('2d').getImageData(64,48,1,1).data]);
    assert.ok(rgba[3]>0&&rgba[0]>rgba[2],'weighted heatmap reflects dominant red top region');

    await mount(page,scene([bottom,top],'priority'));
    await expectShares(page,{Base:100/3,top:200/3},['bottom']);
    // Equal-z tie must favor the later CRF document region, not display visibility.
    await mount(page,scene([
      region('early',{blend:{weight:1,z_index:5,feather_px:0}}),
      region('later',{blend:{weight:1,z_index:5,feather_px:0}})
    ],'priority'));
    await expectShares(page,{Base:50,later:50},['early']);

    // 50% feathered top mask leaves 50% of the lower priority region unoccluded.
    const feathered=region('top',{
      geometry:{type:'bbox',bbox:[.375,0,.625,1]},
      blend:{weight:2,z_index:2,feather_px:1024},color:'#ff0000'
    });
    await mount(page,scene([bottom,feathered],'priority'));
    await expectShares(page,{Base:40,top:40,bottom:20});
    await hover(page,.1,.5);
    await expectShares(page,{Base:50,bottom:50},['top']);
    assert.match(await readout(page),/At 10\.[0-9]%/, 'footer uses pointer location');
    await hover(page,.5,.5);
    await expectShares(page,{Base:40,top:40,bottom:20});

    // Visibility is UI-only; enabled and zero weight independently exclude conditioning.
    await mount(page,scene([bottom,region('top',{...top,visible:false})]));
    await expectShares(page,{Base:25,top:50,bottom:25});
    assert.equal(await page.locator('#overlay [data-region="top"]').count(),0,'hidden region does not draw an overlay');
    await mount(page,scene([bottom,region('top',{...top,enabled:false})]));
    await expectShares(page,{Base:50,bottom:50},['top']);
    await mount(page,scene([bottom,region('top',{...top,blend:{weight:0,z_index:2,feather_px:0}})]));
    await expectShares(page,{Base:50,bottom:50},['top']);
    await mount(page,scene([region('silent',{enabled:false})]));
    await expectShares(page,{Base:100},['silent']);
    rgba=await page.evaluate(()=>[...document.querySelector('#weightMap').getContext('2d').getImageData(64,48,1,1).data]);
    assert.equal(rgba[3],0,'base-only heatmap has no regional overlay');

    // Authoring an incomplete zero-base scene must show true coverage loss.
    await mount(page,scene([], 'weighted',0));
    assert.match(await page.locator('#influenceWarnings').innerText(),/No conditioning coverage at 100\.0%/);
    assert.match(await readout(page),/NO COVERAGE/);
    await mount(page,scene([bottom],'weighted',.0001));
    assert.match(await page.locator('#influenceWarnings').innerText(),/Global base contribution below 5% at 100\.0%/);

    // Multiple distinct regional adapters overlap; prediction isolation is not active in P02.
    await mount(page,scene([
      region('a',{loras:setLoRA('style_a','prediction')}),
      region('b',{loras:setLoRA('style_b')})
    ]));
    let warnings=await page.locator('#influenceWarnings').innerText();
    assert.match(warnings,/Different LoRA intents overlap at 100\.0%/);
    assert.match(warnings,/1 prediction-isolation LoRA request/);
    assert.match(warnings,/not a measured model effect/);
    await mount(page,scene([
      region('a',{visible:false,loras:setLoRA('same_style')}),
      region('b',{loras:setLoRA('same_style')})
    ]));
    warnings=await page.locator('#influenceWarnings').innerText();
    assert.doesNotMatch(warnings,/Different LoRA intents overlap/,'same adapters do not warn merely due to visibility');
    await mount(page,scene([
      region('a',{enabled:false,loras:setLoRA('style_a','prediction')}),
      region('b',{loras:setLoRA('style_b')})
    ]));
    warnings=await page.locator('#influenceWarnings').innerText();
    assert.doesNotMatch(warnings,/prediction-isolation|Different LoRA intents/);

    // Raster-mask cache loads asynchronously. The warning must not count its pending pixels as uncovered.
    const maskData=await page.evaluate(()=>{
      const canvas=document.createElement('canvas');canvas.width=128;canvas.height=96;
      const ctx=canvas.getContext('2d');ctx.fillStyle='black';ctx.fillRect(0,0,128,96);
      ctx.fillStyle='white';ctx.fillRect(0,0,64,96);
      return canvas.toDataURL('image/png');
    });
    const masked=region('bitmap',{geometry:{type:'mask',asset_id:'mask-test'},color:'#ff0000'});
    await mount(page,scene([masked],'weighted',1,{maskData:{'mask-test':maskData}}));
    await page.waitForFunction(()=>!document.querySelector('#influenceWarnings').textContent.includes('still loading'));
    await hover(page,.25,.5);
    await expectShares(page,{Base:50,bitmap:50});
    await hover(page,.75,.5);
    await expectShares(page,{Base:100},['bitmap']);
    assert.deepEqual(errors,[],'no uncaught browser script errors');
    await page.close();
    console.log('P02 influence browser: PASS (numeric weighted/priority/feather/ties, pointer, masks, coverage, LoRA advisories)');
  }finally{
    if(browser)await browser.close();
    await new Promise(resolve=>server.close(resolve));
  }
})().catch(err=>{console.error(err);process.exitCode=1});
