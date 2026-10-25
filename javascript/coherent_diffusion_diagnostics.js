/* P07 live diagnostics for Forge Neo. Installed via native javascript/*.js loader.
 * Only event identifiers, status codes and error classes are recorded.
 * Never store HTTP bodies, prompts, sessions, headers, paths or model weights.
 */
(() => {
  "use strict";
  if (window.__cdDiagnosticsV1) return;
  const STORE = "coherent-diffusion-diagnostic-events-v1";
  const KEEP = 350;
  const MAX_RENDER = 110;
  let events = [];
  let dependencyMap = new Map();
  let panel, status, output, toggle;
  try {
    const existing = JSON.parse(localStorage.getItem(STORE) || "[]");
    if (Array.isArray(existing)) events = existing.slice(-KEEP).filter(x => x && typeof x.kind === "string");
  } catch (_) {}
  function clean(v) {
    if (v === null || v === undefined || typeof v === "boolean" || Number.isInteger(v)) return v ?? null;
    if (typeof v === "string") return /^[\w:./-]{1,110}$/.test(v) ? v : "[redacted]";
    return null;
  }
  function put(kind, fields={}) {
    const row = {at:new Date().toISOString(), kind:String(kind).slice(0,70)};
    for (const key of ["fn_index","trigger_id","backend_fn","status","method","route","event","target","error","phase"]) {
      if (Object.prototype.hasOwnProperty.call(fields,key)) row[key] = clean(fields[key]);
    }
    events.push(row);
    if(events.length>KEEP) events.splice(0,events.length-KEEP);
    try{localStorage.setItem(STORE,JSON.stringify(events));}catch(_){}
    render();
  }
  function target(fn_index) {
    const dep = dependencyMap.get(Number(fn_index));
    return dep ? dep.target : null;
  }
  function recordRequestData(args) {
    try {
      const input = args[0];
      const init = args[1];
      const url = new URL(typeof input === "string" ? input : input?.url || "", location.href);
      if (url.origin !== location.origin) return null;
      const route = url.pathname;
      if(!/\/(queue\/join|queue\/data|run\/predict|api\/predict|config)$/.test(route)) return null;
      let body = init?.body;
      // Never clone or read a stream. Parse only a bounded in-memory JSON string,
      // extracting the event ID and discarding the contents immediately.
      let fn_index = null, trigger_id = null;
      if(typeof body === "string" && body.length < 2_000_000 && route.endsWith("/queue/join")){
        const obj=JSON.parse(body);
        fn_index = Number.isInteger(obj.fn_index) ? obj.fn_index : null;
        trigger_id = Number.isInteger(obj.trigger_id) ? obj.trigger_id : null;
      }
      const method = String(init?.method || input?.method || "GET").toUpperCase();
      return {route, method, fn_index, trigger_id};
    }catch(_){return null;}
  }
  const nativeFetch = window.fetch;
  if(typeof nativeFetch === "function"){
    window.fetch = function(...args){
      const info = recordRequestData(args);
      if(info && info.route.endsWith("/queue/join")){
        put("gradio_submit",{...info, target:target(info.fn_index)});
      }
      let p;
      try{p=nativeFetch.apply(this,args);}catch(e){
        if(info) put("fetch_throw",{...info,error:e?.name||"Error"});
        throw e;
      }
      return p.then(response=>{
        if(info && info.route.endsWith("/config")){
          try{
            response.clone().json().then(cfg=>{
              const cm=new Map((cfg.components||[]).map(c=>[c.id,c.props?.elem_id||c.props?.label||String(c.id)]));
              dependencyMap=new Map((cfg.dependencies||[]).map(d=>[Number(d.id),{
                backend_fn:d.backend_fn===true,
                target:(d.targets||[]).map(t=>(cm.get(t[0])||t[0])+":"+t[1]).join(",").slice(0,110)
              }]));
              put("gradio_config",{status:200});
            }).catch(()=>{});
          }catch(_){}
        }
        if(info && info.route.endsWith("/queue/join")){
          put("gradio_queue_response",{...info,status:response.status,
            backend_fn:dependencyMap.get(info.fn_index)?.backend_fn,
            target:target(info.fn_index)});
        } else if(info && !response.ok && !info.route.endsWith("/config")){
          put("http_error",{...info,status:response.status});
        }
        return response;
      },error=>{
        if(info) put("fetch_failure",{...info,error:error?.name||"Error"});
        throw error;
      });
    };
  }
  window.addEventListener("error", ev => put("browser_error",{error:ev.error?.name||"Error"}));
  window.addEventListener("unhandledrejection", ev => put("promise_rejection",{error:ev.reason?.name||"Error"}));
  function backendText() {
    const e = document.querySelector("#cd_debug_backend textarea, #cd_debug_backend input");
    return e && typeof e.value === "string" ? e.value.slice(-90000) : "";
  }
  function snapshot() {
    return {format:"coherent-diffusion-diagnostics-v1",generated_at:new Date().toISOString(),
      privacy:"No request bodies, prompts, secrets, image data, session hashes or model paths captured",
      browser_events:events.slice(),server_log_jsonl:backendText(),
      note:"Forge Python persistent log: outputs/coherent-diffusion/diagnostics/events.jsonl"};
  }
  function download() {
    const data=JSON.stringify(snapshot(),null,2);
    const blob=new Blob([data],{type:"application/json"});
    const url=URL.createObjectURL(blob);
    const a=document.createElement("a");a.href=url;
    a.download="coherent-diffusion-diagnostics-"+new Date().toISOString().replace(/[:.]/g,"-")+".json";
    document.body.append(a);a.click();a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1200);
    put("log_exported");
  }
  function render(){
    if(!output) return;
    const log=events.slice(-MAX_RENDER).map(e=>e.at.slice(11,19)+" "+e.kind+" "
      +Object.entries(e).filter(([k])=>k!=="at"&&k!=="kind").map(([k,v])=>k+"="+v).join(" ")).join("\n");
    const server=backendText().split("\n").slice(-25).join("\n");
    output.textContent="[Browser / Gradio event IDs]\n"+log+
      "\n\n[Forge Python events]\n"+(server||"Backend log loads when the Coherent Diffusion tab is open");
    if(status) status.textContent="Live · "+events.length+" events";
  }
  function mount(){
    if(document.getElementById("cd-debug-toggle")) return;
    if(!document.body) return;
    const root=document.createElement("aside");root.id="cd-debug-root";
    root.style.cssText="position:fixed;bottom:14px;right:14px;z-index:2147483000;font:12px system-ui;color:#eee";
    toggle=document.createElement("button");toggle.id="cd-debug-toggle";toggle.type="button";
    toggle.textContent="CD live debug";
    toggle.style.cssText="background:#273748;color:#fff;border:1px solid #728698;padding:8px 12px;border-radius:7px;cursor:pointer";
    panel=document.createElement("section");panel.id="cd-debug-panel";
    panel.style.cssText="display:none;position:absolute;bottom:43px;right:0;width:min(600px,calc(100vw - 25px));height:min(470px,60vh);padding:10px;background:#171c26;color:#e4e9f1;border:1px solid #64758a;border-radius:8px;box-shadow:0 6px 25px #0007;box-sizing:border-box";
    const heading=document.createElement("div");heading.textContent="Coherent Diffusion · Live diagnostics";
    heading.style.cssText="font-weight:650;margin-bottom:6px";
    status=document.createElement("span");status.style.cssText="display:block;font-size:11px;opacity:.85";
    const actions=document.createElement("div");actions.style.cssText="display:flex;gap:8px;margin:6px 0";
    const dl=document.createElement("button");dl.textContent="Download logs";dl.type="button";
    const close=document.createElement("button");close.textContent="Close";close.type="button";
    for(const b of [dl,close])b.style.cssText="background:#32435b;color:white;border:1px solid #66758b;border-radius:5px;padding:6px;cursor:pointer";
    dl.addEventListener("click",download);
    close.addEventListener("click",()=>panel.style.display="none");
    actions.append(dl,close);
    output=document.createElement("pre");output.id="cd-debug-logs";output.style.cssText="overflow:auto;height:calc(100% - 94px);white-space:pre-wrap;word-break:break-word;margin:0;background:#0c1118;padding:8px;color:#dce6f4";
    panel.append(heading,status,actions,output);
    toggle.addEventListener("click",()=>{panel.style.display=panel.style.display==="none"?"block":"none";render();});
    root.append(panel,toggle);document.body.append(root);render();
  }
  window.__cdDiagnosticsV1={record:put,download,snapshot,open:()=>{mount();panel.style.display="block";render();}};
  if(document.readyState==="loading") document.addEventListener("DOMContentLoaded",mount,{once:true});
  else mount();
  setInterval(()=>{mount();if(panel && panel.style.display!=="none")render();},1500);
  put("browser_diagnostics_ready");
})();