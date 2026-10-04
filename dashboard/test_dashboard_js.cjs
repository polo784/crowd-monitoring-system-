/* Node regression harness for our own frontend. Mock DOM/transport, NOT a browser test. */
'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const config=JSON.parse(fs.readFileSync('dashboard_config.json','utf8'));
const source=fs.readFileSync('dashboard/dashboard.js','utf8');
const tick=()=>new Promise(resolve=>setImmediate(resolve));

async function fixture(){
  let now=1000,download=null,fetchImpl=async()=>({ok:true,json:async()=>({model_state:'ready',config,session:null,model:{checkpoint_epoch:16,device_name:'test'}})});
  const elements=new Map();
  const draw=new Proxy({},{get:()=>()=>{}});
  function element(id){
    if(!elements.has(id))elements.set(id,{id,textContent:'',disabled:false,hidden:false,style:{},dataset:{},classList:{add(){},remove(){}},
      parentElement:{},listeners:{},addEventListener(name,fn){this.listeners[name]=fn;},
      getContext:()=>draw,getBoundingClientRect:()=>({width:800,height:200}),
      querySelector:()=>({textContent:''}),contains:()=>false,click(){},load(){},removeAttribute(){},
      width:960,height:540,videoWidth:960,videoHeight:540,currentTime:1,duration:20,readyState:2,
      seeking:false,paused:false,ended:false,pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();},
      toBlob:fn=>fn(new Blob(['jpeg']))});
    return elements.get(id);
  }
  const sandbox={console,performance:{now:()=>now},document:{getElementById:element,hidden:false,
    createElement:type=>type==='a'?{click(){}}:element('capture')},window:{devicePixelRatio:1},
    Image:class{set src(value){queueMicrotask(()=>this.onload?.());}},ResizeObserver:class{observe(){}},
    AbortController,FormData,Blob,Date,setTimeout,clearTimeout,setInterval:()=>0,queueMicrotask,
    URL:{createObjectURL:blob=>{download=blob;return 'blob:test';},revokeObjectURL(){}},fetch:(...args)=>fetchImpl(...args)};
  const scope=vm.createContext(sandbox);vm.runInContext(source,scope);await tick();
  vm.runInContext(`session={id:'s',segment_id:'seg'};generation=1;readyGeneration=1;backendReady=true;
    exportMetadata={model:{checkpoint_sha256:'hash',checkpoint_epoch:16},video:{original:{codec:'FMP4'},playback:{codec:'avc1'}}};`,scope);
  return {scope,element,run:code=>vm.runInContext(code,scope),setNow:value=>now=value,setFetch:fn=>fetchImpl=fn,getDownload:()=>download};
}
const response=stamp=>({session_id:'s',generation:1,segment_id:'seg',timestamp:stamp,count:42.123456789012345,
  attention:'WATCH',attention_reason:'Raw count ≥ WATCH threshold 40',inference_ms:170,model_fps:5.882,
  service_ms:190,input_shape:[1,3,544,960],output_shape:[1,1,68,120],heatmap:'data:image/png;base64,test',
  capture_time_utc:'2026-10-05T00:00:00.000Z',observation_time_utc:'2026-10-05T00:00:00.200Z'});

(async()=>{
  let passed=0;
  const test=async(name,work)=>{await work();console.log('PASS '+name);passed++;};
  await test('disconnect pauses locally, invalidates current values and keeps exportable history',async()=>{
    const f=await fixture();f.scope.result=response(1);f.run('receiveResult(result,1,800)');await tick();
    f.setFetch(async()=>{throw new Error('offline');});await f.run('health()');
    assert.equal(f.element('video').paused,true);assert.equal(f.element('count').textContent,'—');
    f.element('video').listeners.pause();assert.equal(f.element('attention').textContent,'UNAVAILABLE');
    assert.equal(f.element('pause').disabled,false);assert.equal(f.element('export-json').disabled,false);
    assert.equal(f.run('history.length'),1);assert.equal(f.element('heatmap').style.visibility,'hidden');
    f.scope.result=response(1.1);f.run('receiveResult(result,1)');assert.equal(f.run('history.length'),1);
  });
  await test('paused snapshot expires on wall-clock even when video time does not move',async()=>{
    const f=await fixture();f.element('video').paused=true;f.scope.result=response(1);f.run('receiveResult(result,1,800)');
    f.setNow(7000);f.run('updatePlayback()');assert.equal(f.run('stateName'),'stale');
    assert.equal(f.element('count').textContent,'—');assert.equal(f.run('history.length'),1);
  });
  await test('seek/restart generations discard late network and image responses',async()=>{
    for(const action of ['seek','restart']){
      const f=await fixture();f.scope.result=response(1);f.run('receiveResult(result,1)');
      f.setFetch(async()=>({ok:true,json:async()=>({generation:2,segment_id:'next'})}));
      await f.run(`syncControl('${action}',true)`);await tick();f.run('receiveResult(result,1)');
      assert.equal(f.run('history.length'),0);assert.equal(f.run('latest'),null);
      assert.equal(f.element('heatmap').style.visibility,'hidden');
    }
  });
  await test('video replacement cannot accept old-session responses',async()=>{
    const f=await fixture();f.run("session={id:'other',segment_id:'other'}");f.scope.result=response(1);
    f.run('receiveResult(result,1)');assert.equal(f.run('history.length'),0);
  });
  await test('bounded requests cancel on timeout and cancellation; no hung flight',async()=>{
    const f=await fixture();let aborted=false;
    f.setFetch((url,options)=>new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>{
      aborted=true;const error=new Error('aborted');error.name='AbortError';reject(error);
    })));
    await assert.rejects(f.run("jsonRequest('/api/status',{},.01)"),/timed out/);assert.ok(aborted);
    f.run('config={...config,inference_timeout_seconds:.01}; snapshotWanted=true');
    await f.run('sampleFrame()');assert.equal(f.run('flight'),null);assert.equal(f.element('video').paused,true);
  });
  await test('recovery requires readiness and never autoplays',async()=>{
    const f=await fixture();f.run("fail('offline',true)");
    f.setFetch(async()=>({ok:true,json:async()=>({model_state:'ready',config,session:null,model:{checkpoint_epoch:16,device_name:'test'}})}));
    await f.run('health()');assert.equal(f.run('backendReady'),true);assert.equal(f.run('recoveryNeeded'),true);
    assert.equal(f.element('video').paused,true);assert.equal(f.element('play').disabled,false);
  });
  await test('CSV/JSON preserve precision, provenance, bounded history and offline export',async()=>{
    const f=await fixture();f.run('config={...config,history_limit:2}');
    for(const stamp of [.1,.2,.3]){f.scope.result=response(stamp);f.run('receiveResult(result,1,800)');}
    f.run("fail('offline',true);exportData('json')");
    const exported=JSON.parse(await f.getDownload().text());assert.equal(exported.observations.length,2);
    assert.equal(exported.metadata.discarded_observations,1);assert.equal(exported.metadata.backend_connected,false);
    assert.equal(exported.observations[0].count,response(.2).count);assert.equal(exported.metadata.model.checkpoint_epoch,16);
    assert.equal(exported.metadata.video.original.codec,'FMP4');assert.equal(exported.metadata.video.playback.codec,'avc1');
    assert.equal(exported.observations[0].browser_latency_ms,200);
    f.run("exportData('csv')");const csv=await f.getDownload().text();assert.ok(csv.includes(String(response(.2).count)));
    assert.ok(csv.includes('input_shape'));assert.equal(csv.split('\r\n').length,3);
  });
  await test('history rejects repeated timestamps even after visible observation expires',async()=>{
    const f=await fixture();f.scope.result=response(1);f.run('receiveResult(result,1,800)');f.setNow(7000);f.run('updatePlayback()');
    f.run('receiveResult(result,1)');assert.equal(f.run('history.length'),1);
  });
  await test('frame capture failure releases outstanding ticket and invalidates current data',async()=>{
    const f=await fixture();f.element('capture').toBlob=fn=>fn(null);
    await f.run('sampleFrame()');assert.equal(f.run('flight'),null);assert.equal(f.run('backendReady'),false);
    assert.equal(f.element('video').paused,true);assert.equal(f.element('count').textContent,'—');
  });
  console.log(`${passed} frontend regression tests passed (mock DOM; no browser claim)`);
})().catch(error=>{console.error(error);process.exitCode=1;});
