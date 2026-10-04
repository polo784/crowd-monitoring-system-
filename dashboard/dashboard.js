/* Native-rate playback + bounded sampled inference; no frontend framework/CDN. */
'use strict';
const $ = id => document.getElementById(id);
const video = $('video'), heatmap = $('heatmap'), capture = document.createElement('canvas');
const ctx = heatmap.getContext('2d'), captureCtx = capture.getContext('2d');
let session = null, config = null, generation = 0, readyGeneration = -1;
let flight = null, selecting = false, snapshotWanted = false, nextSampleAt = 0;
let history = [], latest = null, fileVersion = 0, uploadController = null, scrubbing = false;
let graphStart = 0, sampleMediaTime = null;
let backendReady=false, healthFlight=false, retainedFile=null, recoveryNeeded=false;
let exportMetadata=null, observationsTotal=0, stateName='loading', completions=[];
let resetPending=false;
const buttons = ['play','pause','back','forward','restart','seek'];

function clock(t, decimal=false) {
  if (!Number.isFinite(t)) return '00:00';
  const m = Math.floor(Math.max(0,t)/60), s = Math.floor(Math.max(0,t)%60);
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}${decimal ? '.'+Math.floor((t%1)*10) : ''}`;
}
function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
function activity(message) { $('activity').textContent = message; }
function enableControls() {
  buttons.forEach(id=>$(id).disabled=!session||selecting||(id==='play'&&!backendReady));
  $('pause').disabled=!session;
  $('choose').disabled=selecting||!backendReady;
  $('export-csv').disabled=$('export-json').disabled=!history.length;
}
function setState(name,message) {
  stateName=name; $('play-state').textContent=name.toUpperCase();
  $('play-state').className=`status-pill ${['error','unavailable'].includes(name)?'error':name==='processing'?'playing':name==='ready'?'ready':''}`;
  if(message)activity(message);
}
function clearObservation(resetHistory=false,label='AWAITING DATA') {
  latest = null; ctx.clearRect(0,0,heatmap.width,heatmap.height);heatmap.style.visibility='hidden';
  if(resetHistory){history=[];observationsTotal=0;completions=[];}
  $('count').textContent = $('latency').textContent = $('sample-fps').textContent = $('model-fps').textContent = '—';
  $('service-time').textContent=$('browser-latency').textContent=$('observation-age').textContent='—';
  $('attention').textContent = label; $('attention').className = 'attention-badge';
  $('attention-reason').textContent='No current assessment; previous measurements are history only.';
  $('sample-time').textContent = 'No fresh model observation'; drawChart();enableControls();
}
function invalidatePending(){readyGeneration=-1;++generation;sampleMediaTime=null;snapshotWanted=false;if(flight)flight.controller.abort();}
function fail(message,unavailable=false){
  if(selecting){++fileVersion;selecting=false;uploadController?.abort();}
  backendReady=false;invalidatePending();clearObservation(false,unavailable?'UNAVAILABLE':'ERROR');
  video.pause();setState(unavailable?'unavailable':'error',message);notice(message);enableControls();
}
async function jsonRequest(url, options={}, timeoutSeconds=5) {
  const controller=new AbortController(),parent=options.signal,abort=()=>controller.abort();
  parent?.addEventListener('abort',abort,{once:true});if(parent?.aborted)controller.abort();
  let timedOut=false;
  const timer=setTimeout(()=>{timedOut=true;controller.abort();},timeoutSeconds*1000);
  try{
    const response = await fetch(url,{...options,signal:controller.signal});
    const data = await response.json();
    if (!response.ok) { const error = new Error(typeof data.detail==='string' ? data.detail : 'Invalid request'); error.status = response.status; throw error; }
    return data;
  }catch(error){
    if(timedOut){const timeout=new Error(`Request timed out after ${timeoutSeconds}s (${url}). Check the backend, then press Play after recovery.`);timeout.name='TimeoutError';throw timeout;}
    throw error;
  }finally{clearTimeout(timer);parent?.removeEventListener('abort',abort);}
}
async function syncControl(action, reset=false) {
  if (!session) return;
  const id = session.id, epoch = ++generation;
  readyGeneration = -1;
  if (flight) flight.controller.abort();
  sampleMediaTime = null;
  completions=[];clearObservation(reset);if(reset){graphStart=video.currentTime;resetPending=true;}
  if(!backendReady||recoveryNeeded)return;
  try {
    const response = await jsonRequest('/api/control', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({session_id:id,generation:epoch,action,timestamp:video.currentTime,playing:!video.paused&&!video.ended})},config.control_timeout_seconds);
    if (session?.id === id && epoch === generation) { readyGeneration = response.generation;session.segment_id=response.segment_id;nextSampleAt = 0;if(reset)resetPending=false; }
  } catch (error) {
    if (session?.id === id && epoch === generation)fail(error.message,true);
  }
}
async function selectFile(file) {
  if (!file) return;
  if (selecting) { notice('Please wait for the current video to finish opening.'); return; }
  if (file.size > (config?.max_video_bytes || 2147483648)) { notice('This video exceeds the configured 2 GiB upload limit.'); return; }
  const version = ++fileVersion;
  retainedFile=file;
  selecting = true; video.pause();
  if (flight) flight.controller.abort();
  if (uploadController) uploadController.abort();
  uploadController = new AbortController();
  session = null; generation = 0; readyGeneration = -1; snapshotWanted = false; graphStart = 0;
  clearObservation(true); enableControls(); notice('');setState('loading');
  video.removeAttribute('src'); video.load(); $('empty').hidden = false;
  $('empty').querySelector('h3').textContent = 'Preparing your video';
  $('empty').querySelector('p').textContent = 'Making a local playback copy. Your original file stays unchanged.';
  $('filename').textContent = file.name; activity('Opening video locally…');
  const body = new FormData(); body.append('file',file);
  try {
    const selected = await jsonRequest('/api/video', {method:'POST',body,signal:uploadController.signal},config.upload_timeout_seconds);
    if (version !== fileVersion) return;
    session = selected; generation = selected.generation; readyGeneration = generation;
    recoveryNeeded=false;resetPending=false;exportMetadata=selected.export_metadata;
    $('video-stage').style.aspectRatio = `${selected.video.width}/${selected.video.height}`;
    $('source-info').textContent = `RECORDED · ${selected.video.width} × ${selected.video.height} · ${selected.video.fps.toFixed(1)} FPS`;
    $('video-fps').textContent = (selected.video.original?.fps||selected.video.fps).toFixed(1);
    const [h,w] = selected.video.model_size_hw;
    $('resolution').textContent = `Input ${selected.video.width}×${selected.video.height} → model ${w}×${h}. Whole scene; no crop or count calibration.${selected.video.playback_conversion?' Browser copy converted to H.264.':''}`;
    video.src = selected.video_url; video.load(); $('empty').hidden = true;
    setState('ready','Ready — press Play');
  } catch (error) {
    if (version !== fileVersion || error.name === 'AbortError') return;
    fail(error.message,true); $('empty').querySelector('h3').textContent = 'Could not open this video';
    $('empty').querySelector('p').textContent = 'Try a supported MP4 file, or choose another video.'; activity('Video unavailable');
  } finally {
    if (version === fileVersion) { selecting = false; enableControls(); }
  }
}

function updatePlayback() {
  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  $('clock').textContent = `${clock(video.currentTime)} / ${clock(duration)}`;
  $('seek').max = duration || 1;
  if (!scrubbing) $('seek').value = video.currentTime;
  if (!latest) return;
  const lag = video.currentTime - latest.timestamp;
  const age=(performance.now()-latest.receivedAt)/1000+latest.browser_latency_ms/1000;
  if(age>config.observation_max_age_seconds||lag>config.overlay_max_age_seconds||lag<-.12){
    clearObservation(false,'STALE');setState('stale','Observation expired — retained graph is history only');return;
  }
  $('observation-age').textContent=`${age.toFixed(1)} s`;
  $('sample-time').textContent = `Sample ${clock(latest.timestamp,true)} · ${Math.max(0,lag).toFixed(1)}s behind · ${age.toFixed(1)}s old`;
  heatmap.style.visibility='visible';
}
function receiveResult(result, epoch,capturedAt=performance.now()) {
  if (!backendReady || !session || result.session_id !== session.id || epoch !== generation || result.generation !== generation) return;
  if (result.timestamp > video.currentTime + .12 || (history.length && result.timestamp <= history.at(-1).timestamp)) return;
  if(!Number.isFinite(result.count)){fail('Non-finite model count; no current assessment.');return;}
  const receivedAt=performance.now();result.browser_latency_ms=receivedAt-capturedAt;
  const {heatmap:png,...observation}=result;history.push(observation);observationsTotal++;
  if (history.length > config.history_limit) history.shift();
  completions.push(receivedAt);if(completions.length>20)completions.shift();
  observation.completed_samples_per_second=completions.length>1?(completions.length-1)*1000/(receivedAt-completions[0]):null;
  latest=result;latest.receivedAt=receivedAt;
  $('count').textContent = result.count.toFixed(1);
  $('latency').textContent = result.inference_ms.toFixed(0);
  $('sample-fps').textContent = observation.completed_samples_per_second == null ? '—' : observation.completed_samples_per_second.toFixed(1);
  $('model-fps').textContent = result.model_fps.toFixed(1);
  $('service-time').textContent=result.service_ms.toFixed(0)+' ms';$('browser-latency').textContent=result.browser_latency_ms.toFixed(0)+' ms';
  $('attention').textContent = result.attention;
  $('attention').className = `attention-badge attention-${result.attention.toLowerCase()}`;
  $('attention-reason').textContent=result.attention_reason;
  const bitmap = new Image(); bitmap.onload = () => {
    // PNG decoding is asynchronous too: another seek can happen before onload.
    if (!backendReady || epoch !== generation || latest !== result) return;
    heatmap.width = video.videoWidth; heatmap.height = video.videoHeight;
    ctx.clearRect(0,0,heatmap.width,heatmap.height); ctx.drawImage(bitmap,0,0,heatmap.width,heatmap.height);
    updatePlayback();
  }; bitmap.src = png;
  setState(video.paused?'paused':'processing',video.paused ? 'Paused — current-position sample' : 'Playing · sampled inference active');
  updatePlayback(); drawChart();
}
async function sampleFrame() {
  if (!backendReady || !session || selecting || flight || readyGeneration !== generation || video.seeking || video.readyState < 2) return;
  const snapshot = snapshotWanted;
  if ((video.paused || video.ended || document.hidden) && !snapshot) return;
  if (performance.now() < nextSampleAt && !snapshot) return;
  const epoch = generation, id = session.id;
  const timestamp = sampleMediaTime == null ? video.currentTime : sampleMediaTime;
  if (latest && timestamp <= latest.timestamp) { snapshotWanted = false; return; }
  const controller = new AbortController(), ticket = {controller}; flight = ticket; snapshotWanted = false;
  const capturedAt=performance.now(),captureTime=new Date().toISOString();
  nextSampleAt = performance.now() + 1000/config.max_sample_hz;
  try {
    capture.width = video.videoWidth; capture.height = video.videoHeight;
    captureCtx.drawImage(video,0,0,capture.width,capture.height);
    const blob = await new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Frame capture timed out')),5000);capture.toBlob(blob=>{clearTimeout(timer);resolve(blob);},'image/jpeg',.92);});
    if (epoch !== generation || controller.signal.aborted) return;
    if(!blob)throw new Error('Could not encode this video frame; choose a supported video.');
    const body = new FormData(); body.append('session_id',id); body.append('generation',String(epoch));
    body.append('timestamp',String(timestamp)); body.append('snapshot',String(snapshot)); body.append('frame',blob,'frame.jpg');
    body.append('capture_time',captureTime);
    const result = await jsonRequest('/api/infer',{method:'POST',body,signal:controller.signal},config.inference_timeout_seconds);
    receiveResult(result,epoch,capturedAt);
  } catch (error) {
    if (epoch !== generation || error.name === 'AbortError') return;
    if (error.status === 429) { if (snapshot) snapshotWanted = true; nextSampleAt = performance.now()+1000/config.max_sample_hz; }
    else if (error.status !== 409)fail(error.message,error.name==='TimeoutError'||!error.status);
  } finally { if (flight === ticket) flight = null; }
}

function drawChart() {
  $('history-info').textContent = `${history.length}/${config?.history_limit||1200} retained observations · HISTORY ONLY · ${Math.max(0,observationsTotal-history.length)} discarded`;
  $('export-csv').disabled=$('export-json').disabled=!history.length;
  $('chart-empty').hidden = history.length > 0;
  const c = $('chart'), box = c.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  c.width = Math.max(1,Math.round(box.width*ratio)); c.height = Math.max(1,Math.round(box.height*ratio));
  const g = c.getContext('2d'); g.scale(ratio,ratio);
  const W = box.width,H = box.height,left = 40,right = W-16,top = 10,bottom = H-28;
  const end = history.length ? history.at(-1).timestamp : graphStart;
  const start = Math.max(graphStart,end-60), xmax = Math.max(start+10,end);
  const points = history.filter(p => p.timestamp >= start);
  const minimum = Math.min(0,...points.map(p=>p.count));
  const maximum = Math.max(10,...points.map(p=>p.count))*1.15;
  const y = n => bottom-(n-minimum)/(maximum-minimum)*(bottom-top), x = t => left+(t-start)/(xmax-start)*(right-left);
  g.font = '10px Segoe UI'; g.fillStyle = '#8798ac'; g.strokeStyle = '#2b3543'; g.lineWidth = 1;
  for(let i=0;i<=4;i++){const n=minimum+(maximum-minimum)*i/4, yy=y(n);g.beginPath();g.moveTo(left,yy);g.lineTo(right,yy);g.stroke();g.textAlign='right';g.fillText(n.toFixed(0),left-8,yy+3);}
  for(let i=0;i<=4;i++){const t=start+(xmax-start)*i/4;g.textAlign=i===0?'left':i===4?'right':'center';g.fillText(clock(t),x(t),H-7);}
  if(!points.length)return;
  g.beginPath();points.forEach((p,i)=>i?g.lineTo(x(p.timestamp),y(p.count)):g.moveTo(x(p.timestamp),y(p.count)));
  g.strokeStyle='#a8c4ec';g.lineWidth=2;g.stroke();
  const last=points.at(-1);g.beginPath();g.arc(x(last.timestamp),y(last.count),3,0,2*Math.PI);g.fillStyle='#a8c4ec';g.fill();
}

async function health() {
  if(healthFlight)return;healthFlight=true;
  try {
    const state = await jsonRequest('/api/status',{},config?.health_timeout_seconds||5); config = state.config;
    const ready = state.model_state === 'ready'; $('model-status').dataset.ready = String(ready);
    $('model-status').className = `status-pill ${ready?'ready':state.model_state==='error'?'error':''}`;
    $('model-status').textContent = ready ? 'CSRNet loaded' : state.model_state==='error'?'Model unavailable':'Loading CSRNet…';
    if(state.model){$('device').textContent=state.model.device_name;$('checkpoint').textContent=`Mall · epoch ${state.model.checkpoint_epoch}`;}
    if(!ready){fail(state.error?`Model unavailable: ${state.error}`:'Model loading — playback paused',true);if(state.model_state==='loading')setState('loading');return;}
    const recovered=!backendReady;backendReady=true;
    if(session&&(!state.session||state.session.id!==session.id)&&!selecting){
      if(!recoveryNeeded){invalidatePending();clearObservation(false,'UNAVAILABLE');video.pause();}
      recoveryNeeded=true;setState('ready','Backend ready. Export prior history or press Play to reopen the video (new segment).');
    }else if(recovered){if(session)await syncControl(resetPending?'seek':'invalidate',resetPending);if(backendReady)setState('ready','Backend ready — press Play to resume');}
    const t=config.attention_thresholds;$('thresholds').textContent=`NORMAL < ${t.watch} · WATCH ≥ ${t.watch} · ELEVATED ≥ ${t.elevated} · REVIEW ≥ ${t.review}`;
    enableControls();
  } catch(error) {
    $('model-status').dataset.ready='false';$('model-status').textContent='Backend disconnected';
    if(backendReady||!['unavailable','error'].includes(stateName))fail(`Backend unavailable: ${error.message}. Playback paused; retained history can still be exported.`,true);
  }finally{healthFlight=false;enableControls();}
}
function exportData(format){
  if(!history.length)return;
  const observations=history.map(p=>({...p}));
  const metadata={...exportMetadata,segment_id:session?.segment_id,exported_at_utc:new Date().toISOString(),history_capacity:config.history_limit,
    observations_total:observationsTotal,discarded_observations:Math.max(0,observationsTotal-history.length),
    retained_range_seconds:[observations[0].timestamp,observations.at(-1).timestamp],
    history_status:'retained historical observations, not necessarily current',backend_connected:backendReady};
  let content,type;
  if(format==='json'){content=JSON.stringify({metadata,observations},null,2);type='application/json';}
  else{const keys=Object.keys(observations[0]);const cell=value=>{const text=value==null?'':typeof value==='object'?JSON.stringify(value):String(value);return '"'+text.replaceAll('"','""')+'"';};
    content=[keys.map(cell).join(','),...observations.map(p=>keys.map(k=>cell(p[k])).join(','))].join('\r\n');type='text/csv';}
  const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;
  a.download=`crowd-${session?.segment_id||'history'}.${format}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('export-csv').onclick=()=>exportData('csv');$('export-json').onclick=()=>exportData('json');
$('choose').onclick=()=>$('file').click();$('file').onchange=e=>{selectFile(e.target.files[0]);e.target.value='';};
const stage=$('video-stage');['dragenter','dragover'].forEach(name=>stage.addEventListener(name,e=>{e.preventDefault();stage.classList.add('dragging');}));
stage.addEventListener('dragleave',e=>{if(!stage.contains(e.relatedTarget))stage.classList.remove('dragging');});
stage.addEventListener('drop',e=>{e.preventDefault();stage.classList.remove('dragging');selectFile(e.dataTransfer.files[0]);});
$('play').onclick=async()=>{if(!backendReady)return;notice('');
  if(recoveryNeeded){await selectFile(retainedFile);notice('Video reopened after recovery; previous segment reset. Press Play.');return;}
  try{await video.play();}catch(error){fail('Browser could not play this video. Try an H.264 MP4.');}};
$('pause').onclick=()=>video.pause();$('back').onclick=()=>video.currentTime=Math.max(0,video.currentTime-5);
$('forward').onclick=()=>video.currentTime=Math.min(video.duration,video.currentTime+5);
$('restart').onclick=()=>{video.pause();video.currentTime=0;syncControl('restart',true);snapshotWanted=true;activity('Restarted — press Play');};
$('seek').oninput=()=>{scrubbing=true;};$('seek').onchange=e=>{scrubbing=false;video.currentTime=Number(e.target.value);};
video.addEventListener('play',()=>{if(!backendReady){video.pause();return;}syncControl('play');setState('processing','Playing · waiting for model sample');});
video.addEventListener('pause',()=>{if(!backendReady){invalidatePending();return;}syncControl('pause');setState('paused','Paused; retained graph is history');});
video.addEventListener('seeking',()=>{snapshotWanted=false;syncControl('seek',true);activity('Seeking — clearing previous observations');});
video.addEventListener('seeked',()=>{sampleMediaTime=null;snapshotWanted=true;updatePlayback();});
video.addEventListener('ended',()=>{syncControl('ended');$('play-state').textContent='Ended';activity('Video ended — restart or choose another source');});
video.addEventListener('loadedmetadata',()=>{updatePlayback();enableControls();});
video.addEventListener('error',()=>{if(session&&!selecting)fail('Browser playback failed. Try an H.264 MP4 video.');});
$('heatmap-toggle').onchange=e=>{heatmap.hidden=!e.target.checked;};$('opacity').oninput=e=>{heatmap.style.opacity=e.target.value;$('opacity-value').textContent=`${Math.round(e.target.value*100)}%`;};
if(video.requestVideoFrameCallback){const presented=(now,metadata)=>{sampleMediaTime=metadata.mediaTime;video.requestVideoFrameCallback(presented);};video.requestVideoFrameCallback(presented);}
new ResizeObserver(drawChart).observe($('chart').parentElement);
setInterval(()=>{updatePlayback();if(config)sampleFrame();},50);health();setInterval(health,2000);
