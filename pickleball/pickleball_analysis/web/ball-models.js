import {ReconstructionViews,estimatedBall} from './comparison-3d.js';
const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), ctx = overlay.getContext('2d');
let manifest, reconstructionManifest, reconstructions, clip, predictions, loading = 0;
const enabled = new Set(['pickleball-512','v3-512','v3-960']);
const cards = new Map();
const sceneCards = new Map();
const scenes = new ReconstructionViews();
const clock = t => `${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(3).padStart(6,'0')}`;
async function json(path) {const r = await fetch(path); if(!r.ok) throw new Error(`${path}: HTTP ${r.status}`); return r.json();}

function makeCards() {
  for(const m of manifest.models){
    const card = document.createElement('article'); card.className = 'model'; card.style.setProperty('--model-color',m.color);
    card.innerHTML = `<div class="model-header"><h3>${m.name}</h3><label><input type="checkbox" aria-label="${m.name} overlay" ${enabled.has(m.id)?'checked':''}> Overlay</label></div><div class="model-input"><span>Input <b>${m.size[0]} × ${m.size[1]}</b></span><span>${m.training} weights</span></div><div class="prediction"><canvas width="192" height="108" aria-label="${m.name} candidate detail"></canvas><div class="stats"><strong>Loading…</strong><p class="peak"></p><p class="pixel"></p></div></div><div class="coverage"><div><span>Frames with candidates</span><b>—</b></div><canvas width="600" height="8" aria-label="Candidate coverage over the excerpt"></canvas><small>Coverage, not measured accuracy</small></div>`;
    card.querySelector('input').addEventListener('change', e => {if(e.target.checked) enabled.add(m.id); else enabled.delete(m.id); card.classList.toggle('selected',enabled.has(m.id)); draw();});
    card.classList.toggle('selected', enabled.has(m.id));
    if(reconstructionManifest.selected.includes(m.id)){
      const sceneCard=document.createElement('article');sceneCard.className='scene-card';sceneCard.style.setProperty('--model-color',m.color);
      sceneCard.innerHTML=`<div class="scene-card-head"><h3>${m.name}</h3><span>${m.size[0]} × ${m.size[1]}</span></div><div class="scene-card-body"><div class="reconstruction-stage"><span class="scene-state">Loading…</span></div><div class="flight-metrics"><div><span>Ball height</span><strong>—</strong></div><p class="flight-detail">Loading…</p><p class="flight-summary"></p></div></div>`;
      $('scene-models').append(sceneCard);sceneCards.set(m.id,sceneCard);
      $('models').append(card);scenes.add(m.id,sceneCard.querySelector('.reconstruction-stage'),m.color);
    }else $('other-models').append(card);
    cards.set(m.id,card);
  }
}

async function loadClip(index){
  const ticket = ++loading;
  video.pause(); predictions=null; reconstructions=null; scenes.reset();
  clip=manifest.clips[index];
  $('matched-clip').textContent=`Same excerpt as the original viewer · ${clock(clip.viewer_start).slice(0,-4)}–${clock(clip.viewer_end).slice(0,-2)} · ${(clip.viewer_end-clip.viewer_start).toFixed(1)} seconds · 1080p`;
  $('notice').textContent='Loading 1080p frames and model predictions…';
  $('timeline').max = Math.max(0,clip.duration-1/clip.fps); $('timeline').value = 0;
  $('download').href = `/media/${clip.predictions}`;
  for(const card of cards.values()) card.querySelector('.stats strong').textContent = 'Loading…';
  video.src = `/media/${clip.video}`; video.playbackRate=Number($('speed').value);
  const reconstructionFile=reconstructionManifest.clips.find(c=>c.id===clip.id)?.file;
  $('download-3d').href=`/media/${reconstructionFile}`;
  const [result,reconstructionResult] = await Promise.all([json(`/media/${clip.predictions}`),json(`/media/${reconstructionFile}`)]);
  if(ticket!==loading) return;
  predictions=result;reconstructions=reconstructionResult;
  const hasFlights=Object.values(reconstructions.models).some(m=>m.segments.length);
  $('jump-flight').disabled=!hasFlights;
  $('reconstruction-help').textContent=hasFlights?'Solid balls show supported flight fits at 00:56.523–00:59.720. Other detections appear as hollow ground-projection markers: height and true court position are unknown.':'No reviewed flight fits in this excerpt. Detected candidates appear as hollow ground-projection markers, with unknown height.';
  const complete = manifest.models.filter(m=>predictions.models[m.id]).length;
  $('notice').textContent=`${clip.frame_count} native-rate frames · ${complete}/6 configurations ready · Rings are raw candidates at threshold 0.5. Missing detections stay empty.`;
  for(const m of manifest.models){
    const card=cards.get(m.id), data=result.models[m.id], bar=card.querySelector('.coverage canvas'), bctx=bar.getContext('2d');
    bctx.clearRect(0,0,bar.width,bar.height);
    if(!data){card.querySelector('.coverage b').textContent='Pending'; continue;}
    card.querySelector('.coverage b').textContent=`${(100*data.frames_with_candidates/data.frame_count).toFixed(1)}% · ${data.frames_with_candidates}/${data.frame_count}`;
    bctx.fillStyle=m.color;
    data.frames.forEach((row,i)=>{if(row.candidates.length) bctx.fillRect(i/data.frame_count*bar.width,0,Math.max(1,bar.width/data.frame_count),bar.height);});
  }
  for(const id of reconstructionManifest.selected){
    const data=reconstructions.models[id],card=sceneCards.get(id);
    card.querySelector('.flight-summary').textContent=`${data.segments.length} fitted flights · ${data.reconstructed_frames}/${data.total_frames} frames with height`;
  }
  draw();
}

function rowAt(data,t){
  if(!data?.frames?.length) return null;
  // Nearest decoded frame timestamp, never interpolate missing predictions.
  let lo=0,hi=data.frames.length-1;
  while(lo<hi){const mid=Math.floor((lo+hi)/2); if(data.frames[mid].t<t) lo=mid+1; else hi=mid;}
  const a=data.frames[lo],b=data.frames[Math.max(0,lo-1)];
  return Math.abs(a.t-t)<Math.abs(b.t-t)?a:b;
}

function ring(context,x,y,r,color,line=2){context.strokeStyle=color;context.lineWidth=line;context.beginPath();context.arc(x,y,r,0,2*Math.PI);context.stroke();context.beginPath();context.moveTo(x-r-4,y);context.lineTo(x-r+1,y);context.moveTo(x+r-1,y);context.lineTo(x+r+4,y);context.moveTo(x,y-r-4);context.lineTo(x,y-r+1);context.moveTo(x,y+r-1);context.lineTo(x,y+r+4);context.stroke();}

function crop(context,center,w,h){
  const x=Math.max(0,Math.min(clip.width-w,center[0]-w/2)),y=Math.max(0,Math.min(clip.height-h,center[1]-h/2));
  context.imageSmoothingEnabled=false;
  context.clearRect(0,0,w,h);
  if(video.readyState>=2) context.drawImage(video,x,y,w,h,0,0,w,h);
  return [x,y];
}

function draw(){
  if(!clip) return;
  const t=video.currentTime;
  ctx.clearRect(0,0,overlay.width,overlay.height);
  $('source-time').textContent=clock(clip.viewer_start+t);
  $('clip-time').textContent=`${t.toFixed(2)} / ${clip.duration.toFixed(2)}s`;
  if(document.activeElement!==$('timeline')) $('timeline').value=t;
  $('play').textContent=video.paused?'▶ Play':'Ⅱ Pause';
  if(!predictions) return;
  drawReconstructions(t);
  for(const m of manifest.models){
    const card=cards.get(m.id),data=predictions.models[m.id],row=rowAt(data,t);
    const cropCanvas=card.querySelector('.prediction canvas'),cctx=cropCanvas.getContext('2d');
    if(!row){card.querySelector('.stats strong').textContent='Pending run'; cctx.clearRect(0,0,192,108); continue;}
    const candidate=row.candidates[0];
    card.querySelector('.stats strong').textContent=candidate?'Candidate':'No detection';
    card.querySelector('.peak').textContent=`Heatmap peak ${row.peak.toFixed(3)}${row.candidates.length>1?` · ${row.candidates.length} regions`:''}`;
    card.querySelector('.pixel').textContent=candidate?`x ${candidate.pixel[0].toFixed(1)} · y ${candidate.pixel[1].toFixed(1)}`:'No region above 0.5';
    const o=crop(cctx,candidate?candidate.pixel:[1100,300],192,108);
    if(candidate) ring(cctx,candidate.pixel[0]-o[0],candidate.pixel[1]-o[1],7,m.color,1.5);
    else{cctx.fillStyle='#101516b0';cctx.fillRect(0,0,192,108);cctx.fillStyle='#c9d5d0';cctx.font='11px sans-serif';cctx.textAlign='center';cctx.fillText('No candidate ≥ 0.5',96,58);}
    if(!enabled.has(m.id)) continue;
    const visible=$('all-candidates').checked?row.candidates:row.candidates.slice(0,1);
    visible.forEach((c,i)=>{
      const [x,y]=c.pixel; ctx.globalAlpha=i===0?1:.45; ring(ctx,x,y,13+manifest.models.indexOf(m)*2,m.color,3);
      ctx.globalAlpha=1;
    });
  }
}

function drawReconstructions(t){
  if(!reconstructions)return;
  const sourceTime=clip.source_start+t;
  const pose=rowAt({frames:reconstructions.poses},t);
  const supportedPose=pose&&Math.abs(pose.t-t)<.12?pose:null;
  const options={players:$('show-players').checked,trails:$('flight-trails').checked};
  const baseline=estimatedBall(reconstructions.models['pickleball-512'].segments,sourceTime);
  for(const id of reconstructionManifest.selected){
    const model=reconstructions.models[id],card=sceneCards.get(id);
    const projection=rowAt({frames:model.detection_display},t)?.projection||null;
    const ball=scenes.update(id,model.segments,sourceTime,supportedPose,{...options,projection});
    const raw=rowAt(predictions.models[id],t);
    card.querySelector('.flight-metrics strong').textContent=ball?`${ball.xyz[2].toFixed(2)} m`:'Unknown';
    card.querySelector('.scene-state').textContent=ball?(projection?.show_with_fit?'Flight + unmatched 2D candidate':'Fitted flight'):projection?'2D projection · height unknown':raw?.candidates.length?'2D candidate · projection unavailable':'No ball detected';
    const imageScale=clip.width/model.calibration_image_size[0];
    let detail=ball?`Fit RMS ${(ball.error*imageScale).toFixed(1)} px`:projection?'Hollow marker assumes ground level; actual height is unknown.':'No model detection or supported flight.';
    if(ball&&baseline&&id!=='pickleball-512')detail+=` · Δ ${Math.hypot(...ball.xyz.map((x,i)=>x-baseline.xyz[i])).toFixed(2)} m vs Pickleball`;
    card.querySelector('.flight-detail').textContent=detail;
  }
}

$('play').addEventListener('click',async()=>{try{if(video.paused)await video.play();else video.pause();}catch(e){$('notice').textContent=`Playback unavailable: ${e.message}`;} draw();});
$('back').addEventListener('click',()=>{video.pause();video.currentTime=Math.max(0,video.currentTime-1/clip.fps);});
$('next').addEventListener('click',()=>{video.pause();video.currentTime=Math.min(clip.duration-1/clip.fps,video.currentTime+1/clip.fps);});
$('timeline').addEventListener('input',e=>{video.pause();video.currentTime=Number(e.target.value);});
$('speed').addEventListener('change',e=>video.playbackRate=Number(e.target.value));
$('all-candidates').addEventListener('change',draw);
for(const id of ['show-players','flight-trails'])$(id).addEventListener('change',draw);
for(const name of ['perspective','top','side'])$('view-'+name).addEventListener('click',()=>{scenes.preset(name);for(const n of ['perspective','top','side'])$('view-'+n).classList.toggle('active',name===n);});
$('jump-flight').addEventListener('click',()=>{video.pause();const segments=reconstructions?.models['pickleball-512'].segments||[];if(segments.length)video.currentTime=(segments[0].start+segments[0].end)/2-clip.source_start;});
$('clips').addEventListener('change',e=>loadClip(Number(e.target.value)).catch(showError));
video.addEventListener('ended',()=>{if($('loop').checked){video.currentTime=0;video.play().catch(showError);}draw();});
video.addEventListener('error',()=>{$('notice').textContent='The 1080p preview could not load. Check the comparison clip file.';});
video.addEventListener('loadeddata',draw);video.addEventListener('seeked',draw);video.addEventListener('pause',draw);
function showError(e){$('notice').textContent=`Comparison unavailable: ${e.message}`;}
async function start(){[manifest,reconstructionManifest]=await Promise.all([json('/media/comparison/manifest.json'),json('/media/comparison/reconstruction-manifest.json')]);manifest.clips.forEach((c,i)=>$('clips').add(new Option(c.label,String(i))));makeCards();await loadClip(0);if(video.requestVideoFrameCallback){const update=()=>{draw();video.requestVideoFrameCallback(update);};video.requestVideoFrameCallback(update);}else{video.addEventListener('timeupdate',draw);}}
start().catch(showError);
