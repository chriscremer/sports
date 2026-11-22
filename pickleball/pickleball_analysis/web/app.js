import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {createMatchOverlay} from './match-overlay.mjs';
import {createMotionClock, sampleBridge, samplePlayers, sampleRallyBall, shortFlightBridges} from './reconstruction-motion.mjs';
import {attachMatchTimeline, matchPlayback} from './match-playback.mjs';
const $=id=>document.getElementById(id), video=$('video'),canvas=$('overlay'),ctx=canvas?.getContext('2d');
const matchUrl=document.body.dataset.matchUrl;
const scrubber=$('timeline');
let playback=video, updateOverview=null;
const mediaRoot=new URL(document.body.dataset.mediaBase||'/media/',document.baseURI);
const mediaURL=path=>new URL(path.replace(/^\/media\//,''),mediaRoot).href;
const clipsURL=document.body.dataset.clipsUrl||'/api/clips';
const ballDepthNote=$('ball-depth-note');
const rallyTimelines=new Map([...document.querySelectorAll('.rally-timeline[data-fps]')].map(lane=>[lane.dataset.rallySection,lane]));
let rallyTimeline=rallyTimelines.get(undefined), rallyMarkers=[];
let rallyPhase;
let rallyDisplayFrame=-1;
function selectRallyTimeline(next){
 if(rallyTimeline)rallyTimeline.hidden=true;
 rallyTimeline=next;
 if(!next)return;
 next.hidden=false;
 rallyMarkers=[...next.querySelectorAll('[data-rally-frame]')];
 rallyPhase=next.querySelector('[data-rally-phase], #rally-phase');
 rallyDisplayFrame=-1;
}
selectRallyTimeline(rallyTimeline);
const connections=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const colors={near:'#a0eacb',far:'#ffb27f'};
const matchOverlayCanvas=$('match-source-overlay'), matchOverlayToggle=$('show-match-overlay');
const drawMatchOverlay=matchOverlayCanvas&&createMatchOverlay(matchOverlayCanvas,connections,colors);
let data=null,clips=[],frame=null,frameIndex=0,lastDraw=-1,lastBall=-1,lastVideoTime=-1;
let ballSegments=[],ballBridges=[];
let lastPoseTime=-1, displayedPlayers=[];
const motionClock=createMotionClock();
const scene=new THREE.Scene();scene.background=new THREE.Color('#111b1e');scene.fog=new THREE.Fog('#111b1e',25,65);
const camera=new THREE.PerspectiveCamera(43,1,.1,100);camera.position.set(10,9,18);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));$('scene').appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(3.048,0,6.7);controls.enableDamping=true;controls.maxPolarAngle=Math.PI/2-.02;controls.minDistance=5;controls.maxDistance=40;controls.update();
scene.add(new THREE.HemisphereLight(0xffffff,0x253635,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(6,15,7);scene.add(light);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(70,70),new THREE.MeshStandardMaterial({color:'#142329',roughness:1}));ground.rotation.x=-Math.PI/2;ground.position.set(3,-.025,6.7);scene.add(ground);
function courtRect(x,z,w,h,color){const mesh=new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshStandardMaterial({color,roughness:1}));mesh.rotation.x=-Math.PI/2;mesh.position.set(x+w/2,.002,z+h/2);scene.add(mesh)}
courtRect(0,0,6.096,13.4112,'#264449');courtRect(0,4.572,6.096,4.2672,'#345b58');
function line(points,color='#a3c7c4',opacity=1){const obj=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p))),new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity}));return obj}
scene.add(line([[0,.025,0],[6.096,.025,0],[6.096,.025,13.4112],[0,.025,13.4112],[0,.025,0]]));
for(const z of [4.572,8.8392])scene.add(line([[0,.025,z],[6.096,.025,z]]));
for(const [a,b] of [[0,4.572],[8.8392,13.4112]])scene.add(line([[3.048,.025,a],[3.048,.025,b]]));
const net=new THREE.Mesh(new THREE.PlaneGeometry(6.4,.914),new THREE.MeshBasicMaterial({color:'#bac9c7',transparent:true,opacity:.12,side:THREE.DoubleSide}));net.position.set(3.048,.457,6.7056);scene.add(net);
scene.add(line([[-.15,.914,6.7056],[6.25,.914,6.7056]],'#d8e4df'));
for(let x=0;x<6.1;x+=.18)scene.add(line([[x,0,6.7056],[x,.91,6.7056]],'#738c88',.3));
for(let y=.1;y<.9;y+=.12)scene.add(line([[0,y,6.7056],[6.1,y,6.7056]],'#738c88',.3));
const playerTrails=new THREE.Group();scene.add(playerTrails);
const playerRigs=new Map();
const limbGeometry=new THREE.CylinderGeometry(.04,.04,1,6);
const headGeometry=new THREE.SphereGeometry(.085,12,10);
const footGeometry=new THREE.RingGeometry(.12,.18,24);
const up=new THREE.Vector3(0,1,0), direction=new THREE.Vector3();
const ballMesh=new THREE.Mesh(new THREE.SphereGeometry(.12,16,12),new THREE.MeshBasicMaterial({color:'#e3fa68',transparent:true}));scene.add(ballMesh);ballMesh.visible=false;
const groundBall=matchUrl?new THREE.Mesh(new THREE.RingGeometry(.082,.125,20),new THREE.MeshBasicMaterial({color:'#e3fa68',side:THREE.DoubleSide})):null;
if(groundBall){groundBall.rotation.x=-Math.PI/2;groundBall.visible=false;scene.add(groundBall)}
const rippleGeometry=new THREE.RingGeometry(.68,1,32);
let contactRipples=[];
let ballTrail=null,ballTrailKey=null;
const coord=p=>[p[0],p[2],p[1]];
function makePlayerRig(team){
 const group=new THREE.Group();
 const material=new THREE.MeshBasicMaterial({color:colors[team],side:THREE.DoubleSide,transparent:true});
 const limbs=connections.map(()=>new THREE.Mesh(limbGeometry,material));
 const neck=new THREE.Mesh(limbGeometry,material),head=new THREE.Mesh(headGeometry,material);
 const feet=new THREE.Mesh(footGeometry,material);feet.rotation.x=-Math.PI/2;
 group.add(...limbs,neck,head,feet);scene.add(group);
 return {group,limbs,neck,head,feet,material};
}
function setBone(mesh,a,b){
 mesh.visible=!!(a&&b);
 if(!mesh.visible)return;
 direction.set(b[0]-a[0],b[2]-a[2],b[1]-a[1]);
 const length=direction.length();
 if(length<1e-6){mesh.visible=false;return}
 mesh.position.set((a[0]+b[0])/2,(a[2]+b[2])/2,(a[1]+b[1])/2);
 mesh.quaternion.setFromUnitVectors(up,direction.multiplyScalar(1/length));
 mesh.scale.y=length;
}
function renderPlayers(players){
 for(const {group} of playerRigs.values())group.visible=false;
 for(const player of players){
  const key=`${player.team}-${player.id}`;
  if(!playerRigs.has(key))playerRigs.set(key,makePlayerRig(player.team));
  const rig=playerRigs.get(key), points=player.joints3d;
  rig.group.visible=true;
  rig.material.opacity=player.opacity??(player.tracking_status==='interpolated'?.55:1);
  rig.material.depthWrite=rig.material.opacity>.99;
  for(let i=0;i<connections.length;i++){
   const [a,b]=connections[i];setBone(rig.limbs[i],points[a],points[b]);
  }
  const shoulders=points[5]&&points[6]?points[5].map((value,i)=>(value+points[6][i])/2):null;
  setBone(rig.neck,points[0],shoulders);
  rig.head.visible=rig.neck.visible;
  if(rig.head.visible)rig.head.position.set(...coord(points[0]));
  rig.feet.position.set(player.xy[0],.04,player.xy[1]);
 }
}
function rebuildPlayerTrails(f){
 for(const old of [...playerTrails.children]){
  playerTrails.remove(old);old.geometry.dispose();old.material.dispose();
 }
 if(!f||!$('trails').checked)return;
 for(const player of f.players){
  const trail=[];
  for(let i=frameIndex;i>=Math.max(0,frameIndex-45);i--){
   const sample=data.frames[i].players.find(p=>p.id===player.id&&p.team===player.team);
   if(!sample)break;
   trail.push([sample.xy[0],.03,sample.xy[1]]);
  }
  if(trail.length>1)playerTrails.add(line(trail,colors[player.team],.35));
 }
}
function estimateAt(t){
 for(const s of ballSegments){if(t<s.start||t>s.end)continue;
  const before=s.support_times.filter(x=>x<=t).at(-1),after=s.support_times.find(x=>x>=t);
  const maxGap=s.max_gap_s??.35,maxDistance=s.max_support_distance_s??.12;
  if((before!==undefined&&after!==undefined&&after-before>maxGap)||Math.min(...s.support_times.map(x=>Math.abs(x-t)))>maxDistance)continue;
  const k=s.drag_per_second,dt=t-s.start,d=s.end-s.start,f=-Math.expm1(-k*dt)/k,end=-Math.expm1(-k*d)/k,g=[0,0,-9.81];
  const v=s.p0.map((p,j)=>(s.p1[j]-p-g[j]*(d-end)/k)/end);const xyz=s.p0.map((p,j)=>p+f*v[j]+g[j]*(dt-f)/k);
  if(!canvas)return {xyz,height:xyz[2],error:s.reprojection_rmse_px};
  const {K,R,C}=data.calibration;const cam=R.map(r=>r.reduce((sum,n,j)=>sum+n*(xyz[j]-C[j]),0));
  const pixel=[K[0][0]*cam[0]/cam[2]+K[0][2],K[1][1]*cam[1]/cam[2]+K[1][2]];
  return {xyz,pixel,height:xyz[2],error:s.reprojection_rmse_px};
 }
 return null;
}
function ballAt(t){
 const track=data?.ball_track||[];if(!track.length||t<data.clip.start||t>=data.clip.end)return {observation:null,estimate:null,index:-1};
 let lo=0,hi=track.length;while(lo<hi){const mid=(lo+hi)>>1;if(track[mid].t<t)lo=mid+1;else hi=mid}
 let i=Math.min(lo,track.length-1);if(i>0&&Math.abs(track[i-1].t-t)<Math.abs(track[i].t-t))i--;
 const row=Math.abs(track[i].t-t)<.04?track[i]:null;
 return {observation:row?.ball||null,estimate:estimateAt(t),index:row?i:-1};
}
const rippleStyles={
 bounce:{color:'#e3fa68',label:'Ground bounce'},
 volley:{color:'#ff88d0',label:'Volley · pink ripple'},
 serve:{color:'#91dbe9',label:'Serve'},
 hit:{color:'#91dbe9',label:'Paddle contact'},
 turn:{color:'#e0e7ef',label:'Estimated direction change'},
};
function setContactRipples(review){
 for(const {mesh} of contactRipples){scene.remove(mesh);mesh.material.dispose()}
 const events=review.ripples||[...(review.bounces||[]).map(({t,xy})=>({t,xyz:[...xy,.017],kind:'bounce'})),
  ...(review.volleys||[]).map(event=>({...event,kind:'volley'}))];
 contactRipples=[];
 for(const {t,xyz,kind,approximate} of events){
  const position=xyz||sampleRallyBall(data.ball_playback,t)?.xyz;
  if(!position)continue;
  const {color,label}=rippleStyles[kind];
  const material=new THREE.MeshBasicMaterial({color,transparent:true,depthWrite:false,side:THREE.DoubleSide});
  const mesh=new THREE.Mesh(rippleGeometry,material);
  mesh.rotation.x=-Math.PI/2;
  mesh.position.set(...coord(position));
  mesh.visible=false;
  scene.add(mesh);
  contactRipples.push({t,mesh,kind,label:approximate&&kind!=='volley'?`Approximate ${label.toLowerCase()}`:label});
 }
}
function renderContactRipples(time,enabled){
 let active=null;
 for(const event of contactRipples){
  const {t,mesh,kind}=event;
  const progress=(time-t)/.4;
  mesh.visible=enabled&&progress>=0&&progress<1;
  if(!mesh.visible)continue;
  if(!active||kind==='volley'||active.kind!=='volley'&&t>active.t)active=event;
  const fade=1-progress;
  mesh.scale.setScalar(.065+.285*(1-fade**2));
  mesh.material.opacity=.92*Math.sqrt(fade);
 }
 return active?.label;
}
function renderBall(ball,t,motionTime,sample){
 const enabled=$('show-ball').checked;
 const contact=renderContactRipples(motionTime,enabled);
 const continuous=Boolean(data.ball_playback);
 const fit=continuous?null:motionTime===t?ball.estimate:estimateAt(motionTime);
 const position=enabled?(continuous?sample?.xyz:fit?.xyz||sampleBridge(ballBridges,motionTime)):null;
 ballMesh.visible=!!position;
 if(position){
  ballMesh.position.set(...coord(position));
  ballMesh.material.opacity=sample?.predicted ? .65 : 1;
 }
 const projection=enabled&&!continuous&&!position?ball.observation?.ground_xy:null;
 if(groundBall){
  groundBall.visible=!!projection;
  if(projection)groundBall.position.set(projection[0],.035,projection[1]);
 }
 if(ballDepthNote){
  ballDepthNote.textContent=!enabled?'Ball display off':contact|| (sample?.predicted?'Predicted ball position':sample||fit?'Estimated ball flight':position?'Interpolated ball position':projection?'Ground projection · height unknown':continuous?'Between rallies':ball.observation?'Ball visible in video · height unknown':'No ball position at this time');
 }
 const show=!!position;
 const trailIndex=sample?`${sample.rally.start}:${sample.index}`:ball.index;
 const key=`${data.clip.start}-${trailIndex}-${show}-${$('trails').checked}`;
 if(key!==ballTrailKey){
  if(ballTrail){scene.remove(ballTrail);ballTrail.geometry.dispose();ballTrail.material.dispose();ballTrail=null}
  if(show&&$('trails').checked){
   const points=[coord(position)];
   if(sample){
    const rows=sample.rally.samples;
    for(let i=sample.index;i>=0&&rows[i][0]>=motionTime-.6;i--)points.push(coord(rows[i].slice(1,4)));
   }else{
    const recent=data.ball_track.filter(r=>r.t>=t-.6&&r.t<=t);
    for(const row of recent.reverse()){
     const position=row.reconstruction||row.review_reconstruction;
     if(!position)break;
     points.push(coord(position.xyz));
    }
   }
   if(points.length>1){ballTrail=line(points,'#e3fa68',.7);ballTrail.frustumCulled=false;scene.add(ballTrail)}
  }
  ballTrailKey=key;
 }
 if(show&&ballTrail){
  const vertices=ballTrail.geometry.attributes.position;
  vertices.setXYZ(0,...coord(position));vertices.needsUpdate=true;
 }
 $('ball-status').textContent=sample?`${sample.predicted?'≈ ':''}${sample.xyz[2].toFixed(2)} m`:ball.estimate?`${ball.estimate.height.toFixed(2)} m`:ball.observation?'2D only':'Unknown';
 const reviewed=ball.observation?.tracking_status==='reviewed';
 $('ball-detail').textContent=sample?.predicted?'Position predicted from nearby ball flights, detections and contacts.':reviewed?'Ball center reviewed in the source frame.':ball.estimate?`Estimated flight · ${ball.observation?'model detection':'gap filled by flight fit'} · ${ball.estimate.error.toFixed(1)} px fit RMS`:sample?'Position follows a supported flight.':ball.observation?'TrackNet detected the ball. No supported height estimate at this time.':'No supported ball track at this time.';
}
function currentTime(){return playback.currentTime+(data?.clip.start||0)}
function fmt(t,ms=false){return `${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}${ms?'.'+String(Math.floor(t%1*1000)).padStart(3,'0'):''}`}
function poseIndex(t){
 if(t<data.clip.start||t>=data.clip.end)return -1;
 if(matchUrl){
  const window=data.camera_windows.find(({start,end})=>start<=t&&t<end);
  if(!window)return -1;
  let lo=0,hi=data.frames.length;
  while(lo<hi){const mid=(lo+hi)>>1;if(data.frames[mid].t<t)lo=mid+1;else hi=mid}
  const valid=[lo-1,lo].filter(index=>{
   const frame=data.frames[index];
   return frame&&frame.t>=window.start-.000001&&frame.t<window.end&&Math.abs(frame.t-t)<=1.1/data.clip.hz;
  });
  return valid.sort((a,b)=>Math.abs(data.frames[a].t-t)-Math.abs(data.frames[b].t-t))[0]??-1;
 }
 const index=Math.min(data.frames.length-1,Math.max(0,Math.round((t-data.clip.start)*data.clip.hz)));
 return data.frames[index]&&Math.abs(data.frames[index].t-t)<.12?index:-1;
}
// Composite decoded footage and detections together: the embedded browser can
// lose its separate video presentation layer while continuing to decode frames.
function drawOverlay(f,ball){
 const width=data?.clip.width||1280,height=data?.clip.height||720;
 if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
 ctx.clearRect(0,0,width,height);ctx.shadowBlur=0;ctx.setLineDash([]);
 ctx.drawImage(video,0,0,width,height);
 if(!f)return;
 if($('show-pose').checked)for(const p of f.players){const k=p.keypoints;ctx.strokeStyle=colors[p.team];ctx.fillStyle=colors[p.team];ctx.lineWidth=3;ctx.shadowColor='#000';ctx.shadowBlur=2;ctx.setLineDash(p.tracking_status==='interpolated'?[6,4]:[]);
  for(const [a,b] of connections)if(k[a][2]>.2&&k[b][2]>.2){ctx.beginPath();ctx.moveTo(k[a][0],k[a][1]);ctx.lineTo(k[b][0],k[b][1]);ctx.stroke()}
  for(const [x,y,c] of k)if(c>.2){ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}
  if(k[0][2]>.2){ctx.font='bold 16px sans-serif';ctx.fillText(`${p.team==='near'?'N':'F'}${p.id}${p.tracking_status==='interpolated'?'*':''}`,k[0][0]+10,k[0][1]-12)}
 }
 ctx.setLineDash([]);
 const position=ball.observation?.pixel||ball.estimate?.pixel;
 if($('show-ball').checked&&position){
  const [x,y]=position, reviewed=ball.observation?.tracking_status==='reviewed';
  const color=reviewed?'#ff9fe2':'#e3fa68';
  ctx.strokeStyle=color;ctx.lineWidth=reviewed?3:2;ctx.setLineDash(ball.observation?[]:[4,4]);
  ctx.beginPath();ctx.arc(x,y,11,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle=color;ctx.font='12px sans-serif';ctx.fillText(reviewed?'Grid review':ball.observation?'TrackNet':'flight estimate',x+15,y);
 }
}
function metrics(f){$('count').innerHTML=`${f?.players.length??'—'} <small>/ 4</small>`;const ps=f?.players||[],observed=ps.filter(p=>p.tracking_status!=='interpolated'),filled=ps.length-observed.length;const avg=observed.length?observed.reduce((s,p)=>s+p.confidence,0)/observed.length:0;$('confidence').textContent=observed.length?`Detection confidence: ${Math.round(avg*100)}%${filled?` · ${filled} gap interpolated`:''}`:'Detection confidence: unavailable';
 const near=ps.filter(p=>p.team==='near'),far=ps.filter(p=>p.team==='far');const dist=(p,near)=>Math.abs(p.xy[1]-(near?8.8392:4.572));const mean=(arr,near)=>arr.reduce((s,p)=>s+dist(p,near),0)/arr.length;
 $('near-distance').textContent=near.length?`${mean(near,true).toFixed(2)} m`:'—';$('far-distance').textContent=far.length?`${mean(far,false).toFixed(2)} m`:'—';
 const index=near.length===2&&far.length===2&&!filled?Math.round(50+Math.max(-50,Math.min(50,(mean(far,false)-mean(near,true))*8))):null;$('index').textContent=index??'—';$('score-bar').style.width=(index??0)+'%';
 $('notice').textContent=!f?(matchUrl?'No player reconstruction at this time.':'No reconstruction at this time. Choose an analyzed excerpt to compare observed poses.'):`${fmt(f.t)} in original match · ${ps.length}/4 players tracked${filled?` · ${filled} short gap interpolated (dashed overlay)`:''} · Body depth approximated · ${data.ball_playback?'Ball height fitted or predicted during rallies':'Ball height estimated only on reviewed flights'}`;$('notice').classList.toggle('warning',!f||ps.length!==4||filled>0);
}
function updateRallyTimeline(clipTime){
 if(!rallyTimeline)return;
 const {fps,duration,offset,startFrame,endFrame}=rallyTimeline.dataset;
 const relative=clipTime-Number(offset||0);
 rallyTimeline.style.setProperty('--playhead',`${Math.max(0,Math.min(100,100*relative/Number(duration)))}%`);
 const frame=Math.floor(relative*Number(fps));
 if(frame===rallyDisplayFrame)return;
 rallyDisplayFrame=frame;
 const active=rallyMarkers.filter(marker=>Number(marker.dataset.rallyFrame)===frame);
 for(const marker of rallyMarkers){
  if(active.includes(marker))marker.setAttribute('aria-current','true');
  else marker.removeAttribute('aria-current');
 }
 rallyPhase.textContent=active.at(-1)?.dataset.rallyLabel||
  (frame<Number(startFrame)?'Before serve':frame>Number(endFrame)?'Rally over':'In play');
}
function update(force=false,now=performance.now()){
 if(!data)return;
 const t=currentTime(), clipTime=playback.currentTime;
 if(matchUrl&&clipTime>=playback.duration&&!playback.paused)playback.pause();
 const ballStart=data.ball_review?.source_start??data.clip.start,ballTime=ballStart+clipTime;
 const running=!force&&!playback.paused&&!playback.seeking&&playback.readyState>=3;
 const fps=data.ball_review?.fps||data.ball_summary?.source_hz||30;
 const smoothTime=Math.min(playback.duration||Infinity,motionClock(clipTime,now,playback.playbackRate,running,1/fps));
 const poseTime=data.clip.start+smoothTime;
 const ballMotionTime=ballStart+smoothTime;
 frameIndex=poseIndex(t);frame=data.frames[frameIndex]||null;
 const ball=ballAt(ballTime);
 const sample=data.ball_playback?sampleRallyBall(data.ball_playback,ballTime,ballMotionTime):null;
 if(force||frameIndex!==lastDraw){rebuildPlayerTrails(frame);metrics(frame)}
 if(force||poseTime!==lastPoseTime){
  displayedPlayers=samplePlayers(data.frames,poseIndex(poseTime),poseTime,data.clip.hz);
  renderPlayers(displayedPlayers);lastPoseTime=poseTime;
 }
 const frameReady=canvas&&video.readyState>=2&&!video.seeking;
 if(frameReady&&(force||frameIndex!==lastDraw||ball.index!==lastBall||clipTime!==lastVideoTime)){
  drawOverlay(frame,ball);lastBall=ball.index;lastVideoTime=clipTime;
 }
 lastDraw=frameIndex;renderBall(ball,ballTime,ballMotionTime,sample);
 if(drawMatchOverlay)drawMatchOverlay(data.source_projection,ballTime,displayedPlayers,$('show-ball').checked?sample:null,
  matchOverlayToggle.checked&&playback.hasFrame,force||poseTime!==t);
 $('source-time').textContent=fmt(ballTime,true);$('clip-time').textContent=`${fmt(clipTime)} / ${fmt(Number.isFinite(playback.duration)?playback.duration:0)}`;
 if(scrubber)scrubber.value=clipTime;
 $('play').textContent=playback.paused?'▶ Play':'Ⅱ Pause';
 if(updateOverview){
  const analyzed=updateOverview(clipTime);
  const next=analyzed==null?undefined:rallyTimelines.get(String(analyzed));
  if(next!==rallyTimeline)selectRallyTimeline(next);
 }
 updateRallyTimeline(clipTime);
 if(canvas)$('overlay-label').style.display=$('show-pose').checked&&frame?'':'none';
 $('scene-note').textContent=frame?'Drag to orbit · Scroll to zoom':matchUrl?'Player reconstruction unavailable at this time':'No analyzed pose at this time';
}
function animate(now){requestAnimationFrame(animate);controls.update();update(false,now);renderer.render(scene,camera)}
requestAnimationFrame(animate);
new ResizeObserver(()=>{const rect=$('scene').getBoundingClientRect();renderer.setSize(rect.width,rect.height);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix()}).observe($('scene'));
async function loadClip(url){
 video.pause();data=await fetch(url).then(r=>{if(!r.ok)throw Error('Tracking file unavailable');return r.json()});
 ballSegments=[...(data.ball_segments||[]),...(data.ball_review?.additional_segments||[])].sort((a,b)=>a.start-b.start);
 const contacts=(data.ball_events?.events||[]).map(event=>event.t);
 for(const event of data.ball_review?.rally_events?.events||[])contacts.push(event.source_seconds);
 ballBridges=shortFlightBridges(ballSegments,contacts);
 setContactRipples(data.ball_review||{});
 lastDraw=lastBall=-2;
 if(matchUrl){
  playback=matchPlayback(video,data.source,data.clip.end,()=>update(true));
  updateOverview=attachMatchTimeline(data,time=>{playback.currentTime=time;update(true)});
 }else{video.src=mediaURL(data.clip.video);video.load()}
 $('dataset-info').textContent=`Pose ${data.clip.hz} Hz · Ball ${data.ball_summary?.source_hz.toFixed(2)||'—'} Hz · ${data.ball_summary?.model||'TrackNet-Pickleball'}`;
 if($('coverage'))$('coverage').textContent=data.ball_summary?`${data.ball_summary.detected_frames}/${data.ball_summary.total_frames} frames have model detections. ${data.ball_summary.reconstructed_frames||0} frames have estimated height. Coverage is not an accuracy score.`:'Run the ball tracker for this excerpt.';
 update(true);
}
video.addEventListener('loadeddata',()=>update(true));
video.addEventListener('loadedmetadata',()=>{if(scrubber)scrubber.max=playback.duration;update(true)});video.addEventListener('seeked',()=>update(true));video.addEventListener('ended',()=>{if($('loop')?.checked){video.currentTime=0;video.play().catch(()=>{})}});video.addEventListener('error',()=>{$('notice').textContent='Video failed to load. Check that the local media server is running.'});
if($('mute'))$('mute').onclick=()=>{video.muted=!video.muted;$('mute').textContent=video.muted?'Unmute':'Mute';$('mute').setAttribute('aria-pressed',String(video.muted))};
if($('fullscreen'))$('fullscreen').onclick=()=>{const stage=video.closest('.video-stage');if(document.fullscreenElement)document.exitFullscreen();else stage.requestFullscreen().catch(showError)};
$('play').onclick=()=>playback.paused?playback.play().catch(e=>{$('notice').textContent=e.message}):playback.pause();
if(scrubber)scrubber.oninput=e=>{playback.currentTime=+e.target.value;update(true)};
$('speed').onchange=e=>playback.playbackRate=+e.target.value;
for(const [id,d] of [['back',-1],['next',1]])$(id)?.addEventListener('click',()=>{if(!data)return;playback.pause();const rate=data.ball_review?.fps||data.clip.hz;playback.currentTime=Math.max(0,Math.min(playback.duration,playback.currentTime+d/rate));update(true)});
for(const id of ['show-pose','show-ball','trails','show-match-overlay'])if($(id))$(id).onchange=()=>update(true);
for(const button of document.querySelectorAll('[data-review-seek]')){
 button.onclick=()=>{if(!data||(!matchUrl&&video.readyState<1))return;playback.pause();playback.currentTime=Number(button.dataset.reviewSeek);update(true);};
}
for(const [id,pos] of [['perspective',[10,9,18]],['top',[3.048,21,6.71]],['side',[21,5,6.7]]])$(id).onclick=()=>{camera.position.set(...pos);controls.target.set(3.048,0,6.7);controls.update();for(const n of ['perspective','top','side'])$(n).classList.toggle('active',id===n)};
function showError(e){$('notice').textContent=e.message;$('notice').classList.add('warning')}
try{
 const reviewURL=matchUrl||document.body.dataset.reviewUrl;
 if(reviewURL){await loadClip(reviewURL)}
 else{
  clips=await fetch(clipsURL).then(r=>{if(!r.ok)throw Error('Clip list unavailable');return r.json()});
  const clip=clips.find(c=>c.start===55);
  if(!clip)throw Error('The 10-second rally is unavailable.');
  await loadClip(mediaURL('analysis/'+clip.file));
 }
}catch(e){showError(e)}
