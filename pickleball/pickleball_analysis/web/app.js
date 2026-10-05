import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
const $=id=>document.getElementById(id), video=$('video'),canvas=$('overlay'),ctx=canvas.getContext('2d');
const mediaRoot=new URL(document.body.dataset.mediaBase||'/media/',document.baseURI);
const mediaURL=path=>new URL(path.replace(/^\/media\//,''),mediaRoot).href;
const clipsURL=document.body.dataset.clipsUrl||'/api/clips';
const connections=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const colors={near:'#a0eacb',far:'#ffb27f'};
let data=null,clips=[],frame=null,frameIndex=0,lastDraw=-1,lastBall=-1,lastVideoTime=-1;
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
const dynamic=new THREE.Group();scene.add(dynamic);
const ballMesh=new THREE.Mesh(new THREE.SphereGeometry(.12,16,12),new THREE.MeshBasicMaterial({color:'#e3fa68'}));scene.add(ballMesh);ballMesh.visible=false;
const ballShadow=new THREE.Mesh(new THREE.RingGeometry(.10,.14,24),new THREE.MeshBasicMaterial({color:'#e3fa68',transparent:true,opacity:.5,side:THREE.DoubleSide}));ballShadow.rotation.x=-Math.PI/2;scene.add(ballShadow);ballShadow.visible=false;
let ballTrail=null,ballTrailKey=null;
const coord=p=>[p[0],p[2],p[1]];
function disposeDynamic(){while(dynamic.children.length){const o=dynamic.children[0];dynamic.remove(o);o.traverse(n=>{n.geometry?.dispose();if(n.material)n.material.dispose()})}}
function segment(a,b,color,r=.04){const va=new THREE.Vector3(...a),vb=new THREE.Vector3(...b),dir=vb.clone().sub(va);const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,dir.length(),6),new THREE.MeshBasicMaterial({color}));m.position.copy(va.add(vb).multiplyScalar(.5));m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize());dynamic.add(m)}
function rebuild(f){disposeDynamic();if(!f)return;
 for(const player of f.players){const color=colors[player.team],points=player.joints3d,start=dynamic.children.length;
  for(const [a,b] of connections)if(points[a]&&points[b])segment(coord(points[a]),coord(points[b]),color);
  if(points[0]&&points[5]&&points[6]){const shoulders=points[5].map((n,i)=>(n+points[6][i])/2);segment(coord(points[0]),coord(shoulders),color);const head=new THREE.Mesh(new THREE.SphereGeometry(.085,12,10),new THREE.MeshBasicMaterial({color}));head.position.set(...coord(points[0]));dynamic.add(head)}
  const marker=new THREE.Mesh(new THREE.RingGeometry(.12,.18,24),new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide}));marker.rotation.x=-Math.PI/2;marker.position.set(player.xy[0],.04,player.xy[1]);dynamic.add(marker);
  if($('trails').checked){const trail=data.frames.slice(Math.max(0,frameIndex-45),frameIndex+1).map(f=>f.players.find(p=>p.id===player.id)).filter(Boolean).map(p=>[p.xy[0],.03,p.xy[1]]);if(trail.length>1)dynamic.add(line(trail,color,.35))}
  if(player.tracking_status==='interpolated')for(const object of dynamic.children.slice(start)){object.material.transparent=true;object.material.opacity=.55}
 }
}
function ballAt(t){
 const track=data?.ball_track||[];if(!track.length||t<data.clip.start||t>=data.clip.end)return {observation:null,estimate:null,index:-1};
 let lo=0,hi=track.length;while(lo<hi){const mid=(lo+hi)>>1;if(track[mid].t<t)lo=mid+1;else hi=mid}
 let i=Math.min(lo,track.length-1);if(i>0&&Math.abs(track[i-1].t-t)<Math.abs(track[i].t-t))i--;
 const row=Math.abs(track[i].t-t)<.04?track[i]:null;let estimate=null;
 for(const s of data.ball_segments||[]){if(t<s.start||t>s.end)continue;
  const before=s.support_times.filter(x=>x<=t).at(-1),after=s.support_times.find(x=>x>=t);
  if((before!==undefined&&after!==undefined&&after-before>.35)||Math.min(...s.support_times.map(x=>Math.abs(x-t)))>.12)continue;
  const k=s.drag_per_second,dt=t-s.start,d=s.end-s.start,f=-Math.expm1(-k*dt)/k,end=-Math.expm1(-k*d)/k,g=[0,0,-9.81];
  const v=s.p0.map((p,j)=>(s.p1[j]-p-g[j]*(d-end)/k)/end);const xyz=s.p0.map((p,j)=>p+f*v[j]+g[j]*(dt-f)/k);
  const {K,R,C}=data.calibration;const cam=R.map(r=>r.reduce((sum,n,j)=>sum+n*(xyz[j]-C[j]),0));
  const pixel=[K[0][0]*cam[0]/cam[2]+K[0][2],K[1][1]*cam[1]/cam[2]+K[1][2]];
  estimate={xyz,pixel,height:xyz[2],error:s.reprojection_rmse_px};break;
 }
 return {observation:row?.ball||null,estimate,index:row?i:-1};
}
function renderBall(ball,t){
 const show=$('show-ball').checked&&!!ball.estimate;ballMesh.visible=ballShadow.visible=show;
 if(show){const p=ball.estimate.xyz;ballMesh.position.set(...coord(p));ballShadow.position.set(p[0],.035,p[1]);}
 const key=`${data.clip.start}-${ball.index}-${show}-${$('trails').checked}`;
 if(key!==ballTrailKey){
  if(ballTrail){scene.remove(ballTrail);ballTrail.geometry.dispose();ballTrail.material.dispose();ballTrail=null}
  if(show&&$('trails').checked){const points=data.ball_track.filter(r=>r.t>=t-.6&&r.t<=t&&r.reconstruction).map(r=>coord(r.reconstruction.xyz));points.push(coord(ball.estimate.xyz));if(points.length>1){ballTrail=line(points,'#e3fa68',.7);scene.add(ballTrail)}}
  ballTrailKey=key;
 }
 $('ball-status').textContent=ball.estimate?`${ball.estimate.height.toFixed(2)} m`:ball.observation?'2D only':'Unknown';
 $('ball-detail').textContent=ball.estimate?`Estimated flight · ${ball.observation?'model detection':'gap filled by flight fit'} · ${ball.estimate.error.toFixed(1)} px fit RMS`:ball.observation?'TrackNet detected the ball. No supported height estimate at this time.':'No supported ball track at this time.';
}
function currentTime(){return video.currentTime+(data?.clip.start||0)}
function fmt(t,ms=false){return `${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}${ms?'.'+String(Math.floor(t%1*1000)).padStart(3,'0'):''}`}
function findFrame(t){if(!data||t<data.clip.start||t>=data.clip.end)return null;frameIndex=Math.min(data.frames.length-1,Math.max(0,Math.round((t-data.clip.start)*data.clip.hz)));const f=data.frames[frameIndex];return Math.abs(f.t-t)<.12?f:null}
// Composite decoded footage and detections together: the embedded browser can
// lose its separate video presentation layer while continuing to decode frames.
function drawOverlay(f,ball){
 const width=data?.clip.width||1280,height=data?.clip.height||720;
 if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
 ctx.clearRect(0,0,width,height);ctx.shadowBlur=0;ctx.setLineDash([]);
 if(video.readyState>=2)ctx.drawImage(video,0,0,width,height);
 if(!f)return;
 if($('show-pose').checked)for(const p of f.players){const k=p.keypoints;ctx.strokeStyle=colors[p.team];ctx.fillStyle=colors[p.team];ctx.lineWidth=3;ctx.shadowColor='#000';ctx.shadowBlur=2;ctx.setLineDash(p.tracking_status==='interpolated'?[6,4]:[]);
  for(const [a,b] of connections)if(k[a][2]>.2&&k[b][2]>.2){ctx.beginPath();ctx.moveTo(k[a][0],k[a][1]);ctx.lineTo(k[b][0],k[b][1]);ctx.stroke()}
  for(const [x,y,c] of k)if(c>.2){ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}
  if(k[0][2]>.2){ctx.font='bold 16px sans-serif';ctx.fillText(`${p.team==='near'?'N':'F'}${p.id}${p.tracking_status==='interpolated'?'*':''}`,k[0][0]+10,k[0][1]-12)}
 }
 ctx.setLineDash([]);
 const position=ball.observation?.pixel||ball.estimate?.pixel;
 if($('show-ball').checked&&position){const [x,y]=position;ctx.strokeStyle='#e3fa68';ctx.lineWidth=2;ctx.setLineDash(ball.observation?[]:[4,4]);ctx.beginPath();ctx.arc(x,y,11,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#e3fa68';ctx.font='12px sans-serif';ctx.fillText(ball.observation?'TrackNet':'flight estimate',x+15,y)}
}
function metrics(f){$('count').innerHTML=`${f?.players.length??'—'} <small>/ 4</small>`;const ps=f?.players||[],observed=ps.filter(p=>p.tracking_status!=='interpolated'),filled=ps.length-observed.length;const avg=observed.length?observed.reduce((s,p)=>s+p.confidence,0)/observed.length:0;$('confidence').textContent=observed.length?`Detection confidence: ${Math.round(avg*100)}%${filled?` · ${filled} gap interpolated`:''}`:'Detection confidence: unavailable';
 const near=ps.filter(p=>p.team==='near'),far=ps.filter(p=>p.team==='far');const dist=(p,near)=>Math.abs(p.xy[1]-(near?8.8392:4.572));const mean=(arr,near)=>arr.reduce((s,p)=>s+dist(p,near),0)/arr.length;
 $('near-distance').textContent=near.length?`${mean(near,true).toFixed(2)} m`:'—';$('far-distance').textContent=far.length?`${mean(far,false).toFixed(2)} m`:'—';
 const index=near.length===2&&far.length===2&&!filled?Math.round(50+Math.max(-50,Math.min(50,(mean(far,false)-mean(near,true))*8))):null;$('index').textContent=index??'—';$('score-bar').style.width=(index??0)+'%';
 $('notice').textContent=!f?'No reconstruction at this time. Choose an analyzed excerpt to compare observed poses.':`${fmt(f.t)} in original match · ${ps.length}/4 players tracked${filled?` · ${filled} short gap interpolated (dashed overlay)`:''} · Body depth approximated · Ball height estimated only on reviewed flights`;$('notice').classList.toggle('warning',!f||ps.length!==4||filled>0);
}
function update(force=false){const t=currentTime();frame=findFrame(t);const i=frame?frameIndex:-1,ball=ballAt(t);if(force||i!==lastDraw){rebuild(frame);metrics(frame)}if(force||i!==lastDraw||ball.index!==lastBall||video.currentTime!==lastVideoTime){drawOverlay(frame,ball);lastBall=ball.index;lastVideoTime=video.currentTime}lastDraw=i;renderBall(ball,t);
 $('source-time').textContent=fmt(t,true);$('clip-time').textContent=`${fmt(video.currentTime)} / ${fmt(Number.isFinite(video.duration)?video.duration:0)}`;$('timeline').value=video.currentTime;
 $('overlay-label').style.display=$('show-pose').checked&&frame?'':'none';$('scene-note').textContent=frame?'Drag to orbit · Scroll to zoom':'No analyzed pose at this time';
}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);if(data)update()};animate();
new ResizeObserver(()=>{const rect=$('scene').getBoundingClientRect();renderer.setSize(rect.width,rect.height);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix()}).observe($('scene'));
async function loadClip(index){video.pause();data=await fetch(mediaURL('analysis/'+clips[index].file)).then(r=>{if(!r.ok)throw Error('Tracking file unavailable');return r.json()});lastDraw=lastBall=-2;video.src=mediaURL(data.clip.video);video.load();$('dataset-info').textContent=`Pose ${data.clip.hz} Hz · Ball ${data.ball_summary?.source_hz.toFixed(2)||'—'} Hz · TrackNet-Pickleball`;$('coverage').textContent=data.ball_summary?`${data.ball_summary.detected_frames}/${data.ball_summary.total_frames} frames have model detections. ${data.ball_summary.reconstructed_frames||0} frames have estimated height. Coverage is not an accuracy score.`:'Run the ball tracker for this excerpt.';update(true)}
video.addEventListener('loadeddata',()=>update(true));
video.addEventListener('loadedmetadata',()=>{$('timeline').max=video.duration;update(true)});video.addEventListener('seeked',()=>update(true));video.addEventListener('play',()=>{$('play').textContent='Ⅱ Pause'});video.addEventListener('pause',()=>{$('play').textContent='▶ Play'});video.addEventListener('ended',()=>{if($('loop').checked){video.currentTime=0;video.play().catch(()=>{})}});video.addEventListener('error',()=>{$('notice').textContent='Video failed to load. Check that the local media server is running.'});
$('mute').onclick=()=>{video.muted=!video.muted;$('mute').textContent=video.muted?'Unmute':'Mute';$('mute').setAttribute('aria-pressed',String(video.muted))};
$('fullscreen').onclick=()=>{const stage=video.closest('.video-stage');if(document.fullscreenElement)document.exitFullscreen();else stage.requestFullscreen().catch(showError)};
$('play').onclick=()=>video.paused?video.play().catch(e=>{$('notice').textContent=e.message}):video.pause();$('timeline').oninput=e=>{video.currentTime=+e.target.value;update(true)};$('speed').onchange=e=>video.playbackRate=+e.target.value;
for(const [id,d] of [['back',-1],['next',1]])$(id).onclick=()=>{video.pause();video.currentTime=Math.max(0,Math.min(video.duration,video.currentTime+d/data.clip.hz));update(true)};
for(const id of ['show-pose','show-ball','trails'])$(id).onchange=()=>update(true);
for(const [id,pos] of [['perspective',[10,9,18]],['top',[3.048,21,6.71]],['side',[21,5,6.7]]])$(id).onclick=()=>{camera.position.set(...pos);controls.target.set(3.048,0,6.7);controls.update();for(const n of ['perspective','top','side'])$(n).classList.toggle('active',id===n)};
$('clips').onchange=e=>loadClip(+e.target.value).catch(showError);
function showError(e){$('notice').textContent=e.message;$('notice').classList.add('warning')}
try{clips=await fetch(clipsURL).then(r=>{if(!r.ok)throw Error('Clip list unavailable');return r.json()});if(!clips.length)throw Error('No analyzed excerpts yet. Run analyze.py to generate real pose data.');$('clips').innerHTML=clips.map((c,i)=>`<option value="${i}">${fmt(c.start)}–${fmt(c.end)} · ${(c.end-c.start).toFixed(0)}s rally</option>`).join('');await loadClip(0)}catch(e){showError(e)}
