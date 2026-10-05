import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';

const connections=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const coord=p=>[p[0],p[2],p[1]];

export function estimatedBall(segments,t){
  for(const s of segments||[]){
    if(t<s.start||t>s.end) continue;
    const before=s.support_times.filter(x=>x<=t).at(-1),after=s.support_times.find(x=>x>=t);
    if((before!==undefined&&after!==undefined&&after-before>.35)||Math.min(...s.support_times.map(x=>Math.abs(x-t)))>.12) continue;
    const k=s.drag_per_second,dt=t-s.start,d=s.end-s.start,f=-Math.expm1(-k*dt)/k,end=-Math.expm1(-k*d)/k,g=[0,0,-9.81];
    const v=s.p0.map((p,j)=>(s.p1[j]-p-g[j]*(d-end)/k)/end);
    const xyz=s.p0.map((p,j)=>p+f*v[j]+g[j]*(dt-f)/k);
    return {xyz,error:s.reprojection_rmse_px,segment:s};
  }
  return null;
}

function line(points,color,opacity=1){return new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p))),new THREE.LineBasicMaterial({color,transparent:opacity<1,opacity}));}

class CourtView{
  constructor(host,color,onCamera){
    this.host=host;this.color=color;this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#111b1e');
    this.camera=new THREE.PerspectiveCamera(43,1,.1,100);this.camera.position.set(10,9,18);
    this.renderer=new THREE.WebGLRenderer({antialias:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.domElement.setAttribute('aria-label','Interactive 3D court and independently fitted ball flight');host.prepend(this.renderer.domElement);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.target.set(3.048,0,6.7);this.controls.minDistance=5;this.controls.maxDistance=40;this.controls.maxPolarAngle=Math.PI/2-.02;this.controls.update();this.controls.addEventListener('change',()=>onCamera(this));
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x253635,2));const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(6,15,7);this.scene.add(light);
    const ground=new THREE.Mesh(new THREE.PlaneGeometry(70,70),new THREE.MeshStandardMaterial({color:'#142329',roughness:1}));ground.rotation.x=-Math.PI/2;ground.position.set(3,-.025,6.7);this.scene.add(ground);
    for(const [z,length,col] of [[0,13.4112,'#264449'],[4.572,4.2672,'#345b58']]){
      const floor=new THREE.Mesh(new THREE.PlaneGeometry(6.096,length),new THREE.MeshStandardMaterial({color:col,roughness:1}));floor.rotation.x=-Math.PI/2;floor.position.set(3.048,z===0?.001:.003,z+length/2);this.scene.add(floor);
    }
    this.scene.add(line([[0,.025,0],[6.096,.025,0],[6.096,.025,13.4112],[0,.025,13.4112],[0,.025,0]],'#a3c7c4'));
    for(const z of [4.572,8.8392])this.scene.add(line([[0,.025,z],[6.096,.025,z]],'#a3c7c4'));
    for(const [a,b] of [[0,4.572],[8.8392,13.4112]])this.scene.add(line([[3.048,.025,a],[3.048,.025,b]],'#a3c7c4'));
    const net=new THREE.Mesh(new THREE.PlaneGeometry(6.4,.914),new THREE.MeshBasicMaterial({color:'#bac9c7',transparent:true,opacity:.14,side:THREE.DoubleSide}));net.position.set(3.048,.457,6.7056);this.scene.add(net);
    this.scene.add(line([[-.15,.914,6.7056],[6.25,.914,6.7056]],'#d8e4df'));
    for(let x=0;x<6.1;x+=.3)this.scene.add(line([[x,0,6.7056],[x,.91,6.7056]],'#738c88',.3));
    this.players=new THREE.Group();this.scene.add(this.players);this.slots=[];this.poseKey=null;
    this.boneGeometry=new THREE.CylinderGeometry(.04,.04,1,6);this.headGeometry=new THREE.SphereGeometry(.085,10,8);
    this.materials={};for(const team of ['near','far']) for(const filled of [false,true])this.materials[`${team}-${filled}`]=new THREE.MeshBasicMaterial({color:team==='near'?'#a0eacb':'#ffb27f',transparent:filled,opacity:filled?.55:1});
    this.ball=new THREE.Mesh(new THREE.SphereGeometry(.13,16,12),new THREE.MeshBasicMaterial({color}));this.scene.add(this.ball);
    this.shadow=new THREE.Mesh(new THREE.RingGeometry(.1,.14,20),new THREE.MeshBasicMaterial({color,transparent:true,opacity:.65,side:THREE.DoubleSide}));this.shadow.rotation.x=-Math.PI/2;this.scene.add(this.shadow);
    this.ball.visible=this.shadow.visible=false;this.trail=null;this.trailKey='';
    // A billboard keeps an unknown-height marker readable on a small court.
    // Its anchor is the ground projection; the hollow symbol conveys uncertainty.
    const markerCanvas=document.createElement('canvas');markerCanvas.width=markerCanvas.height=64;
    const markerContext=markerCanvas.getContext('2d');markerContext.strokeStyle='#ffffff';markerContext.lineWidth=4;
    markerContext.beginPath();markerContext.arc(32,32,23,0,Math.PI*2);markerContext.stroke();
    markerContext.lineWidth=2;markerContext.beginPath();markerContext.arc(32,32,15,0,Math.PI*2);markerContext.stroke();
    this.detectedMarker=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(markerCanvas),color,depthTest:false,transparent:true}));
    this.scene.add(this.detectedMarker);this.detectedMarker.visible=false;
    this.visible=true;new IntersectionObserver(entries=>{this.visible=entries[0].isIntersecting;}).observe(host);
    new ResizeObserver(()=>{const r=host.getBoundingClientRect();if(r.width&&r.height){this.renderer.setSize(r.width,r.height);this.camera.aspect=r.width/r.height;this.camera.updateProjectionMatrix();this.render();}}).observe(host);
  }

  updatePlayers(pose,show){
    this.players.visible=show;
    if(this.poseKey===pose) return;
    this.poseKey=pose;
    const players=pose?.players||[];
    while(this.slots.length<players.length){
      const group=new THREE.Group(),bones=Array.from({length:connections.length+1},()=>{const mesh=new THREE.Mesh(this.boneGeometry,this.materials['near-false']);group.add(mesh);return mesh;});
      const head=new THREE.Mesh(this.headGeometry,this.materials['near-false']);group.add(head);this.players.add(group);this.slots.push({group,bones,head});
    }
    this.slots.forEach((slot,i)=>{
      const p=players[i];slot.group.visible=!!p;if(!p)return;
      const pts=p.joints3d,material=this.materials[`${p.team}-${p.tracking_status==='interpolated'}`];
      const links=connections.map(([a,b])=>[pts[a],pts[b]]);
      links.push([pts[0],pts[5]&&pts[6]?pts[5].map((n,j)=>(n+pts[6][j])/2):null]);
      slot.bones.forEach((mesh,j)=>{
        const [a,b]=links[j];mesh.visible=!!a&&!!b;if(!mesh.visible)return;
        const va=new THREE.Vector3(...coord(a)),vb=new THREE.Vector3(...coord(b)),direction=vb.clone().sub(va),length=direction.length();
        mesh.material=material;mesh.position.copy(va.add(vb).multiplyScalar(.5));mesh.scale.y=length;mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction.normalize());
      });
      slot.head.visible=!!pts[0];slot.head.material=material;if(pts[0])slot.head.position.set(...coord(pts[0]));
    });
  }

  update(segments,t,pose,options){
    this.updatePlayers(pose,options.players);
    const ball=estimatedBall(segments,t);this.ball.visible=this.shadow.visible=!!ball;
    if(ball){this.ball.position.set(...coord(ball.xyz));this.shadow.position.set(ball.xyz[0],.035,ball.xyz[1]);}
    const projection=options.projection;
    this.detectedMarker.visible=!!projection&&(!ball||projection.show_with_fit);
    if(this.detectedMarker.visible)this.detectedMarker.position.set(projection.ground_xy[0],.06,projection.ground_xy[1]);
    const key=`${Math.round(t*30)}-${options.trails}-${segments?.length}`;
    if(key!==this.trailKey){
      if(this.trail){this.scene.remove(this.trail);this.trail.geometry.dispose();this.trail.material.dispose();this.trail=null;}
      if(ball&&options.trails){
        const points=[];
        // Separate line pairs prevent drawing across unsupported gaps.
        let previous=null;for(let s=Math.max(ball.segment.start,t-.65);s<=t+.0001;s+=1/60){const p=estimatedBall(segments,s);if(p&&previous)points.push(new THREE.Vector3(...coord(previous.xyz)),new THREE.Vector3(...coord(p.xyz)));previous=p;}
        if(points.length){this.trail=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:this.color,transparent:true,opacity:.8}));this.scene.add(this.trail);}
      }
      this.trailKey=key;
    }
    this.render();return ball;
  }
  render(){if(this.visible){
    if(this.detectedMarker.visible){const size=10*2*this.camera.position.distanceTo(this.detectedMarker.position)*Math.tan(this.camera.fov*Math.PI/360)/Math.max(1,this.host.clientHeight);this.detectedMarker.scale.set(size,size,1);}
    this.renderer.render(this.scene,this.camera);
  }}
}

export class ReconstructionViews{
  constructor(){this.views=new Map();this.syncing=false;const animate=()=>{requestAnimationFrame(animate);for(const view of this.views.values())view.render();};animate();}
  add(id,host,color){const view=new CourtView(host,color,source=>this.sync(source));this.views.set(id,view);if(this.views.size>1)this.sync(this.views.values().next().value);}
  sync(source){if(this.syncing)return;this.syncing=true;for(const view of this.views.values()){if(view===source)continue;view.camera.position.copy(source.camera.position);view.controls.target.copy(source.controls.target);view.controls.update();}this.syncing=false;}
  preset(name){const positions={perspective:[10,9,18],top:[3.048,21,6.71],side:[21,5,6.7]};const first=this.views.values().next().value;if(!first)return;first.camera.position.set(...positions[name]);first.controls.target.set(3.048,0,6.7);first.controls.update();this.sync(first);}
  update(id,segments,t,pose,options){return this.views.get(id)?.update(segments,t,pose,options)||null;}
  reset(){for(const view of this.views.values()){view.poseKey=null;view.trailKey='';view.update([],0,null,{players:true,trails:false});}}
}
