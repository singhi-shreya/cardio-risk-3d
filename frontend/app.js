import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

/* ============================================================== helpers */
const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
const pct = (p) => `${Math.round(p * 100)}%`;
const fmt = (v) => (Number.isInteger(Number(v)) ? String(Number(v)) : Number(v).toFixed(1));
const cssRisk = (p) => `hsl(${Math.round((1 - Math.min(Math.max(p, 0), 1)) * 119)} 60% 42%)`;
const bandWord = { low: "Lower risk", moderate: "Moderate risk", high: "Higher risk" };

const VESSELS = ["LAD", "LCX", "RCA"];
const TARGET_INFO = {
  CAD: { name: "Overall", about: "Overall coronary artery disease: at least one major vessel narrowed by 50% or more.", disease: "coronary artery disease" },
  LAD: { name: "LAD", about: "Left Anterior Descending: supplies blood to the front of the heart.", disease: "LAD narrowing" },
  LCX: { name: "LCX", about: "Left Circumflex: supplies blood to the side and back of the heart.", disease: "LCX narrowing" },
  RCA: { name: "RCA", about: "Right Coronary Artery: supplies blood to the right side and bottom of the heart.", disease: "RCA narrowing" },
};

// Sample people for the demo buttons (matched to dataset columns by normalised name)
const PERSONAS = {
  alex: { age: 58, sex: "male", bp: 142, pr: 78, currentsmoker: "yes", dm: "no", typicalchestpain: "yes", ldl: 140, fbs: 105, eftte: 45 },
  priya: { age: 45, sex: "female", bp: 116, pr: 68, currentsmoker: "no", dm: "no", typicalchestpain: "no", ldl: 95, fbs: 88, eftte: 60 },
};
// Inputs offered in the "What if...?" box (first ones that exist are used)
const WHATIF = ["ldl", "bp", "fbs", "hdl", "eftte", "pr"];

const state = {
  schema: [], groups: [], byName: new Map(), byNorm: new Map(), values: {}, ctl: {}, helpOpen: new Set(),
  samples: [], metrics: null, result: null, selected: "CAD", mode: "simple", metricsMode: "holdout",
  truth: null, whatField: null, whatBase: null, whatOrig: null,
};

/* ============================================================== 3D scene */
const wrap = $("#viewer");
const canvas = $("#scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5)); // smooth without a dedicated GPU
renderer.setClearColor(0x000000, 0);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
const HOME_POS = new THREE.Vector3(0.3, 0.25, 6.0);
const HOME_TARGET = new THREE.Vector3(0, 0.15, 0);
camera.position.copy(HOME_POS);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.minDistance = 2.4;
controls.maxDistance = 10;
controls.target.copy(HOME_TARGET);
let swaying = true; // gentle idle sway until the person grabs the heart
controls.addEventListener("start", () => { swaying = false; });

// Soft, even lighting
scene.add(new THREE.HemisphereLight(0xffffff, 0xcad8e8, 1.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
keyLight.position.set(3, 4, 5);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffd9de, 0.45);
fillLight.position.set(-4, -1, 3);
scene.add(fillLight);

/* ---- rib cage (optional context) ---- */
const body = new THREE.Group();
body.visible = false;
scene.add(body);
const ribMat = new THREE.MeshStandardMaterial({ color: 0xb9c7d6, transparent: true, opacity: 0.35, roughness: 0.7, depthWrite: false });
for (let i = 0; i < 7; i++) {
  const r = 1.65 + 0.12 * Math.sin((i / 6) * Math.PI);
  const rib = new THREE.Mesh(new THREE.TorusGeometry(r, 0.03, 8, 64, Math.PI), ribMat);
  rib.rotation.x = Math.PI / 2;
  rib.scale.set(1, 0.6, 1);
  rib.position.set(0, 1.5 - i * 0.5, 0);
  body.add(rib);
}
const sternum = new THREE.Mesh(new THREE.BoxGeometry(0.26, 2.9, 0.07), ribMat);
sternum.position.set(0, 0.05, 1.0);
body.add(sternum);

/* ---- procedural heart (swap in a real mesh: frontend/models/heart.glb) ---- */
const heart = new THREE.Group();
heart.rotation.y = -0.25;
scene.add(heart);

const heartMat = new THREE.MeshPhysicalMaterial({
  color: 0xf3c0c7, roughness: 0.42, transparent: true, opacity: 0.6, depthWrite: false,
  clearcoat: 0.5, clearcoatRoughness: 0.35, sheen: 0.6, sheenColor: new THREE.Color(0xffe3e6),
});
let heartPick = [];
const proceduralParts = [];

function ellipsoid(center, radii, rotZ = 0) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 56, 40), heartMat);
  m.scale.set(...radii);
  m.position.set(...center);
  m.rotation.z = rotZ;
  heart.add(m);
  proceduralParts.push(m);
}
function greatVessel(points, radius, color) {
  const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)));
  const m = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 48, radius, 20, false),
    new THREE.MeshPhysicalMaterial({ color, roughness: 0.45, transparent: true, opacity: 0.72, depthWrite: false, clearcoat: 0.4 })
  );
  heart.add(m);
  proceduralParts.push(m);
}
ellipsoid([0.3, -0.25, -0.05], [0.62, 0.85, 0.55], 0.4); // left ventricle
ellipsoid([-0.35, -0.2, 0.3], [0.5, 0.7, 0.45], 0.25); // right ventricle
ellipsoid([-0.7, 0.3, 0.0], [0.38, 0.5, 0.38]); // right atrium
ellipsoid([0.35, 0.55, -0.4], [0.4, 0.3, 0.35]); // left atrium
greatVessel([[0, 0.55, 0], [0, 1.1, 0], [0.15, 1.45, 0], [0.5, 1.55, 0], [0.8, 1.3, 0]], 0.2, 0xe9a3a0); // aorta
greatVessel([[-0.05, 0.5, 0.4], [0.15, 1.1, 0.3], [0.5, 1.25, 0.1]], 0.16, 0xa9bce6); // pulmonary trunk
heartPick = proceduralParts.slice();

/* ---- coronary arteries: one mesh per target vessel ---- */
const NEUTRAL = new THREE.Color(0x9fb1c2);
const ARTERIES = {
  LAD: { pts: [[0.12, 0.5, 0.4], [0.05, 0.2, 0.62], [0.03, -0.2, 0.7], [0.15, -0.6, 0.64], [0.35, -0.9, 0.5], [0.5, -1.0, 0.35]], label: [-0.2, -0.75, 1.0] },
  LCX: { pts: [[0.2, 0.5, 0.35], [0.5, 0.45, 0.3], [0.78, 0.3, 0.1], [0.9, 0.0, -0.15], [0.85, -0.35, -0.4]], label: [1.35, 0.4, 0.25] },
  RCA: { pts: [[-0.2, 0.55, 0.35], [-0.5, 0.4, 0.45], [-0.75, 0.05, 0.42], [-0.8, -0.4, 0.35], [-0.65, -0.85, 0.3], [-0.4, -1.05, 0.1]], label: [-1.3, -0.15, 0.6] },
};

function makeLabel(text) {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 96;
  const tex = new THREE.CanvasTexture(c);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sprite.scale.set(0.95, 0.36, 1);
  sprite.renderOrder = 10;
  sprite.userData.draw = (t, color = "#3f7cac") => {
    const g = c.getContext("2d");
    g.clearRect(0, 0, 256, 96);
    g.fillStyle = "rgba(255,255,255,0.95)";
    g.fillRect(4, 8, 248, 80);
    g.lineWidth = 6;
    g.strokeStyle = color;
    g.strokeRect(4, 8, 248, 80);
    g.fillStyle = "#1e2a3a";
    g.font = "bold 40px Nunito, Segoe UI, Arial, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(t, 128, 50);
    tex.needsUpdate = true;
  };
  sprite.userData.draw(text);
  return sprite;
}

const art = {};
const hitMeshes = [];
for (const [name, def] of Object.entries(ARTERIES)) {
  const curve = new THREE.CatmullRomCurve3(def.pts.map((p) => new THREE.Vector3(...p)));
  const mat = new THREE.MeshStandardMaterial({ color: NEUTRAL, emissive: NEUTRAL, emissiveIntensity: 0.35, roughness: 0.4 });
  const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 72, 0.055, 16, false), mat);
  const halo = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 72, 0.095, 16, false),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55, depthWrite: false })
  );
  halo.visible = false;
  const hit = new THREE.Mesh(new THREE.TubeGeometry(curve, 32, 0.16, 6, false), new THREE.MeshBasicMaterial({ visible: false }));
  hit.userData.target = name;
  const label = makeLabel(name);
  label.position.set(...def.label);
  heart.add(mesh, halo, hit, label);
  art[name] = { curve, mat, halo, label, p: null, color: NEUTRAL.clone() };
  hitMeshes.push(hit);
}

/* ---- optional real anatomical mesh: drop a heart.glb into frontend/models/ ---- */
async function tryLoadMesh() {
  try {
    const head = await fetch("models/heart.glb", { method: "HEAD" });
    if (!head.ok) return;
    const gltf = await new GLTFLoader().loadAsync("models/heart.glb");
    const obj = gltf.scene;
    const box = new THREE.Box3().setFromObject(obj);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const s = 2.6 / Math.max(size.x, size.y, size.z);
    obj.scale.setScalar(s);
    obj.position.copy(center).multiplyScalar(-s);
    const meshes = [];
    obj.traverse((o) => {
      if (o.isMesh) {
        o.material = o.material.clone();
        o.material.transparent = true;
        o.material.opacity = 0.6;
        o.material.depthWrite = false;
        meshes.push(o);
      }
    });
    proceduralParts.forEach((m) => (m.visible = false));
    heart.add(obj);
    heartPick = meshes;
    setStatus("Custom heart mesh loaded");
  } catch (err) {
    console.warn("Custom mesh not loaded:", err);
  }
}

/* ---- camera helpers ---- */
const focus = { target: HOME_TARGET.clone(), pos: null, frames: 0 };
function focusOn(name) {
  if (name === "CAD" || !art[name]) {
    focus.target.copy(HOME_TARGET);
  } else {
    heart.updateMatrixWorld(true);
    const p = heart.localToWorld(art[name].curve.getPoint(0.5).clone()).multiplyScalar(0.6);
    focus.target.copy(p).add(HOME_TARGET.clone().multiplyScalar(0.4));
  }
  focus.pos = null;
  focus.frames = 50;
}
function resetView() {
  focus.target.copy(HOME_TARGET);
  focus.pos = HOME_POS.clone();
  focus.frames = 70;
  swaying = true;
}

/* ---- apply model output to the anatomy ---- */
const HEART_LOW = new THREE.Color(0xf3c0c7);
const HEART_HIGH = new THREE.Color(0xd9808a);
const heartTarget = HEART_LOW.clone();
const riskColor = (p) => new THREE.Color().setHSL((1 - Math.min(Math.max(p, 0), 1)) * 0.33, 0.62, 0.5);
function applyResultTo3D(res) {
  for (const v of VESSELS) {
    const p = res.vessels[v].probability;
    art[v].p = p;
    art[v].color.copy(riskColor(p));
    art[v].label.userData.draw(`${v} ${pct(p)}`, cssRisk(p));
  }
  heartTarget.copy(HEART_LOW).lerp(HEART_HIGH, res.cad.probability);
}

/* ---- picking: click an artery -> that vessel; click the heart -> overall ---- */
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = $("#tooltip");
function pick(ev) {
  const r = canvas.getBoundingClientRect();
  mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const a = ray.intersectObjects(hitMeshes, false);
  if (a.length) return a[0].object.userData.target;
  return ray.intersectObjects(heartPick, false).length ? "CAD" : null;
}
let down = null;
canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener("pointerup", (e) => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  down = null;
  if (moved < 5) {
    const t = pick(e);
    if (t) select(t, { fly: true });
  }
});
canvas.addEventListener("pointermove", (e) => {
  if (down) { tooltip.hidden = true; return; }
  const t = pick(e);
  canvas.style.cursor = t ? "pointer" : "grab";
  if (!t) { tooltip.hidden = true; return; }
  const p = t === "CAD" ? state.result?.cad.probability : art[t].p;
  const name = t === "CAD" ? "Heart, overall" : `${t} narrowing`;
  tooltip.textContent = p == null ? `${name}: waiting` : `${name}: ${pct(p)}`;
  const r = wrap.getBoundingClientRect();
  tooltip.style.left = `${e.clientX - r.left}px`;
  tooltip.style.top = `${e.clientY - r.top}px`;
  tooltip.hidden = false;
});
canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });

/* ---- resize + render loop ---- */
new ResizeObserver(() => {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}).observe(wrap);

const clock = new THREE.Clock();
let time = 0, phase = 0;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function currentPulse() {
  const f = state.byNorm.get("pr");
  const v = f ? Number(state.values[f.name]) : NaN;
  return v >= 30 && v <= 200 ? v : 70;
}
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  time += dt;
  if (!reduceMotion) {
    phase += dt * (currentPulse() / 60) * Math.PI * 2; // the heart beats at the entered pulse rate
    heart.scale.setScalar(1 + 0.035 * Math.pow(Math.max(0, Math.sin(phase)), 4));
    if (swaying) heart.rotation.y = -0.25 + 0.3 * Math.sin(time * 0.5);
  }
  for (const [name, a] of Object.entries(art)) {
    a.mat.color.lerp(a.color, 0.12);
    a.mat.emissive.copy(a.mat.color);
    const sel = state.selected === name;
    const pulse = !reduceMotion && a.p != null && a.p >= 0.5 ? 0.2 * (0.5 + 0.5 * Math.sin(time * 4)) : 0;
    a.mat.emissiveIntensity = 0.3 + pulse + (sel ? 0.2 : 0);
    a.halo.visible = sel;
  }
  heartMat.color.lerp(heartTarget, 0.1);
  if (focus.frames > 0) {
    focus.frames--;
    controls.target.lerp(focus.target, 0.12);
    if (focus.pos) camera.position.lerp(focus.pos, 0.1);
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

/* ============================================================== labels & wording */
function setStatus(t) { $("#status").textContent = t; }
const labelOf = (f) => (state.mode === "clinician" ? f.name : f.label);
function optLabel(f, o) {
  if (f.kind === "yesno") return String(o) === String(f.yes) ? "Yes" : "No";
  return f.option_labels?.[norm(o)] ?? String(o);
}
function fieldOptions(f) {
  return f.kind === "yesno" ? [f.yes, f.no] : f.options;
}

// Turn a model feature (maybe one-hot, e.g. "Sex_Male") into a readable label + value text
function describe(feature, value) {
  const f = state.byName.get(feature);
  if (f) {
    if (f.kind === "number") return { label: labelOf(f), text: `${fmt(value)}${f.unit ? " " + f.unit : ""}`, yes: null };
    if (f.kind === "yesno") { const y = String(value) === String(f.yes); return { label: labelOf(f), text: y ? "Yes" : "No", yes: y }; }
    return { label: labelOf(f), text: optLabel(f, value), yes: null };
  }
  const base = state.schema.find((s) => s.type === "category" && feature.startsWith(s.name + "_"));
  if (base) {
    const level = feature.slice(base.name.length + 1);
    const on = value >= 0.5;
    if (base.kind === "yesno") return { label: labelOf(base), text: on ? "Yes" : "No", yes: on };
    return { label: `${labelOf(base)}: ${optLabel(base, level)}`, text: on ? "yes" : "no", yes: on };
  }
  return { label: feature, text: fmt(value), yes: null };
}
function phrase(c) {
  const d = describe(c.feature, c.value);
  if (d.yes === true) return d.label;
  if (d.yes === false) return `no ${d.label.toLowerCase()}`;
  return d.text && d.text !== d.label ? `${d.label} (${d.text})` : d.label;
}
const joinAnd = (a) => (a.length <= 1 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1]);

/* ============================================================== form */
function setValue(name, v, fromControl = false) {
  state.values[name] = v;
  if (!fromControl) state.ctl[name]?.(v);
  if (state.whatField === name) syncWhatSlider();
  updateProgress();
  schedulePredict();
}

function renderField(f, asSlider) {
  const row = el("div", "field");
  const head = el("div", "frow");
  const lab = el("span", "flabel");
  lab.appendChild(el("span", null, esc(labelOf(f))));
  const hasHelp = f.help || f.ref;
  if (hasHelp) {
    const q = el("button", "q", "?");
    q.type = "button";
    q.setAttribute("aria-label", `What is ${f.label}?`);
    q.addEventListener("click", () => {
      const box = row.querySelector(".helpbox");
      if (box) { box.remove(); state.helpOpen.delete(f.name); }
      else { row.appendChild(helpBox(f)); state.helpOpen.add(f.name); }
    });
    lab.appendChild(q);
  }
  head.appendChild(lab);
  row.appendChild(head);

  if (f.kind === "yesno" || f.kind === "choice") {
    const wrapP = el("span", "pills");
    const btns = fieldOptions(f).map((o) => {
      const b = el("button", "pill");
      b.type = "button";
      b.textContent = optLabel(f, o);
      b.dataset.v = String(o);
      b.addEventListener("click", () => setValue(f.name, String(state.values[f.name]) === String(o) ? "" : o));
      wrapP.appendChild(b);
      return b;
    });
    head.appendChild(wrapP);
    state.ctl[f.name] = (v) => btns.forEach((b) => b.classList.toggle("on", v !== "" && b.dataset.v === String(v)));
  } else if (f.kind === "select") {
    const s = el("select");
    s.appendChild(new Option("Not sure", ""));
    f.options.forEach((o) => s.appendChild(new Option(optLabel(f, o), String(o))));
    s.addEventListener("change", () => setValue(f.name, s.value === "" ? "" : f.type === "discrete" ? Number(s.value) : s.value, true));
    head.appendChild(s);
    state.ctl[f.name] = (v) => { s.value = v === "" ? "" : String(v); };
  } else if (asSlider) {
    const val = el("b", "fval");
    head.appendChild(val);
    const r = el("input");
    r.type = "range";
    r.min = f.min; r.max = f.max;
    r.step = f.max - f.min > 40 ? 1 : 0.1;
    r.setAttribute("aria-label", f.label);
    r.addEventListener("input", () => setValue(f.name, Number(r.value), true));
    row.appendChild(r);
    state.ctl[f.name] = (v) => {
      r.value = v === "" ? f.default : v;
      val.textContent = v === "" ? "not set" : `${fmt(v)}${f.unit ? " " + f.unit : ""}`;
    };
    // keep the label text of the value fresh when the slider itself moved
    r.addEventListener("input", () => { val.textContent = `${fmt(r.value)}${f.unit ? " " + f.unit : ""}`; });
  } else {
    const n = el("input");
    n.type = "number";
    n.step = "any";
    n.min = f.min; n.max = f.max;
    n.placeholder = f.unit || "";
    n.setAttribute("aria-label", f.label);
    n.addEventListener("input", () => setValue(f.name, n.value === "" ? "" : Number(n.value), true));
    head.appendChild(n);
    state.ctl[f.name] = (v) => { if (document.activeElement !== n) n.value = v === "" ? "" : v; };
  }
  if (state.helpOpen.has(f.name) && hasHelp) row.appendChild(helpBox(f));
  return row;
}
function helpBox(f) {
  const range = f.ref ? ` Typical: ${f.ref[0]} to ${f.ref[1]}${f.unit ? " " + f.unit : ""}.` : "";
  return el("div", "helpbox", esc((f.help || "") + range));
}

function buildForm() {
  state.ctl = {};
  const keyFields = state.schema.filter((f) => f.key_rank !== null).sort((a, b) => a.key_rank - b.key_rank);
  const rest = state.schema.filter((f) => f.key_rank === null);
  const kf = $("#key-fields");
  kf.innerHTML = "";
  keyFields.forEach((f) => kf.appendChild(renderField(f, true)));
  const mf = $("#more-fields");
  mf.innerHTML = "";
  for (const g of state.groups) {
    const fs = rest.filter((f) => f.group === g);
    if (!fs.length) continue;
    const d = el("details", "grp");
    d.appendChild(el("summary", null, `${esc(g)} (${fs.length})`));
    fs.forEach((f) => d.appendChild(renderField(f, false)));
    mf.appendChild(d);
  }
  $("#btn-more").textContent = `${mf.hidden ? "Add" : "Hide"} more details (${rest.length})`;
  syncAll();
}
function syncAll() {
  for (const f of state.schema) state.ctl[f.name]?.(state.values[f.name] ?? "");
  updateProgress();
  syncWhatSlider();
}
function updateProgress() {
  const n = state.schema.filter((f) => state.values[f.name] !== "" && state.values[f.name] != null).length;
  $("#progress-text").textContent = `You have filled in ${n} of ${state.schema.length} details`;
  $("#progress-bar").style.width = `${(n / Math.max(state.schema.length, 1)) * 100}%`;
}

function resetValues() {
  state.values = {};
  for (const f of state.schema) state.values[f.name] = f.key_rank !== null ? f.default : "";
}
function mapValue(f, v) {
  if (v === "yes") return f.kind === "yesno" ? f.yes : undefined;
  if (v === "no") return f.kind === "yesno" ? f.no : undefined;
  if (v === "male") return f.options?.find((o) => /^m/i.test(String(o)));
  if (v === "female") return f.options?.find((o) => /^f/i.test(String(o)));
  if (typeof v === "number" && f.kind === "number") return Math.min(Math.max(v, f.min), f.max);
  return v;
}
function applyPersona(p) {
  resetValues();
  for (const [k, v] of Object.entries(p)) {
    const f = state.byNorm.get(k);
    const m = f ? mapValue(f, v) : undefined;
    if (m !== undefined) state.values[f.name] = m;
  }
  state.truth = null;
  finishLoad();
}
function applySample(s) {
  state.values = {};
  for (const f of state.schema) state.values[f.name] = s.features[f.name] ?? "";
  state.truth = s.truth;
  finishLoad();
}
function finishLoad() {
  state.whatBase = null;
  $("#wi-reset").hidden = true;
  syncAll();
  showTruth();
  predict();
}
function showTruth() {
  const t = $("#truth");
  if (!state.truth) { t.hidden = true; return; }
  const yn = (v) => (v ? "narrowed" : "clear");
  const x = state.truth;
  t.textContent = `Dataset (angiography) result: overall ${x.CAD ? "CAD" : "no CAD"} · LAD ${yn(x.LAD)} · LCX ${yn(x.LCX)} · RCA ${yn(x.RCA)}`;
  t.hidden = false;
}

/* ============================================================== prediction */
let timer = null, reqId = 0;
function schedulePredict() {
  clearTimeout(timer);
  timer = setTimeout(predict, 350);
}
async function predict() {
  const id = ++reqId;
  setStatus("Thinking…");
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: state.values }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const res = await r.json();
    if (id !== reqId) return;
    state.result = res;
    applyResultTo3D(res);
    renderAll();
    setStatus("Up to date");
  } catch (err) {
    console.error(err);
    setStatus("Something went wrong");
  }
}

/* ============================================================== rendering */
function renderAll() {
  renderChips();
  renderOverall();
  renderStory();
  renderDrawer();
  renderWhatDelta();
}

function renderChips() {
  const res = state.result;
  const c = $("#chips");
  c.innerHTML = "";
  const items = [["CAD", res.cad], ...VESSELS.map((v) => [v, res.vessels[v]])];
  for (const [t, r] of items) {
    const b = el("button", "chip" + (state.selected === t ? " sel" : ""));
    b.type = "button";
    b.innerHTML = `<span class="d" style="background:${cssRisk(r.probability)}"></span>${TARGET_INFO[t].name} ${pct(r.probability)}`;
    b.addEventListener("click", () => select(t, { fly: true }));
    c.appendChild(b);
  }
}

function renderOverall() {
  const res = state.result;
  const p = res.cad.probability;
  const cov = res.coverage;
  const confText = { limited: "limited", moderate: "moderate", good: "good" }[cov.level];
  $("#overall").innerHTML = `
    <p class="res-label">Overall heart check</p>
    <div class="headline ${res.cad.band}">${bandWord[res.cad.band]}</div>
    <div>About a ${pct(p)} chance of coronary artery disease, as estimated by the model</div>
    <div class="meter"><i style="width:${p * 100}%;background:${cssRisk(p)}"></i></div>
    <p class="muted">Confidence: ${confText}, based on ${cov.provided} of ${cov.total} details${cov.level === "good" ? "" : ". Adding more makes it more reliable"}</p>`;
}

function renderStory() {
  const res = state.result;
  const sel = state.selected;
  const info = TARGET_INFO[sel];
  const contribs = res.explanations[sel].contributions;
  const ups = contribs.filter((c) => c.shap > 0).slice(0, 2);
  const downs = contribs.filter((c) => c.shap < 0).slice(0, 1);
  const where = sel === "CAD" ? "the overall risk" : `the ${sel} risk`;
  let text = ups.length ? `${joinAnd(ups.map(phrase))} raised ${where} the most.` : `No single detail pushed ${where} up much.`;
  text = text.charAt(0).toUpperCase() + text.slice(1);
  if (downs.length) text += ` ${(() => { const s = phrase(downs[0]); return s.charAt(0).toUpperCase() + s.slice(1); })()} helped lower it.`;

  let sim = "";
  const sp = res.similar;
  if (sp) {
    const labels = sp.labels[sel];
    const n = labels.reduce((a, b) => a + b, 0);
    sim = `<h2 style="margin-top:16px">Patients like this</h2>
      <div class="dots" aria-hidden="true">${labels.map((v) => `<span class="d ${v ? "" : "no"}" style="${v ? "background:var(--high)" : ""}"></span>`).join("")}</div>
      <p class="muted">${n} of the ${sp.k} most similar patients in the dataset had ${info.disease}. Matched on the details you filled in.</p>`;
  }
  $("#story").innerHTML = `<h2>What stood out: ${info.name}</h2><p>${esc(text)}</p>
    <p class="muted">${esc(info.about)}</p>${sim}`;
}

/* ---- details drawer ---- */
function renderDrawer() {
  const res = state.result;
  const sel = state.selected;
  const tabs = $("#drawer-tabs");
  tabs.innerHTML = "";
  for (const t of ["CAD", ...VESSELS]) {
    const b = el("button", t === sel ? "on" : "", TARGET_INFO[t].name);
    b.type = "button";
    b.addEventListener("click", () => select(t, { fly: true }));
    tabs.appendChild(b);
  }
  const ex = res.explanations[sel];
  $("#why-title").textContent = `Why: ${TARGET_INFO[sel].name === "Overall" ? "overall CAD" : sel + " narrowing"}`;
  const top = ex.contributions.slice(0, 10);
  const max = Math.max(...top.map((c) => Math.abs(c.shap)), 1e-9);
  $("#drivers").innerHTML =
    top.map((c) => {
      const d = describe(c.feature, c.value);
      const up = c.shap > 0;
      return `<div class="drv" title="${esc(c.feature)}: SHAP ${c.shap}">
        <div class="drv-name">${esc(d.label)} <small>${esc(d.text)}</small></div>
        <div class="drv-bar"><div class="mid"></div><div class="fill ${up ? "up" : "down"}" style="width:${(Math.abs(c.shap) / max) * 50}%"></div></div>
        <div class="drv-share">${up ? "+" : "−"}${c.share.toFixed(1)}%</div></div>`;
    }).join("") + `<div class="axis-note"><span>◀ lowers risk</span><span>raises risk ▶</span></div>`;

  let rows = "";
  for (const f of state.schema.filter((x) => x.vital)) {
    const raw = state.values[f.name];
    const has = raw !== "" && raw != null && !Number.isNaN(Number(raw));
    const c = ex.contributions.find((x) => x.feature === f.name);
    let cls = "na";
    if (has && f.ref) cls = Number(raw) >= f.ref[0] && Number(raw) <= f.ref[1] ? "ok" : "out";
    rows += `<tr><td><span class="dot ${cls}"></span>${esc(labelOf(f))}</td>
      <td class="r">${has ? fmt(raw) : "–"} <small>${esc(f.unit || "")}</small></td>
      <td class="r">${f.ref ? `${f.ref[0]}–${f.ref[1]}` : "–"}</td>
      <td class="r">${c ? `${c.shap > 0 ? "+" : "−"}${c.share.toFixed(1)}%` : "–"}</td></tr>`;
  }
  $("#vitals").innerHTML = `<tr><th>Measurement</th><th class="r">Value</th><th class="r">Typical</th><th class="r">Share</th></tr>${rows}`;
}
function renderMetrics() {
  const m = state.metrics;
  if (!m) return;
  const f3 = (v) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(3));
  let rows = "";
  for (const [t, info] of Object.entries(m.targets)) {
    const x = info[state.metricsMode];
    rows += `<tr><td><strong>${t}</strong></td><td>${esc(info.model.replace("_", " "))}</td><td class="r">${f3(x.accuracy)}</td>
      <td class="r">${f3(x.precision)}</td><td class="r">${f3(x.recall)}</td><td class="r">${f3(x.f1)}</td><td class="r">${f3(x.roc_auc)}</td></tr>`;
  }
  $("#metrics").innerHTML = `<tr><th>Target</th><th>Model</th><th class="r">Acc</th><th class="r">Prec</th><th class="r">Recall</th><th class="r">F1</th><th class="r">AUC</th></tr>${rows}`;
  document.querySelectorAll("#metric-mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === state.metricsMode));
}
function openDrawer(open) {
  $("#drawer").classList.toggle("open", open);
  $("#drawer").setAttribute("aria-hidden", String(!open));
  $("#backdrop").hidden = !open;
}

/* ---- "What if...?" ---- */
function buildWhatIf() {
  const fields = WHATIF.map((k) => state.byNorm.get(k)).filter((f) => f && f.kind === "number");
  if (!fields.length) { $("#whatif").hidden = true; return; }
  const sel = $("#wi-field");
  sel.innerHTML = "";
  fields.slice(0, 4).forEach((f) => sel.appendChild(new Option(f.label, f.name)));
  state.whatField = sel.value;
  sel.addEventListener("change", () => {
    state.whatBase = null; state.whatOrig = null; $("#wi-reset").hidden = true;
    state.whatField = sel.value;
    syncWhatSlider();
    renderWhatDelta();
  });
  const sl = $("#wi-slider");
  sl.addEventListener("input", () => {
    if (!state.whatBase && state.result) {
      state.whatBase = { CAD: state.result.cad.probability, ...Object.fromEntries(VESSELS.map((v) => [v, state.result.vessels[v].probability])) };
      state.whatOrig = state.values[state.whatField];
      $("#wi-reset").hidden = false;
    }
    setValue(state.whatField, Number(sl.value));
  });
  $("#wi-reset").addEventListener("click", () => {
    if (state.whatOrig !== null && state.whatOrig !== undefined) setValue(state.whatField, state.whatOrig);
    state.whatBase = null; state.whatOrig = null;
    $("#wi-reset").hidden = true;
    renderWhatDelta();
  });
  syncWhatSlider();
}
function syncWhatSlider() {
  const f = state.byName.get(state.whatField);
  if (!f) return;
  const sl = $("#wi-slider");
  sl.min = f.min; sl.max = f.max; sl.step = f.max - f.min > 40 ? 1 : 0.1;
  const v = state.values[f.name];
  sl.value = v === "" || v == null ? f.default : v;
  $("#wi-val").textContent = `${fmt(sl.value)}${f.unit ? " " + f.unit : ""}`;
}
function renderWhatDelta() {
  const box = $("#wi-delta");
  if (!state.whatBase || !state.result) { box.innerHTML = ""; return; }
  const now = { CAD: state.result.cad.probability, ...Object.fromEntries(VESSELS.map((v) => [v, state.result.vessels[v].probability])) };
  box.innerHTML = ["CAD", ...VESSELS].map((t) => {
    const d = Math.round((now[t] - state.whatBase[t]) * 100);
    const arrow = d > 0 ? "▲" : d < 0 ? "▼" : "•";
    return `<span>${TARGET_INFO[t].name} ${pct(state.whatBase[t])} → ${pct(now[t])} ${arrow}${d === 0 ? "" : Math.abs(d)}</span>`;
  }).join("");
}

/* ---- selection (shared by 3D clicks and the dashboard) ---- */
function select(target, { fly = false } = {}) {
  state.selected = target;
  if (state.result) { renderChips(); renderStory(); renderDrawer(); }
  if (fly) focusOn(target);
}

/* ============================================================== wiring */
$("#personas").addEventListener("click", (e) => {
  const p = e.target.dataset?.p;
  if (!p) return;
  if (p === "random") { if (state.samples.length) applySample(state.samples[Math.floor(Math.random() * state.samples.length)]); }
  else if (p === "reset") { resetValues(); state.truth = null; finishLoad(); }
  else applyPersona(PERSONAS[p]);
});
$("#btn-more").addEventListener("click", () => {
  const mf = $("#more-fields");
  mf.hidden = !mf.hidden;
  $("#btn-more").setAttribute("aria-expanded", String(!mf.hidden));
  $("#btn-more").textContent = $("#btn-more").textContent.replace(/^(Add|Hide)/, mf.hidden ? "Add" : "Hide");
});
$("#view-mode").addEventListener("click", (e) => {
  const m = e.target.dataset?.mode;
  if (!m) return;
  state.mode = m;
  document.querySelectorAll("#view-mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
  buildForm();
  if (state.result) renderAll();
  if (m === "clinician") { $("#more-fields").hidden = false; openDrawer(true); }
});
$("#btn-details").addEventListener("click", () => openDrawer(true));
$("#drawer-close").addEventListener("click", () => openDrawer(false));
$("#backdrop").addEventListener("click", () => openDrawer(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") openDrawer(false); });
$("#btn-view").addEventListener("click", resetView);
$("#tg-ribs").addEventListener("change", (e) => { body.visible = e.target.checked; });
$("#tg-labels").addEventListener("change", (e) => { Object.values(art).forEach((a) => (a.label.visible = e.target.checked)); });
$("#metric-mode").addEventListener("click", (e) => {
  if (e.target.dataset?.mode) { state.metricsMode = e.target.dataset.mode; renderMetrics(); }
});

async function init() {
  try {
    const [schema, metrics, samples] = await Promise.all(
      ["/api/schema", "/api/metrics", "/api/samples"].map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.json(); }))
    );
    state.schema = schema.features;
    state.groups = schema.groups;
    state.byName = new Map(state.schema.map((f) => [f.name, f]));
    state.byNorm = new Map(state.schema.map((f) => [norm(f.name), f]));
    state.metrics = metrics;
    state.samples = samples;
    resetValues();
    buildForm();
    buildWhatIf();
    renderMetrics();
    tryLoadMesh();
    await predict();
  } catch (err) {
    console.error(err);
    setStatus("Backend offline");
    $("#overall").textContent = "Cannot reach the API. Start it with: uvicorn backend.app:app --reload";
  }
}
init();
