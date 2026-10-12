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
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
const pct = (p) => `${Math.round(p * 100)}%`;
const fmt = (v) => (Number.isInteger(v) ? String(v) : Number(v).toFixed(2));
const cssRisk = (p) => `hsl(${Math.round((1 - Math.min(Math.max(p, 0), 1)) * 119)} 72% 42%)`;

const VESSELS = ["LAD", "LCX", "RCA"];
const VESSEL_TEXT = {
  CAD: "Overall coronary artery disease status (dataset definition: at least one major vessel narrowed by 50% or more).",
  LAD: "Left Anterior Descending – supplies blood to the front of the heart.",
  LCX: "Left Circumflex – supplies blood to the side and back of the heart.",
  RCA: "Right Coronary Artery – supplies blood to the right side and bottom of the heart.",
};

// Inputs shown first (matched on normalised column names, so small spelling differences are fine)
const KEY_FIELDS = ["age", "sex", "bp", "pr", "bmi", "fbs", "ldl", "hdl", "tg", "eftte", "typicalchestpain", "dm", "htn",
  "currentsmoker", "fh", "regionrwma", "stelevation", "stdepression", "tinversion", "lvh", "qwave", "dyspnea", "functionclass"];

// Typical adult reference ranges for the measurement table (general guides, not diagnostic thresholds)
const REFS = [
  ["age", "Age", "years", null],
  ["bp", "Systolic BP", "mmHg", [90, 120]],
  ["pr", "Pulse rate", "bpm", [60, 100]],
  ["bmi", "BMI", "kg/m²", [18.5, 24.9]],
  ["fbs", "Fasting glucose", "mg/dL", [70, 100]],
  ["ldl", "LDL cholesterol", "mg/dL", [0, 100]],
  ["hdl", "HDL cholesterol", "mg/dL", [40, 200]],
  ["tg", "Triglycerides", "mg/dL", [0, 150]],
  ["eftte", "Ejection fraction", "%", [55, 70]],
  ["cr", "Creatinine", "mg/dL", [0.6, 1.3]],
  ["hb", "Haemoglobin", "g/dL", [12, 17.5]],
];

const state = {
  schema: [], inputs: {}, values: {}, defaults: {}, samples: [], metrics: null,
  result: null, selected: "CAD", metricsMode: "holdout",
};

/* ============================================================== 3D scene */
const wrap = $("#viewer");
const canvas = $("#scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "low-power" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5)); // keeps it smooth without a dedicated GPU

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
const HOME_POS = new THREE.Vector3(0.4, 0.3, 6.2);
camera.position.copy(HOME_POS);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.minDistance = 2.2;
controls.maxDistance = 10;

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const key = new THREE.DirectionalLight(0xffffff, 1.1);
key.position.set(3, 4, 5);
scene.add(key);
const rim = new THREE.DirectionalLight(0x6aa8ff, 0.55);
rim.position.set(-4, -2, -3);
scene.add(rim);

/* ---- torso / rib cage (decorative context) ---- */
const body = new THREE.Group();
scene.add(body);
const ribMat = new THREE.MeshStandardMaterial({ color: 0xcfd8de, transparent: true, opacity: 0.28, roughness: 0.6, depthWrite: false });
for (let i = 0; i < 7; i++) {
  const r = 1.55 + 0.12 * Math.sin((i / 6) * Math.PI);
  const rib = new THREE.Mesh(new THREE.TorusGeometry(r, 0.028, 8, 64, Math.PI), ribMat);
  rib.rotation.x = Math.PI / 2;
  rib.scale.set(1, 0.6, 1);
  rib.position.set(0, 1.5 - i * 0.5, 0);
  body.add(rib);
}
const sternum = new THREE.Mesh(new THREE.BoxGeometry(0.24, 2.9, 0.07), ribMat);
sternum.position.set(0, 0.05, 0.93);
body.add(sternum);
const shell = new THREE.Mesh(
  new THREE.SphereGeometry(1, 40, 28),
  new THREE.MeshBasicMaterial({ color: 0x4aa3ff, wireframe: true, transparent: true, opacity: 0.05 })
);
shell.scale.set(2.1, 2.7, 1.25);
shell.position.set(0, -0.1, -0.05);
body.add(shell);

/* ---- procedural heart (replace with a real mesh by dropping frontend/models/heart.glb) ---- */
const heart = new THREE.Group();
heart.rotation.y = -0.25;
scene.add(heart);

const heartMat = new THREE.MeshPhysicalMaterial({
  color: 0xd9a0a8, roughness: 0.5, transparent: true, opacity: 0.55, depthWrite: false, clearcoat: 0.3,
});
let heartPick = []; // meshes that select "overall CAD" when clicked
const proceduralParts = [];

function ellipsoid(center, radii, rotZ = 0) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), heartMat);
  m.scale.set(...radii);
  m.position.set(...center);
  m.rotation.z = rotZ;
  heart.add(m);
  proceduralParts.push(m);
  return m;
}
function greatVessel(points, radius, color) {
  const curve = new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)));
  const m = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 40, radius, 16, false),
    new THREE.MeshPhysicalMaterial({ color, roughness: 0.5, transparent: true, opacity: 0.7, depthWrite: false })
  );
  heart.add(m);
  proceduralParts.push(m);
}
ellipsoid([0.3, -0.25, -0.05], [0.62, 0.85, 0.55], 0.4); // left ventricle
ellipsoid([-0.35, -0.2, 0.3], [0.5, 0.7, 0.45], 0.25); // right ventricle
ellipsoid([-0.7, 0.3, 0.0], [0.38, 0.5, 0.38]); // right atrium
ellipsoid([0.35, 0.55, -0.4], [0.4, 0.3, 0.35]); // left atrium
greatVessel([[0, 0.55, 0], [0, 1.1, 0], [0.15, 1.45, 0], [0.5, 1.55, 0], [0.8, 1.3, 0]], 0.2, 0xd9766f); // aorta
greatVessel([[-0.05, 0.5, 0.4], [0.15, 1.1, 0.3], [0.5, 1.25, 0.1]], 0.16, 0x6f8fd9); // pulmonary trunk
heartPick = proceduralParts.slice();

/* ---- coronary arteries: one mesh per target vessel (model output <-> anatomy mapping) ---- */
const NEUTRAL = new THREE.Color(0x8aa0b0);
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
  sprite.userData.draw = (t, color = "#ffffff") => {
    const g = c.getContext("2d");
    g.clearRect(0, 0, 256, 96);
    g.fillStyle = "rgba(10,18,26,0.88)";
    g.fillRect(4, 8, 248, 80);
    g.lineWidth = 5;
    g.strokeStyle = color;
    g.strokeRect(4, 8, 248, 80);
    g.fillStyle = "#fff";
    g.font = "bold 40px Segoe UI, Arial, sans-serif";
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
  const mat = new THREE.MeshStandardMaterial({ color: NEUTRAL, emissive: NEUTRAL, emissiveIntensity: 0.4, roughness: 0.35 });
  const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 64, 0.05, 12, false), mat);
  const halo = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 64, 0.088, 12, false),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28, depthWrite: false })
  );
  halo.visible = false;
  const hit = new THREE.Mesh(new THREE.TubeGeometry(curve, 32, 0.15, 6, false), new THREE.MeshBasicMaterial({ visible: false }));
  hit.userData.target = name;
  const label = makeLabel(name);
  label.position.set(...def.label);
  heart.add(mesh, halo, hit, label);
  art[name] = { curve, mat, halo, label, p: null, color: NEUTRAL.clone() };
  hitMeshes.push(hit);
}

/* ---- optional real anatomical mesh: put a heart.glb in frontend/models/ ---- */
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
        o.material.opacity = 0.55;
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

/* ---- camera focus helper ---- */
const focus = { target: new THREE.Vector3(), pos: null, frames: 0 };
function focusOn(name) {
  if (name === "CAD" || !art[name]) {
    focus.target.set(0, 0, 0);
  } else {
    heart.updateMatrixWorld(true);
    focus.target.copy(heart.localToWorld(art[name].curve.getPoint(0.5).clone())).multiplyScalar(0.6);
  }
  focus.pos = null;
  focus.frames = 50;
}
function resetView() {
  focus.target.set(0, 0, 0);
  focus.pos = HOME_POS.clone();
  focus.frames = 70;
}

/* ---- apply model output to the anatomy ---- */
const HEART_LOW = new THREE.Color(0xd9a0a8);
const HEART_HIGH = new THREE.Color(0xb3262e);
const heartTarget = HEART_LOW.clone();
function riskColor(p) {
  return new THREE.Color().setHSL((1 - Math.min(Math.max(p, 0), 1)) * 0.33, 0.85, 0.5);
}
function applyResultTo3D(res) {
  for (const v of VESSELS) {
    const p = res.vessels[v].probability;
    art[v].p = p;
    art[v].color.copy(riskColor(p));
    art[v].label.userData.draw(`${v} ${pct(p)}`, cssRisk(p));
  }
  heartTarget.copy(HEART_LOW).lerp(HEART_HIGH, res.cad.probability);
}

/* ---- picking (click vessel -> inspect; click heart -> overall CAD) ---- */
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
  const name = t === "CAD" ? "Heart – overall CAD" : `${t} stenosis`;
  tooltip.textContent = p == null ? `${name}: waiting for prediction` : `${name}: ${pct(p)}`;
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
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();
  if (!reduceMotion) heart.scale.setScalar(1 + 0.012 * Math.max(0, Math.sin(t * 5)));
  for (const [name, a] of Object.entries(art)) {
    a.mat.color.lerp(a.color, 0.12);
    a.mat.emissive.copy(a.mat.color);
    const sel = state.selected === name;
    const pulse = !reduceMotion && a.p != null && a.p >= 0.5 ? 0.25 * (0.5 + 0.5 * Math.sin(t * 4)) : 0;
    a.mat.emissiveIntensity = 0.35 + pulse + (sel ? 0.25 : 0);
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

/* ============================================================== dashboard */
function setStatus(t) { $("#status").textContent = t; }

/* ---- form ---- */
function fieldEl(f) {
  const wrapEl = el("label", "field");
  wrapEl.appendChild(el("span", "fname", esc(f.name)));
  wrapEl.title = f.type === "number" ? `${f.name} (dataset range ${fmt(f.min)} – ${fmt(f.max)})` : f.name;
  let input;
  if (f.type === "number") {
    input = el("input");
    input.type = "number";
    input.step = "any";
    input.min = f.min;
    input.max = f.max;
  } else {
    input = el("select");
    input.appendChild(new Option("—", ""));
    f.options.forEach((o) => input.appendChild(new Option(String(o), String(o))));
  }
  input.addEventListener("input", () => {
    const v = input.value;
    state.values[f.name] = v === "" ? "" : f.type === "category" ? v : Number(v);
    schedulePredict();
  });
  state.inputs[f.name] = input;
  wrapEl.appendChild(input);
  return wrapEl;
}
function addSection(parent, title, fields, open) {
  const d = el("details");
  d.open = open;
  d.appendChild(el("summary", null, esc(title)));
  const g = el("div", "grid2");
  fields.forEach((f) => g.appendChild(fieldEl(f)));
  d.appendChild(g);
  parent.appendChild(d);
}
function buildForm() {
  const form = $("#form");
  form.innerHTML = "";
  const byNorm = new Map(state.schema.map((f) => [norm(f.name), f]));
  const keyFields = KEY_FIELDS.map((k) => byNorm.get(k)).filter(Boolean);
  const keySet = new Set(keyFields.map((f) => f.name));
  const rest = state.schema.filter((f) => !keySet.has(f.name));
  addSection(form, "Key inputs", keyFields, true);
  addSection(form, `All other features (${rest.length})`, rest, false);
}
function setValues(vals) {
  for (const f of state.schema) {
    const v = vals[f.name];
    state.values[f.name] = v == null ? "" : v;
    state.inputs[f.name].value = v == null ? "" : String(v);
  }
}

/* ---- prediction ---- */
let timer = null, reqId = 0;
function schedulePredict() {
  clearTimeout(timer);
  timer = setTimeout(predict, 350);
}
async function predict() {
  const id = ++reqId;
  setStatus("Computing…");
  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: state.values }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const res = await r.json();
    if (id !== reqId) return; // a newer request superseded this one
    state.result = res;
    applyResultTo3D(res);
    renderResults();
    setStatus("Up to date");
  } catch (err) {
    console.error(err);
    setStatus("Error – see console");
  }
}

/* ---- rendering the dashboard ---- */
function prettyFeature(name, value) {
  const base = state.schema.find((f) => f.type === "category" && name.startsWith(f.name + "_"));
  if (base) return { label: `${base.name} = ${name.slice(base.name.length + 1)}`, text: value >= 0.5 ? "yes" : "no" };
  return { label: name, text: fmt(value) };
}
const bandText = (b) => ({ low: "Low risk", moderate: "Moderate risk", high: "High risk" }[b]);

function renderResults() {
  const res = state.result;
  if (!res) return;

  // overall
  const o = $("#overall");
  const p = res.cad.probability;
  o.innerHTML = `
    <div class="row">
      <div><div class="lbl">Predicted overall CAD status</div><div class="big">${pct(p)}</div></div>
      <div style="text-align:right">
        <span class="chip ${res.cad.band}">${bandText(res.cad.band)}</span>
        <div class="lbl">${res.cad.label ? "CAD likely" : "CAD unlikely"} · ${esc(res.models.CAD.replace("_", " "))}</div>
      </div>
    </div>
    <div class="meter"><i style="width:${p * 100}%;background:${cssRisk(p)}"></i></div>`;

  // vessel cards
  const vc = $("#vessel-cards");
  vc.innerHTML = "";
  for (const v of VESSELS) {
    const r = res.vessels[v];
    const b = el("button", "vcard");
    b.dataset.target = v;
    b.innerHTML = `<div class="vn">${v}</div><div class="vp">${pct(r.probability)}</div>
      <div class="meter"><i style="width:${r.probability * 100}%;background:${cssRisk(r.probability)}"></i></div>
      <div class="vd">${r.label ? "Stenosis likely" : "Stenosis unlikely"}</div>`;
    b.addEventListener("click", () => select(v, { fly: true }));
    vc.appendChild(b);
  }
  renderSelection();
}

function renderSelection() {
  const res = state.result;
  const sel = state.selected;
  $("#overall").classList.toggle("sel", sel === "CAD");
  document.querySelectorAll(".vcard").forEach((c) => c.classList.toggle("sel", c.dataset.target === sel));

  // HUD
  const hud = $("#hud-selected");
  if (res) {
    const r = sel === "CAD" ? res.cad : res.vessels[sel];
    hud.innerHTML = `<strong>${sel === "CAD" ? "Overall CAD" : sel}</strong> – ${pct(r.probability)} (${bandText(r.band).toLowerCase()})<br><span style="color:#9fb3c0">${esc(VESSEL_TEXT[sel])}</span>`;
  }
  if (!res) return;

  // SHAP drivers
  const ex = res.explanations[sel];
  $("#why-title").textContent = sel === "CAD" ? "Why: overall CAD prediction" : `Why: ${sel} stenosis prediction`;
  $("#why-sub").textContent = "Each bar is one input's SHAP contribution to this prediction. Red pushes the risk up, blue pushes it down. Percentages are each input's share of the total effect.";
  const top = ex.contributions.slice(0, 10);
  const max = Math.max(...top.map((c) => Math.abs(c.shap)), 1e-9);
  $("#drivers").innerHTML =
    top.map((c) => {
      const pf = prettyFeature(c.feature, c.value);
      const w = (Math.abs(c.shap) / max) * 50;
      const up = c.shap > 0;
      return `<div class="drv" title="${esc(c.feature)}: SHAP ${c.shap}">
        <div class="drv-name">${esc(pf.label)} <small>${esc(pf.text)}</small></div>
        <div class="drv-bar"><div class="mid"></div><div class="fill ${up ? "up" : "down"}" style="width:${w}%"></div></div>
        <div class="drv-share">${up ? "+" : "−"}${c.share.toFixed(1)}%</div></div>`;
    }).join("") +
    `<div class="axis-note"><span>◀ lowers risk</span><span>raises risk ▶</span></div>`;

  // physiological measurements with contribution share
  const byNorm = new Map(state.schema.map((f) => [norm(f.name), f]));
  let rows = "";
  for (const [k, label, unit, range] of REFS) {
    const f = byNorm.get(k);
    if (!f) continue;
    const raw = state.values[f.name];
    const has = raw !== "" && raw != null && !Number.isNaN(Number(raw));
    const c = ex.contributions.find((x) => x.feature === f.name);
    let cls = "na";
    if (has && range) cls = Number(raw) >= range[0] && Number(raw) <= range[1] ? "ok" : "out";
    const share = c ? `${c.shap > 0 ? "+" : "−"}${c.share.toFixed(1)}%` : "–";
    rows += `<tr><td><span class="dot ${cls}"></span>${esc(label)}</td>
      <td class="r">${has ? fmt(Number(raw)) : "–"} <small>${esc(unit)}</small></td>
      <td class="r">${range ? `${range[0]}–${range[1]}` : "–"}</td><td class="r">${share}</td></tr>`;
  }
  $("#vitals").innerHTML = `<tr><th>Measurement</th><th class="r">Value</th><th class="r">Typical</th><th class="r">Share</th></tr>${rows}`;
}

function renderMetrics() {
  const m = state.metrics;
  if (!m) return;
  const mode = state.metricsMode;
  const f3 = (v) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(3));
  let rows = "";
  for (const [t, info] of Object.entries(m.targets)) {
    const x = info[mode];
    rows += `<tr><td><strong>${t}</strong></td><td>${esc(info.model.replace("_", " "))}</td><td class="r">${f3(x.accuracy)}</td>
      <td class="r">${f3(x.precision)}</td><td class="r">${f3(x.recall)}</td><td class="r">${f3(x.f1)}</td><td class="r">${f3(x.roc_auc)}</td></tr>`;
  }
  $("#metrics").innerHTML =
    `<tr><th>Target</th><th>Model</th><th class="r">Acc</th><th class="r">Prec</th><th class="r">Recall</th><th class="r">F1</th><th class="r">AUC</th></tr>${rows}`;
  document.querySelectorAll("#metric-mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
}

/* ---- selection (shared by 3D clicks and dashboard clicks) ---- */
function select(target, { fly = false } = {}) {
  state.selected = target;
  renderSelection();
  if (fly) focusOn(target);
}

/* ---- wiring ---- */
function showTruth(truth) {
  const t = $("#truth");
  if (!truth) { t.hidden = true; return; }
  const yn = (v) => (v ? "stenosed" : "normal");
  t.textContent = `Dataset (angiography) labels – CAD: ${truth.CAD ? "yes" : "no"} · LAD ${yn(truth.LAD)} · LCX ${yn(truth.LCX)} · RCA ${yn(truth.RCA)}`;
  t.hidden = false;
}
$("#btn-sample").addEventListener("click", () => {
  if (!state.samples.length) return;
  const s = state.samples[Math.floor(Math.random() * state.samples.length)];
  setValues(s.features);
  showTruth(s.truth);
  predict();
});
$("#btn-reset").addEventListener("click", () => { setValues(state.defaults); showTruth(null); predict(); });
$("#btn-view").addEventListener("click", resetView);
$("#tg-torso").addEventListener("change", (e) => { body.visible = e.target.checked; });
$("#tg-labels").addEventListener("change", (e) => { Object.values(art).forEach((a) => (a.label.visible = e.target.checked)); });
$("#metric-mode").addEventListener("click", (e) => {
  if (e.target.dataset.mode) { state.metricsMode = e.target.dataset.mode; renderMetrics(); }
});
const overallEl = $("#overall");
overallEl.tabIndex = 0;
overallEl.setAttribute("role", "button");
overallEl.addEventListener("click", () => select("CAD", { fly: true }));
overallEl.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select("CAD", { fly: true }); } });

async function init() {
  try {
    const [schema, metrics, samples] = await Promise.all(
      ["/api/schema", "/api/metrics", "/api/samples"].map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u); return r.json(); }))
    );
    state.schema = schema.features;
    state.metrics = metrics;
    state.samples = samples;
    state.defaults = Object.fromEntries(state.schema.map((f) => [f.name, f.default]));
    buildForm();
    setValues(state.defaults);
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
