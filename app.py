"""
Visualizador de la ley de Schmid y sistemas de deslizamiento (FCC, BCC, HCP)

Notación:
    τ_R = σ cos φ cos λ          (ley de Schmid)
    m   = cos φ cos λ            (factor de Schmid)
    φ : ángulo entre la normal al plano de deslizamiento y el esfuerzo aplicado
    λ : ángulo entre la dirección de deslizamiento y el esfuerzo aplicado
    τ_c (τ_CRSS) : esfuerzo cortante crítico resuelto
Solo se consideran los sistemas PREFERENCIALES de deslizamiento:
    FCC {111}<110>  (4 planos x 3 direcciones = 12 sistemas)
    BCC {110}<111>  (6 planos x 2 direcciones = 12 sistemas)
    HCP (0001)<11 2̄ 0>  (1 plano x 3 direcciones = 3 sistemas)

Ejecutar:  streamlit run app.py
"""

import itertools
import json
from math import gcd

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Ley de Schmid",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Utilidades de interfaz (compatibles con distintas versiones de Streamlit)
# ----------------------------------------------------------------------------


def show_plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# Notación: barra sobre los índices negativos, (p. ej. 1̅ 1 1)
# ----------------------------------------------------------------------------
OVER = "\u0305"  # combining overline


def bar(i):
    i = int(i)
    return f"{abs(i)}{OVER}" if i < 0 else str(i)


def fmt(v, br="()", sep=" "):
    return br[0] + sep.join(bar(i) for i in v) + br[1]


FAMILY_LABEL = {
    "FCC": "{111}<110>",
    "BCC": "{110}<111>",
    "HCP": "(0001)<11" + bar(-2) + "0>",
}

# ----------------------------------------------------------------------------
# Sistemas de deslizamiento preferenciales
# ----------------------------------------------------------------------------
# Planos de la familia {111} 
FCC_PLANES = [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1)]
# Planos de la familia {110} (se omiten los paralelos con todos los signos invertidos)
BCC_PLANES = [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
# Direcciones <11-20> de HCP (permutaciones cíclicas de [1 1 -2 0]), notación [UVTW]
HCP_DIRS = [(1, 1, -2, 0), (-2, 1, 1, 0), (1, -2, 1, 0)]


def _first_positive(v):
    for x in v:
        if x != 0:
            return x > 0
    return True


def _cubic_dirs(family, plane):
    """Direcciones de la familia contenidas en el plano (n·d = 0).
    Signos opuestos son equivalentes: se toma la que tiene el primer índice no nulo positivo."""
    if family == "110":
        cand = [v for v in itertools.product([-1, 0, 1], repeat=3) if sorted(map(abs, v)) == [0, 1, 1]]
    else:  # "111"
        cand = list(itertools.product([-1, 1], repeat=3))
    return [v for v in cand if _first_positive(v) and int(np.dot(v, plane)) == 0]


def hex4_to_cart(v, ca):
    """Dirección [UVTW] -> vector cartesiano (a = 1, eje c vertical)."""
    U, V, T, W = v
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([-0.5, np.sqrt(3) / 2, 0.0])
    a3 = np.array([-0.5, -np.sqrt(3) / 2, 0.0])
    c = np.array([0.0, 0.0, ca])
    return U * a1 + V * a2 + T * a3 + W * c


def hex3_to_cart(u, v, w, ca):
    """Dirección [uvw] (3 índices) -> vector cartesiano."""
    return u * np.array([1.0, 0.0, 0.0]) + v * np.array([-0.5, np.sqrt(3) / 2, 0.0]) + w * np.array([0.0, 0.0, ca])


def hex3_to_4(u, v, w):
    """Conversión [uvw] -> [UVTW] con las fórmulas:
    U=(2u-v)/3, V=(2v-u)/3, T=-(U+V), W=w  (se escala a enteros)."""
    U, V, T, W = 2 * u - v, 2 * v - u, -(u + v), 3 * w
    g = gcd(gcd(abs(U), abs(V)), gcd(abs(T), abs(W))) or 1
    return (U // g, V // g, T // g, W // g)


def build_systems(struct, ca=1.633):
    out = []
    if struct in ("FCC", "BCC"):
        planes = FCC_PLANES if struct == "FCC" else BCC_PLANES
        family = "110" if struct == "FCC" else "111"
        for p in planes:
            n = np.array(p, float)
            n /= np.linalg.norm(n)
            for d in _cubic_dirs(family, p):
                dv = np.array(d, float)
                dv /= np.linalg.norm(dv)
                out.append(dict(plane=p, dir=d, n=n, d=dv, planeTxt=fmt(p), dirTxt=fmt(d, "[]")))
    else:  # HCP: solo plano basal (0001) con las 3 direcciones <11-20>
        p = (0, 0, 0, 1)
        n = np.array([0.0, 0.0, 1.0])
        for d in HCP_DIRS:
            dv = hex4_to_cart(d, ca)
            dv /= np.linalg.norm(dv)
            out.append(dict(plane=p, dir=d, n=n, d=dv, planeTxt=fmt(p, "()", ""), dirTxt=fmt(d, "[]", "")))
    return out


# ----------------------------------------------------------------------------
# Geometría 3D de las celdas unitarias
# ----------------------------------------------------------------------------
CUBE_V = list(itertools.product([0, 1], repeat=3))
CUBE_EDGES = [(a, b) for a in CUBE_V for b in CUBE_V if a < b and sum(x != y for x, y in zip(a, b)) == 1]

C_PLANE = "rgb(226,110,125)"
C_NORMAL = "rgb(20,150,70)"
C_SLIP = "rgb(240,140,0)"
C_FORCE = "rgb(200,20,20)"
C_ON = "rgb(215,30,30)"
C_OFF = "rgb(60,110,190)"


def _plane_d_fcc(p):
    vals = [int(np.dot(p, v)) for v in CUBE_V]
    lo, hi = min(vals), max(vals)
    cands = sorted({x for x in vals if lo < x < hi and vals.count(x) == 3})
    return cands[0]


def _plane_polygon_cube(p, d):
    p = np.array(p, float)
    pts = []
    for a, b in CUBE_EDGES:
        a = np.array(a, float)
        b = np.array(b, float)
        fa, fb = p @ a - d, p @ b - d
        if abs(fa) < 1e-9:
            pts.append(a)
        if abs(fb) < 1e-9:
            pts.append(b)
        if fa * fb < -1e-12:
            pts.append(a + (b - a) * fa / (fa - fb))
    uniq = []
    for q in pts:
        if not any(np.allclose(q, u) for u in uniq):
            uniq.append(q)
    P = np.array(uniq)
    c = P.mean(axis=0)
    n = p / np.linalg.norm(p)
    ref = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0])
    ref /= np.linalg.norm(ref)
    ref2 = np.cross(n, ref)
    ang = [np.arctan2((q - c) @ ref2, (q - c) @ ref) for q in P]
    return P[np.argsort(ang)]


def _add_arrow(fig, p0, p1, color, name, label=None, width=7, cone=0.22, dash=None, show_legend=True):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    fig.add_trace(
        go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines", line=dict(color=color, width=width, dash=dash or "solid"),
            name=name, showlegend=show_legend, hoverinfo="name",
        )
    )
    dvec = p1 - p0
    nrm = np.linalg.norm(dvec)
    if nrm > 1e-9:
        u = dvec / nrm
        fig.add_trace(
            go.Cone(
                x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[u[0]], v=[u[1]], w=[u[2]],
                anchor="tip", sizemode="absolute", sizeref=cone, showscale=False,
                colorscale=[[0, color], [1, color]], hoverinfo="skip", showlegend=False,
            )
        )
    if label:
        fig.add_trace(
            go.Scatter3d(
                x=[p1[0]], y=[p1[1]], z=[p1[2]], mode="text", text=[label],
                textfont=dict(color=color, size=14), textposition="top center",
                showlegend=False, hoverinfo="skip",
            )
        )


def _edges_trace(segments):
    xs, ys, zs = [], [], []
    for a, b in segments:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", line=dict(color="rgb(90,90,90)", width=3),
        hoverinfo="skip", showlegend=False,
    )


def _atoms_traces(atoms, on_plane_mask):
    atoms = np.array(atoms, float)
    mask = np.array(on_plane_mask, bool)
    traces = []
    for sel, color, name in ((mask, C_ON, "Átomos en el plano de deslizamiento"), (~mask, C_OFF, "Otros átomos")):
        if sel.any():
            traces.append(
                go.Scatter3d(
                    x=atoms[sel, 0], y=atoms[sel, 1], z=atoms[sel, 2], mode="markers",
                    marker=dict(size=9, color=color, line=dict(color="black", width=1)),
                    name=name, hoverinfo="skip",
                )
            )
    return traces


def _scene(fig, height=520):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", camera=dict(eye=dict(x=1.3, y=-1.4, z=0.85)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.02, x=0, font=dict(size=11)),
    )
    return fig


def cell_figure(struct, system, t=None, ca=1.633, height=520):
    """Celda unitaria con el plano de deslizamiento, la normal, la dirección de
    deslizamiento y (opcional) el eje de tracción."""
    fig = go.Figure()
    n, dv = system["n"], system["d"]

    if struct in ("FCC", "BCC"):
        p = system["plane"]
        fig.add_trace(_edges_trace([(np.array(a, float), np.array(b, float)) for a, b in CUBE_EDGES]))
        atoms = [np.array(v, float) for v in CUBE_V]
        if struct == "FCC":
            for ax in range(3):
                for val in (0, 1):
                    q = [0.5, 0.5, 0.5]
                    q[ax] = val
                    atoms.append(np.array(q))
            d_plane = _plane_d_fcc(p)
        else:
            atoms.append(np.array([0.5, 0.5, 0.5]))
            d_plane = float(np.dot(p, [0.5, 0.5, 0.5]))
        poly = _plane_polygon_cube(p, d_plane)
        centroid = poly.mean(axis=0)
        center = np.array([0.5, 0.5, 0.5])
        on = [abs(np.dot(p, a) - d_plane) < 1e-9 for a in atoms]
        # ejes cristalográficos
        for vec, lab in (((1.35, 0, 0), "X"), ((0, 1.35, 0), "Y"), ((0, 0, 1.35), "Z")):
            _add_arrow(fig, (0, 0, 0), vec, "rgb(40,160,60)", lab, label=lab, width=3, cone=0.12, show_legend=False)
        L_dir, L_norm, L_t = 0.42, 0.5, 0.95
    else:  # HCP
        ang = np.radians(np.arange(0, 360, 60))
        hexb = [np.array([np.cos(a), np.sin(a), 0.0]) for a in ang]
        hext = [np.array([np.cos(a), np.sin(a), ca]) for a in ang]
        segs = [(hexb[i], hexb[(i + 1) % 6]) for i in range(6)]
        segs += [(hext[i], hext[(i + 1) % 6]) for i in range(6)]
        segs += [(hexb[i], hext[i]) for i in range(6)]
        fig.add_trace(_edges_trace(segs))
        atoms = hexb + hext + [np.array([0, 0, 0.0]), np.array([0, 0, ca])]
        for k in range(3):
            a = np.radians(30 + 120 * k)
            atoms.append(np.array([np.cos(a) / np.sqrt(3), np.sin(a) / np.sqrt(3), ca / 2]))
        poly = np.array(hexb)
        centroid = np.array([0.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, ca / 2])
        on = [abs(a[2]) < 1e-9 for a in atoms]
        _add_arrow(fig, (0, 0, 0), (1.35, 0, 0), "rgb(40,160,60)", "a₁", label="a₁", width=3, cone=0.12, show_legend=False)
        _add_arrow(fig, (0, 0, 0), (-0.68, 1.17, 0), "rgb(40,160,60)", "a₂", label="a₂", width=3, cone=0.12, show_legend=False)
        L_dir, L_norm, L_t = 0.75, 0.7, 1.0

    # plano de deslizamiento
    k = len(poly)
    fig.add_trace(
        go.Mesh3d(
            x=poly[:, 0], y=poly[:, 1], z=poly[:, 2],
            i=[0] * (k - 2), j=list(range(1, k - 1)), k=list(range(2, k)),
            color=C_PLANE, opacity=0.5, flatshading=True,
            name=f"Plano de deslizamiento {system['planeTxt']}", showlegend=True, hoverinfo="name",
        )
    )
    closed = np.vstack([poly, poly[:1]])
    fig.add_trace(
        go.Scatter3d(
            x=closed[:, 0], y=closed[:, 1], z=closed[:, 2], mode="lines",
            line=dict(color=C_PLANE, width=4), hoverinfo="skip", showlegend=False,
        )
    )
    for tr in _atoms_traces(atoms, on):
        fig.add_trace(tr)

    _add_arrow(fig, centroid, centroid + L_norm * n, C_NORMAL, "Normal al plano de deslizamiento", label="n")
    _add_arrow(fig, centroid - L_dir * dv, centroid + L_dir * dv, C_SLIP,
               f"Dirección de deslizamiento {system['dirTxt']}", label="d", cone=0.2)
    if t is not None:
        _add_arrow(fig, center - L_t * t, center + L_t * t, C_FORCE, "Fuerza aplicada/Esfuerzo σ", label="σ", cone=0.25)
    return _scene(fig, height)


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
HCP_CA = {
    "Ideal (c/a = 1.633)": 1.633,
    "Mg (c/a ≈ 1.624)": 1.624,
    "Ti (c/a ≈ 1.587)": 1.587,
    "Zn (c/a ≈ 1.856)": 1.856,
    "Cd (c/a ≈ 1.886)": 1.886,
}

PRESETS = {
    "FCC": {"[100]": (1, 0, 0), "[110]": (1, 1, 0), "[111]": (1, 1, 1),
            "[123]": (1, 2, 3), "[2 1̅ 3̅]": (2, -1, -3)},
    "BCC": {"[123]": (1, 2, 3), "[121]": (1, 2, 1), "[101]": (1, 0, 1),
            "[100]": (1, 0, 0), "[111]": (1, 1, 1), "[2 1̅ 3̅]": (2, -1, -3)},
    "HCP": {"[001]  (eje c)": (0, 0, 1), "[100]": (1, 0, 0), "[101]": (1, 0, 1),
            "[102]": (1, 0, 2), "[111]": (1, 1, 1)},
}

# ----------------------------------------------------------------------------
# Componentes interactivos en JavaScript: al arrastrar un slider el dibujo y los
# resultados se actualizan al instante (sin esperar a soltar la barra).
# ----------------------------------------------------------------------------
PLOTLY_TAGS = (
    '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    '<script>window.Plotly || document.write(\'<script src="https://cdnjs.cloudflare.com/ajax/libs/'
    'plotly.js/2.35.2/plotly.min.js"><\\/script>\')</script>'
)

BASE_CSS = r"""
:root{--fg:#262730;--muted:#6b7280;--card:#f0f2f6;--line:#d1d5db;--ok:#1b7f3b;--okbg:#e3f4e8;
--info:#1d4f91;--infobg:#e4eefb;--warn:#8a5a00;--warnbg:#fff4d6;--err:#9b1c1c;--errbg:#fde8e8;}
.dark{--fg:#fafafa;--muted:#a3a8b4;--card:#262b36;--line:#444b5a;--ok:#7be0a0;--okbg:#17352a;
--info:#9ec5ff;--infobg:#1b2a44;--warn:#ffd77a;--warnbg:#3b3014;--err:#ffa8a8;--errbg:#431d1d;}
*{box-sizing:border-box}
body{margin:0;padding:2px 4px 8px;font-family:"Source Sans Pro",system-ui,-apple-system,Segoe UI,sans-serif;
color:var(--fg);font-size:15px;background:transparent}
h3{margin:4px 0 8px;font-size:1.25rem}
.grid{display:grid;grid-template-columns:minmax(300px,1fr) minmax(320px,1.25fr);gap:18px;align-items:start}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.eq{font-size:1.15rem;margin:2px 0 10px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:6px}
label.f{display:block;font-size:.86rem;color:var(--muted)}
input[type=number],select{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;
background:transparent;color:var(--fg);font:inherit;margin-top:3px}
select option{color:#222}
.sl{margin:10px 0 4px}
.sl .lab{font-size:.9rem;margin-bottom:2px}
.sl b{color:#d62728}
input[type=range]{width:100%;accent-color:#e5484d;cursor:pointer}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}
.card{background:var(--card);border-radius:8px;padding:8px 10px}
.card .k{font-size:.78rem;color:var(--muted)}
.card .v{font-size:1.3rem;font-weight:600;line-height:1.25;word-break:break-word}
.msg{padding:9px 12px;border-radius:8px;margin:8px 0;font-size:.92rem}
.msg.ok{background:var(--okbg);color:var(--ok)}.msg.info{background:var(--infobg);color:var(--info)}
.msg.warn{background:var(--warnbg);color:var(--warn)}.msg.err{background:var(--errbg);color:var(--err)}
.cap{font-size:.82rem;color:var(--muted);margin:4px 0}
.modes label{display:block;margin:3px 0;cursor:pointer}
.pill{display:inline-flex;gap:14px;margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600;position:sticky;top:0}
.tw{overflow-x:auto;margin-top:6px}
.hide{display:none}
"""

COMMON_JS = r"""
const D2R = Math.PI / 180;
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const nrm = a => Math.hypot(a[0], a[1], a[2]);
const unit = a => { const n = nrm(a) || 1; return [a[0]/n, a[1]/n, a[2]/n]; };
const addv = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const mulv = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
const OVER = '\u0305';
const barTxt = i => (i < 0 ? Math.abs(i) + OVER : String(i));
const gcd = (a, b) => (b === 0 ? Math.abs(a) : gcd(b, a % b));

function evalSystems(systems, t, sigma, tauc) {
  return systems.map((s, k) => {
    const cphi = Math.min(Math.abs(dot(t, s.n)), 1);
    const clam = Math.min(Math.abs(dot(t, s.d)), 1);
    const m = cphi * clam;
    return { i: k, phi: Math.acos(cphi) / D2R, lam: Math.acos(clam) / D2R, m: m,
             tauR: sigma * m, sigc: m > 1e-9 ? tauc / m : Infinity };
  });
}

function arrowTraces(p0, p1, color, name, label, o) {
  o = o || {};
  const w = o.width || 7, cone = o.cone || 0.22, leg = o.legend !== false;
  const tr = [{ type: 'scatter3d', mode: 'lines', x: [p0[0], p1[0]], y: [p0[1], p1[1]], z: [p0[2], p1[2]],
                line: { color: color, width: w }, name: name, showlegend: leg, hoverinfo: 'name' }];
  const d = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
  const n = nrm(d);
  if (n > 1e-9) {
    const u = mulv(d, 1 / n);
    tr.push({ type: 'cone', x: [p1[0]], y: [p1[1]], z: [p1[2]], u: [u[0]], v: [u[1]], w: [u[2]],
              anchor: 'tip', sizemode: 'absolute', sizeref: cone, showscale: false,
              colorscale: [[0, color], [1, color]], hoverinfo: 'skip', showlegend: false });
  }
  if (label) tr.push({ type: 'scatter3d', mode: 'text', x: [p1[0]], y: [p1[1]], z: [p1[2]], text: [label],
                       textfont: { color: color, size: 14 }, textposition: 'top center',
                       showlegend: false, hoverinfo: 'skip' });
  return tr;
}

function setupTheme(P) {
  const dark = (P.dark === null || P.dark === undefined)
    ? window.matchMedia('(prefers-color-scheme: dark)').matches : P.dark;
  if (dark) document.documentElement.classList.add('dark');
  return { dark: dark, fg: dark ? '#fafafa' : '#262730' };
}

function themedLayout(base, th, extra) {
  const L = JSON.parse(JSON.stringify(base));
  L.paper_bgcolor = 'rgba(0,0,0,0)'; L.plot_bgcolor = 'rgba(0,0,0,0)';
  L.font = Object.assign({}, L.font || {}, { color: th.fg });
  L.uirevision = 'keep';
  return Object.assign(L, extra || {});
}

function throttled(fn) {
  let pending = false;
  return () => { if (pending) return; pending = true;
    requestAnimationFrame(() => { pending = false; fn(); }); };
}
"""

SCHMID_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">__PLOTLY__<style>__CSS__</style></head><body>
<div class="grid">
  <div>
    <div class="eq">τ<sub>R</sub> = σ cos φ cos λ &nbsp;&nbsp;&nbsp; m = cos φ cos λ</div>
    <div class="row2">
      <label class="f">Esfuerzo aplicado σ (MPa)<input type="number" id="sigma" min="0" step="1"></label>
      <label class="f">Esfuerzo cortante crítico τ<sub>c</sub> (MPa)<input type="number" id="tauc" min="0.01" step="1"></label>
    </div>
    <div class="sl"><div class="lab">φ : ángulo entre la normal al plano de deslizamiento y el esfuerzo aplicado: <b id="phiv"></b></div>
      <input type="range" id="phi" min="0" max="90" step="0.1"></div>
    <div class="sl"><div class="lab">λ : ángulo entre la dirección de deslizamiento y el esfuerzo aplicado: <b id="lamv"></b></div>
      <input type="range" id="lam" min="0" max="90" step="0.1"></div>
    <div class="metrics">
      <div class="card"><div class="k">Factor de Schmid m</div><div class="v" id="m"></div></div>
      <div class="card"><div class="k">Esfuerzo cortante resuelto τ<sub>R</sub></div><div class="v" id="tauR"></div></div>
      <div class="card"><div class="k">σ necesario = τ<sub>c</sub> / m</div><div class="v" id="sigy"></div></div>
      <div class="card"><div class="k">φ + λ</div><div class="v" id="sum"></div></div>
    </div>
    <div id="status" class="msg"></div>
    <div id="warn" class="msg warn hide">Si la normal al plano y la dirección de deslizamiento son perpendiculares,
      siempre φ + λ ≥ 90° (= 90° cuando el eje de esfuerzo, la normal y la dirección son coplanares).
      El cálculo usa los valores ingresados; el dibujo muestra la geometría compatible más cercana.</div>
  </div>
  <div id="diagram" style="height:440px"></div>
</div>
<div id="map" style="height:440px;margin-top:8px"></div>
<div class="cap">La zona vacía (φ + λ &lt; 90°) es geométricamente imposible cuando la normal al plano y la dirección de deslizamiento son perpendiculares.</div>
<script>
const P = __DATA__;
__COMMON__
const th = setupTheme(P);
const $ = id => document.getElementById(id);
const S = { phi: 45, lam: 45, sigma: P.sigma, tauc: P.tauc };
$('sigma').value = S.sigma; $('tauc').value = S.tauc; $('phi').value = S.phi; $('lam').value = S.lam;
const cfg = { displayModeBar: false, responsive: true };

function arc(u, v, r, text) {
  const ang = Math.acos(Math.max(-1, Math.min(1, dot(u, v))));
  if (ang < 1e-6) return [];
  const pts = [];
  for (let k = 0; k <= 29; k++) {
    const x = k / 29, a = Math.sin((1 - x) * ang) / Math.sin(ang), b = Math.sin(x * ang) / Math.sin(ang);
    pts.push([r * (a*u[0] + b*v[0]), r * (a*u[1] + b*v[1]), r * (a*u[2] + b*v[2])]);
  }
  const mid = mulv(pts[15], 1.12);
  return [
    { type: 'scatter3d', mode: 'lines', x: pts.map(p => p[0]), y: pts.map(p => p[1]), z: pts.map(p => p[2]),
      line: { color: th.dark ? '#cfd3da' : '#464646', width: 3 }, showlegend: false, hoverinfo: 'skip' },
    { type: 'scatter3d', mode: 'text', x: [mid[0]], y: [mid[1]], z: [mid[2]], text: [text],
      textfont: { size: 17, color: th.fg }, showlegend: false, hoverinfo: 'skip' }];
}

const diskI = [], diskJ = [], diskK = [], diskX = [0], diskY = [0];
for (let k = 0; k < 60; k++) { const a = 2 * Math.PI * k / 59; diskX.push(1.15 * Math.cos(a)); diskY.push(1.15 * Math.sin(a)); }
for (let k = 1; k < 60; k++) { diskI.push(0); diskJ.push(k); diskK.push(k + 1); }

function diagramTraces() {
  const c1 = Math.cos(S.phi * D2R), c2 = Math.cos(S.lam * D2R);
  const t = unit([c2, Math.sqrt(Math.max(1 - c1*c1 - c2*c2, 0)), c1]);
  const n = [0, 0, 1], d = [1, 0, 0];
  let tr = [{ type: 'mesh3d', x: diskX, y: diskY, z: diskX.map(() => 0), i: diskI, j: diskJ, k: diskK,
              color: P.c_plane, opacity: 0.45, name: 'Plano de deslizamiento', showlegend: true, hoverinfo: 'name' }];
  tr = tr.concat(arrowTraces([0,0,0], mulv(n, 1.3), P.c_normal, 'Normal al plano de deslizamiento', 'n'));
  tr = tr.concat(arrowTraces([0,0,0], mulv(d, 1.3), P.c_slip, 'Dirección de deslizamiento', 'd'));
  tr = tr.concat(arrowTraces(mulv(t, -1.3), mulv(t, 1.3), P.c_force, 'Fuerza aplicada/Esfuerzo σ', 'σ', { cone: 0.3 }));
  tr = tr.concat(arc(n, t, 0.6, 'φ'), arc(d, t, 0.8, 'λ'));
  return tr;
}

const layD = themedLayout(P.layout, th, { height: 440 });
layD.scene.camera = { eye: { x: 1.05, y: -1.15, z: 0.6 } };

const g = Array.from({ length: 91 }, (_, k) => k);
const Z = g.map(l => g.map(p => (p + l >= 90 - 1e-9) ? Math.cos(p * D2R) * Math.cos(l * D2R) : null));
const layM = themedLayout({}, th, {
  height: 440, margin: { l: 55, r: 10, t: 45, b: 50 },
  title: { text: 'Factor de Schmid m en función de φ y λ (máximo: m = 0.5 con φ = λ = 45°)', font: { size: 14 } },
  xaxis: { title: 'φ (°)' }, yaxis: { title: 'λ (°)' }, legend: { orientation: 'h', y: -0.18 } });
Plotly.newPlot('map', [
  { type: 'contour', x: g, y: g, z: Z, colorscale: 'Viridis', zmin: 0, zmax: 0.5,
    contours: { start: 0, end: 0.5, size: 0.05, showlabels: true }, colorbar: { title: 'm' },
    hovertemplate: 'φ=%{x}°<br>λ=%{y}°<br>m=%{z:.3f}<extra></extra>', name: 'm' },
  { type: 'scatter', mode: 'lines', x: [0, 90], y: [90, 0], line: { color: th.dark ? '#d0d0d0' : '#6b6b6b', dash: 'dash' },
    name: 'φ + λ = 90° (coplanar)', hoverinfo: 'skip' },
  { type: 'scatter', mode: 'markers', x: [S.phi], y: [S.lam],
    marker: { size: 14, color: 'red', line: { color: 'white', width: 2 } }, name: 'Actual' }
], layM, cfg);

function update() {
  const c1 = Math.cos(S.phi * D2R), c2 = Math.cos(S.lam * D2R);
  const m = c1 * c2, tauR = S.sigma * m, sigy = m > 1e-9 ? S.tauc / m : Infinity;
  $('phiv').textContent = S.phi.toFixed(1) + '°'; $('lamv').textContent = S.lam.toFixed(1) + '°';
  $('m').textContent = m.toFixed(4); $('tauR').textContent = tauR.toFixed(3) + ' MPa';
  $('sigy').textContent = isFinite(sigy) ? sigy.toFixed(2) + ' MPa' : '∞';
  $('sum').textContent = (S.phi + S.lam).toFixed(1) + '°';
  const st = $('status');
  if (!isFinite(sigy)) { st.className = 'msg err'; st.textContent = 'm = 0: el esfuerzo no produce cortante en la dirección de deslizamiento.'; }
  else if (tauR >= S.tauc) { st.className = 'msg ok';
    st.innerHTML = 'τ<sub>R</sub> = ' + tauR.toFixed(2) + ' MPa ≥ τ<sub>c</sub> = ' + S.tauc.toFixed(2) + ' MPa → <b>el cristal se deforma plásticamente</b>.'; }
  else { st.className = 'msg info';
    st.innerHTML = 'τ<sub>R</sub> = ' + tauR.toFixed(2) + ' MPa &lt; τ<sub>c</sub> = ' + S.tauc.toFixed(2) +
      ' MPa → el cristal <b>no</b> se deforma (aún). Necesita σ ≥ ' + sigy.toFixed(2) + ' MPa.'; }
  $('warn').classList.toggle('hide', S.phi + S.lam >= 90 - 1e-6);
  Plotly.react('diagram', diagramTraces(), layD, cfg);
  Plotly.restyle('map', { x: [[S.phi]], y: [[S.lam]], name: ['Actual: m = ' + m.toFixed(3)] }, [2]);
}
const go = throttled(update);
$('phi').addEventListener('input', e => { S.phi = parseFloat(e.target.value); go(); });
$('lam').addEventListener('input', e => { S.lam = parseFloat(e.target.value); go(); });
$('sigma').addEventListener('input', e => { const v = parseFloat(e.target.value); if (isFinite(v)) { S.sigma = v; go(); } });
$('tauc').addEventListener('input', e => { const v = parseFloat(e.target.value); if (isFinite(v) && v > 0) { S.tauc = v; go(); } });
update();
if (window.ResizeObserver) new ResizeObserver(() => { Plotly.Plots.resize('diagram'); Plotly.Plots.resize('map'); }).observe(document.body);
</script></body></html>
"""

ORIENT_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">__PLOTLY__<style>__CSS__</style></head><body>
<div class="grid">
  <div>
    <h3 id="title"></h3>
    <div class="modes" id="modes">
      <label><input type="radio" name="mode" value="ang" checked> Ángulos (θ, ψ) respecto a la red</label>
      <label><input type="radio" name="mode" value="uvw"> Dirección cristalográfica [u v w]</label>
      <label><input type="radio" name="mode" value="plane"> Rotar φ dentro del plano (normal, dirección) de un sistema</label>
    </div>
    <div id="cA">
      <div class="sl"><div class="lab" id="thLab"></div><input type="range" id="theta" min="0" max="90" step="0.5"></div>
      <div class="sl"><div class="lab" id="psLab"></div><input type="range" id="psi" min="0" max="360" step="0.5"></div>
    </div>
    <div id="cB" class="hide">
      <label class="f">Cargar orientación de ejemplo<select id="preset"></select></label>
      <div class="row2" style="grid-template-columns:1fr 1fr 1fr;margin-top:6px">
        <label class="f">u<input type="number" id="u" step="1"></label>
        <label class="f">v<input type="number" id="v" step="1"></label>
        <label class="f">w<input type="number" id="w" step="1"></label>
      </div>
      <div class="cap" id="uvwTxt"></div>
    </div>
    <div id="cC" class="hide">
      <label class="f">Sistema de deslizamiento<select id="sysSel"></select></label>
      <div class="sl"><div class="lab">φ : ángulo entre la normal al plano de deslizamiento y el esfuerzo: <b id="phiSv"></b></div>
        <input type="range" id="phiS" min="0" max="90" step="0.5"></div>
      <div class="cap" id="planeTxt"></div>
    </div>
    <div class="row2" style="margin-top:8px">
      <label class="f">Esfuerzo aplicado σ (MPa)<input type="number" id="sigma" min="0" step="1"></label>
      <label class="f">Esfuerzo cortante crítico τ<sub>c</sub> (MPa)<input type="number" id="tauc" min="0.01" step="1"></label>
    </div>
    <div id="results"></div>
  </div>
  <div>
    <div class="pill">
      <label><input type="radio" name="draw" value="best" checked> El que se activa primero</label>
      <label><input type="radio" name="draw" value="other"> Elegir otro</label>
    </div>
    <select id="drawSel" class="hide"></select>
    <div id="cell" style="height:540px"></div>
  </div>
</div>
<div id="bars" style="height:350px;margin-top:6px"></div>
<div class="tw"><table id="tbl"></table></div>
<div class="cap">Las direcciones y planos con todos los signos invertidos son equivalentes (paralelos), por lo que φ y λ se reportan entre 0° y 90°.</div>
<script>
const P = __DATA__;
__COMMON__
const th = setupTheme(P);
const $ = id => document.getElementById(id);
const S = { mode: 'ang', theta: 35, psi: 20, u: P.presets[0][1][0], v: P.presets[0][1][1], w: P.presets[0][1][2],
            sysIdx: 0, phiS: 45, sigma: 100, tauc: 40, draw: 'best', drawIdx: 0 };
const refAxis = P.hex ? '[0001]' : '[001]', azRef = P.hex ? 'a₁' : '[100]';
const cfg = { displayModeBar: false, responsive: true };

$('title').textContent = P.struct + ': ' + P.family;
$('thLab').innerHTML = 'θ : ángulo del esfuerzo respecto al eje ' + refAxis + ': <b id="thv"></b>';
$('psLab').innerHTML = 'ψ : azimut del esfuerzo, medido desde ' + azRef + ': <b id="psv"></b>';
$('theta').value = S.theta; $('psi').value = S.psi; $('phiS').value = S.phiS;
$('u').value = S.u; $('v').value = S.v; $('w').value = S.w;
$('sigma').value = S.sigma; $('tauc').value = S.tauc;
P.presets.forEach((p, k) => { const o = document.createElement('option'); o.value = k; o.textContent = p[0]; $('preset').appendChild(o); });
P.systems.forEach((s, k) => {
  for (const id of ['sysSel', 'drawSel']) { const o = document.createElement('option'); o.value = k;
    o.textContent = 'S' + (k + 1) + ': ' + s.planeTxt + ' ' + s.dirTxt; $(id).appendChild(o); } });

function computeT() {
  if (S.mode === 'ang') {
    const a = S.theta * D2R, b = S.psi * D2R;
    return [Math.sin(a) * Math.cos(b), Math.sin(a) * Math.sin(b), Math.cos(a)];
  }
  if (S.mode === 'uvw') {
    const v = P.hex ? [S.u - 0.5 * S.v, S.v * Math.sqrt(3) / 2, S.w * P.ca] : [S.u, S.v, S.w];
    return nrm(v) < 1e-9 ? null : unit(v);
  }
  const s = P.systems[S.sysIdx], p = S.phiS * D2R;
  return addv(mulv(s.n, Math.cos(p)), mulv(s.d, Math.sin(p)));
}

function hex4txt(u, v, w) {
  let U = 2*u - v, V = 2*v - u, T = -(u + v), W = 3*w;
  const g = gcd(gcd(Math.abs(U), Math.abs(V)), gcd(Math.abs(T), Math.abs(W))) || 1;
  return '[' + [U/g, V/g, T/g, W/g].map(barTxt).join('') + ']';
}

function stressTraces(t) {
  const c = P.center, L = P.Lt;
  return arrowTraces(addv(c, mulv(t, -L)), addv(c, mulv(t, L)), P.c_force, 'Fuerza aplicada/Esfuerzo σ', 'σ', { cone: 0.25 });
}

function cellData(idx, t) {
  const data = JSON.parse(JSON.stringify(P.data[idx]));
  if (th.dark) data.forEach(tr => { if (tr.line && tr.line.color === 'rgb(90,90,90)') tr.line.color = '#bdbdbd'; });
  return data.concat(stressTraces(t));
}
const layC = themedLayout(P.layout, th, { height: 540 });

function update() {
  const showA = S.mode === 'ang', showB = S.mode === 'uvw', showC = S.mode === 'plane';
  $('cA').classList.toggle('hide', !showA); $('cB').classList.toggle('hide', !showB); $('cC').classList.toggle('hide', !showC);
  $('thv').textContent = S.theta.toFixed(1) + '°'; $('psv').textContent = S.psi.toFixed(1) + '°';
  $('phiSv').textContent = S.phiS.toFixed(1) + '°';
  $('planeTxt').textContent = 'Esfuerzo coplanar con n y d ⇒ λ = 90° − φ = ' + (90 - S.phiS).toFixed(1) + '°';
  $('uvwTxt').textContent = P.hex ? '[' + S.u + ' ' + S.v + ' ' + S.w + '] → [UVTW] = ' + hex4txt(S.u, S.v, S.w) +
    '  (U=(2u−v)/3, V=(2v−u)/3, T=−(U+V), W=w)' : '';

  const t = computeT();
  const res = $('results');
  if (!t) { res.innerHTML = '<div class="msg err">La dirección [0 0 0] no es válida.</div>'; return; }

  const rows = evalSystems(P.systems, t, S.sigma, S.tauc);
  const mmax = Math.max(...rows.map(r => r.m));
  const tol = 1e-6;
  const isFirst = r => mmax >= tol && r.m >= mmax - tol;
  const firsts = rows.filter(isFirst);
  let html = '';
  if (mmax < tol) {
    html += '<div class="msg err">m = 0 en todos los sistemas preferenciales: ninguno se activa con esta orientación.</div>';
  } else {
    const sy = S.tauc / mmax, tauF = S.sigma * mmax, dlt = tauF - S.tauc;
    const f0 = firsts[0], s0 = P.systems[f0.i];
    html += '<div class="metrics">' +
      '<div class="card"><div class="k">' + (firsts.length === 1 ? 'Sistema que se activa primero' : 'Sistemas que se activan primero') +
      '</div><div class="v">' + (firsts.length === 1 ? s0.planeTxt + ' ' + s0.dirTxt : firsts.length + ' simultáneos') + '</div></div>' +
      '<div class="card"><div class="k">Factor de Schmid máximo m</div><div class="v">' + mmax.toFixed(4) + '</div></div>' +
      '<div class="card"><div class="k">σ necesario = τ<sub>c</sub> / m</div><div class="v">' + sy.toFixed(2) + ' MPa</div></div>' +
      '<div class="card"><div class="k">τ<sub>R</sub> con σ aplicado</div><div class="v">' + tauF.toFixed(2) + ' MPa</div>' +
      '<div class="k" style="color:' + (dlt >= 0 ? 'var(--ok)' : 'var(--err)') + '">' + (dlt >= 0 ? '+' : '') + dlt.toFixed(2) + ' vs τ<sub>c</sub></div></div></div>';
    if (firsts.length > 1) html += '<div class="cap">Empate por simetría de la orientación: ' +
      firsts.map(r => P.systems[r.i].planeTxt + P.systems[r.i].dirTxt).join(', ') + '</div>';
    const rest = rows.filter(r => !isFirst(r) && r.m > tol).sort((a, b) => b.m - a.m);
    if (rest.length) { const nx = rest[0], sn = P.systems[nx.i];
      html += '<div class="cap">Deslizamiento único hasta σ = ' + sy.toFixed(1) + ' MPa; el siguiente sistema (' + sn.planeTxt + sn.dirTxt +
        ', m = ' + nx.m.toFixed(3) + ') se activaría a σ = ' + nx.sigc.toFixed(1) + ' MPa (deslizamiento múltiple).</div>'; }
    if (P.hex && mmax < 0.05) html += '<div class="msg warn">Con m ≈ 0 en los tres sistemas basales, en HCP se activarían otros sistemas ' +
      '(prismáticos o piramidales) con τ<sub>c</sub> mayor; esta app solo considera la familia preferencial.</div>';
  }
  res.innerHTML = html;

  // celda 3D
  const best = firsts.length ? firsts[0].i : 0;
  const di = S.draw === 'best' ? best : S.drawIdx;
  $('drawSel').classList.toggle('hide', S.draw !== 'other');
  Plotly.react('cell', cellData(di, t), layC, cfg);

  // barras de τ_R
  const colors = rows.map(r => isFirst(r) ? 'rgb(30,150,70)' : (r.tauR >= S.tauc ? 'rgb(240,140,0)' : 'rgb(160,160,160)'));
  Plotly.react('bars', [{ type: 'bar', x: rows.map(r => 'S' + (r.i + 1)), y: rows.map(r => r.tauR), marker: { color: colors },
    customdata: rows.map(r => [P.systems[r.i].planeTxt, P.systems[r.i].dirTxt, r.phi.toFixed(2), r.lam.toFixed(2), r.m.toFixed(4)]),
    hovertemplate: '%{customdata[0]} %{customdata[1]}<br>φ=%{customdata[2]}°  λ=%{customdata[3]}°<br>m=%{customdata[4]}<br>τ_R=%{y:.2f} MPa<extra></extra>' }],
    themedLayout({}, th, { height: 350, margin: { l: 55, r: 10, t: 55, b: 45 },
      title: { text: 'Esfuerzo cortante resuelto τ_R en cada sistema preferencial (verde: se activa primero · naranja: τ_R ≥ τ_c con σ aplicado)', font: { size: 13 } },
      xaxis: { title: 'Sistema de deslizamiento' }, yaxis: { title: 'τ_R (MPa)' },
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: S.tauc, y1: S.tauc, line: { color: 'red', dash: 'dash' } }],
      annotations: [{ xref: 'paper', x: 1, y: S.tauc, yanchor: 'bottom', xanchor: 'right', showarrow: false,
                      text: 'τ_c = ' + S.tauc.toFixed(2) + ' MPa', font: { color: 'red' } }] }), cfg);

  // tabla
  const sorted = rows.slice().sort((a, b) => b.m - a.m);
  let tb = '<tr><th>Sistema</th><th>Plano de deslizamiento</th><th>Dirección de deslizamiento</th><th>φ (°)</th><th>λ (°)</th>' +
           '<th>m = cos φ cos λ</th><th>τ_R (MPa)</th><th>σ necesario = τ_c/m (MPa)</th><th>Estado</th></tr>';
  sorted.forEach(r => {
    const s = P.systems[r.i];
    const est = isFirst(r) ? (r.tauR >= S.tauc ? '🟢 Primero (activo)' : '🟢 Primero') : (r.tauR >= S.tauc ? '🟠 Activo con σ aplicado' : '⚪ No activo');
    tb += '<tr><td>S' + (r.i + 1) + '</td><td>' + s.planeTxt + '</td><td>' + s.dirTxt + '</td><td>' + r.phi.toFixed(2) + '</td><td>' +
      r.lam.toFixed(2) + '</td><td>' + r.m.toFixed(4) + '</td><td>' + r.tauR.toFixed(3) + '</td><td>' +
      (isFinite(r.sigc) ? r.sigc.toFixed(2) : '—') + '</td><td>' + est + '</td></tr>';
  });
  $('tbl').innerHTML = tb;
}
const go = throttled(update);
const bindRange = (id, key) => $(id).addEventListener('input', e => { S[key] = parseFloat(e.target.value); go(); });
bindRange('theta', 'theta'); bindRange('psi', 'psi'); bindRange('phiS', 'phiS');
for (const k of ['u', 'v', 'w']) $(k).addEventListener('input', e => { const x = parseInt(e.target.value, 10); S[k] = isNaN(x) ? 0 : x; go(); });
$('sigma').addEventListener('input', e => { const x = parseFloat(e.target.value); if (isFinite(x)) { S.sigma = x; go(); } });
$('tauc').addEventListener('input', e => { const x = parseFloat(e.target.value); if (isFinite(x) && x > 0) { S.tauc = x; go(); } });
document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', e => { S.mode = e.target.value; go(); }));
document.querySelectorAll('input[name=draw]').forEach(r => r.addEventListener('change', e => { S.draw = e.target.value; go(); }));
$('sysSel').addEventListener('change', e => { S.sysIdx = parseInt(e.target.value, 10); go(); });
$('drawSel').addEventListener('change', e => { S.drawIdx = parseInt(e.target.value, 10); go(); });
$('preset').addEventListener('change', e => { const p = P.presets[parseInt(e.target.value, 10)][1];
  S.u = p[0]; S.v = p[1]; S.w = p[2]; $('u').value = S.u; $('v').value = S.v; $('w').value = S.w; go(); });
update();
if (window.ResizeObserver) new ResizeObserver(() => { Plotly.Plots.resize('cell'); Plotly.Plots.resize('bars'); }).observe(document.body);
</script></body></html>
"""


def _dark_theme():
    """True / False si Streamlit informa el tema; None => el componente usa el del sistema."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return None


def _render(template, payload, height):
    html = (template.replace("__PLOTLY__", PLOTLY_TAGS)
            .replace("__CSS__", BASE_CSS)
            .replace("__COMMON__", COMMON_JS)
            .replace("__DATA__", json.dumps(payload)))
    if hasattr(st, "iframe"):  # Streamlit reciente
        st.iframe(html, height=height)
    else:  # versiones anteriores
        components.html(html, height=height, scrolling=True)


@st.cache_data(show_spinner=False)
def schmid_payload():
    layout = json.loads(_scene(go.Figure(), 440).to_json())["layout"]
    return dict(sigma=100.0, tauc=40.0, layout=layout, c_plane=C_PLANE, c_normal=C_NORMAL,
                c_slip=C_SLIP, c_force=C_FORCE)


@st.cache_data(show_spinner=False)
def orient_payload(struct, ca):
    systems = build_systems(struct, ca)
    figs = [json.loads(cell_figure(struct, s, None, ca, 540).to_json()) for s in systems]
    hexa = struct == "HCP"
    return dict(
        struct=struct, family=FAMILY_LABEL[struct], hex=hexa, ca=ca,
        systems=[dict(planeTxt=s["planeTxt"], dirTxt=s["dirTxt"], n=[float(x) for x in s["n"]],
                      d=[float(x) for x in s["d"]]) for s in systems],
        data=[f["data"] for f in figs], layout=figs[0]["layout"],
        center=[0.0, 0.0, ca / 2] if hexa else [0.5, 0.5, 0.5], Lt=1.0 if hexa else 0.95,
        presets=[[k, list(v)] for k, v in PRESETS[struct].items()], c_force=C_FORCE,
    )


# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("Parámetros")
    struct = st.radio("Estructura cristalina", ["FCC", "BCC", "HCP"], horizontal=True)
    st.caption(f"Sistema preferencial de deslizamiento: **{FAMILY_LABEL[struct]}**")
    ca = 1.633
    if struct == "HCP":
        hcp_choice = st.selectbox("Relación c/a de la celda HCP", list(HCP_CA))
        ca = HCP_CA[hcp_choice]
    st.divider()
    st.markdown(
        "**Ley de Schmid**\n\n"
        "τ_R = σ cos φ cos λ\n\n"
        "m = cos φ cos λ  (factor de Schmid)\n\n"
        "m grande ⇒ orientación favorable ⇒ menor σ necesaria para deslizar\n\n"
        "m pequeño ⇒ orientación desfavorable ⇒ mayor σ necesaria"
    )

st.title("Ley de Schmid y deslizamiento de dislocaciones")

tab1, tab2, tab3, tab4 = st.tabs([
    "1 · Ley de Schmid (φ y λ)",
    "2 · ¿Qué sistema se activa primero?",
    "3 · Estructuras y familias preferenciales",
    "4 · Equipo",
])

with tab1:
    pl = dict(schmid_payload())
    pl["dark"] = _dark_theme()
    _render(SCHMID_HTML, pl, 990)

with tab2:
    pl = dict(orient_payload(struct, ca))
    pl["dark"] = _dark_theme()
    _render(ORIENT_HTML, pl, 1380 if struct != "HCP" else 1090)

with tab3:
    st.subheader("Sistemas preferenciales de deslizamiento")
    tabla = pd.DataFrame(
        {
            "": ["Plano y dirección de deslizamiento", "Número de planos preferenciales de deslizamiento",
                 "Direcciones de deslizamiento (por plano)", "Sistemas preferenciales de deslizamiento",
                 "Densidad de empaquetamiento"],
            "FCC": [FAMILY_LABEL["FCC"], "4", "3", "12", "74 %"],
            "BCC": [FAMILY_LABEL["BCC"], "6", "2", "12", "68 %"],
            "HCP": [FAMILY_LABEL["HCP"], "1", "3", "3", "74 %"],
        }
    )
    show_table(tabla)
    cols = st.columns(3)
    for col, stc in zip(cols, ["FCC", "BCC", "HCP"]):
        sy = build_systems(stc, ca if stc == "HCP" else 1.633)
        with col:
            st.markdown(f"**{stc}** · {FAMILY_LABEL[stc]}")
            show_plot(cell_figure(stc, sy[0], None, ca if stc == "HCP" else 1.633, height=380), key=f"cell3_{stc}")
            st.caption("Átomos rojos: contenidos en el plano de deslizamiento.")

    with st.expander(f"Los {len(build_systems(struct, ca))} sistemas de {struct} considerados"):
        sy = build_systems(struct, ca)
        show_table(pd.DataFrame({
            "Sistema": [f"S{i}" for i in range(1, len(sy) + 1)],
            "Plano de deslizamiento": [s["planeTxt"] for s in sy],
            "Dirección de deslizamiento": [s["dirTxt"] for s in sy],
        }))

with tab4:
    st.markdown(
        "- Montes Lima Juan Pablo\n"
        "- Vergara Negrete Rodrigo\n"
        "- Vilchis Vélez Alejandro Raúl"
    )
