"""
Visualizador de la ley de Schmid y sistemas de deslizamiento (FCC, BCC, HCP)
Materiales II – Facultad de Ingeniería, UNAM
Tema 2: Deformación plástica a través del movimiento de las dislocaciones

Notación tomada de las notas de clase:
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
from math import gcd

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Ley de Schmid · Materiales II",
    page_icon="🔷",
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
# Notación: barra sobre los índices negativos, como en las notas (p. ej. 1̅ 1 1)
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
# Planos de la familia {111} tal como aparecen en las notas (p. 32)
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
    """Conversión [uvw] -> [UVTW] con las fórmulas de las notas (p. 48):
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


def evaluate(systems, t, sigma, tau_c):
    """Ley de Schmid para todos los sistemas. t: eje de tracción (vector unitario)."""
    rows = []
    for i, s in enumerate(systems, 1):
        cphi = min(abs(float(np.dot(t, s["n"]))), 1.0)
        clam = min(abs(float(np.dot(t, s["d"]))), 1.0)
        m = cphi * clam
        rows.append(
            dict(
                i=i,
                planeTxt=s["planeTxt"],
                dirTxt=s["dirTxt"],
                phi=float(np.degrees(np.arccos(cphi))),
                lam=float(np.degrees(np.arccos(clam))),
                m=m,
                tauR=sigma * m,
                sig_c=(tau_c / m) if m > 1e-9 else np.inf,
            )
        )
    return pd.DataFrame(rows)


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
            aspectmode="data", camera=dict(eye=dict(x=1.6, y=-1.7, z=1.0)),
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


def schmid_diagram(phi_deg, lam_deg, height=430):
    """Esquema de los tres vectores: normal n, dirección de deslizamiento d y eje de esfuerzo."""
    c1, c2 = np.cos(np.radians(phi_deg)), np.cos(np.radians(lam_deg))
    ty2 = 1 - c1**2 - c2**2
    t = np.array([c2, np.sqrt(max(ty2, 0.0)), c1])
    t /= np.linalg.norm(t)
    n = np.array([0.0, 0.0, 1.0])
    d = np.array([1.0, 0.0, 0.0])
    fig = go.Figure()
    # disco = plano de deslizamiento (perpendicular a n)
    th = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(
        go.Mesh3d(
            x=np.r_[0, 1.15 * np.cos(th)], y=np.r_[0, 1.15 * np.sin(th)], z=np.zeros(61),
            i=[0] * 59, j=list(range(1, 60)), k=list(range(2, 61)),
            color=C_PLANE, opacity=0.45, name="Plano de deslizamiento", showlegend=True, hoverinfo="name",
        )
    )
    _add_arrow(fig, (0, 0, 0), 1.3 * n, C_NORMAL, "Normal al plano de deslizamiento", label="n")
    _add_arrow(fig, (0, 0, 0), 1.3 * d, C_SLIP, "Dirección de deslizamiento", label="d")
    _add_arrow(fig, -1.3 * t, 1.3 * t, C_FORCE, "Fuerza aplicada/Esfuerzo σ", label="σ", cone=0.3)

    def arc(u, v, r, text):
        ang = np.arccos(np.clip(np.dot(u, v), -1, 1))
        if ang < 1e-6:
            return
        s = np.linspace(0, 1, 30)
        pts = np.array([(np.sin((1 - x) * ang) * u + np.sin(x * ang) * v) / np.sin(ang) for x in s]) * r
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                                   line=dict(color="rgb(70,70,70)", width=3), showlegend=False, hoverinfo="skip"))
        mid = pts[len(pts) // 2] * 1.12
        fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode="text", text=[text],
                                   textfont=dict(size=16, color="black"), showlegend=False, hoverinfo="skip"))

    arc(n, t, 0.6, "φ")
    arc(d, t, 0.8, "λ")
    fig = _scene(fig, height)
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=1.5, y=-1.6, z=0.8))))
    return fig


# ----------------------------------------------------------------------------
# Datos de las notas
# ----------------------------------------------------------------------------
HCP_CA = {
    "Ideal (c/a = 1.633)": 1.633,
    "Mg (c/a ≈ 1.624)": 1.624,
    "Ti (c/a ≈ 1.587)": 1.587,
    "Zn (c/a ≈ 1.856)": 1.856,
    "Cd (c/a ≈ 1.886)": 1.886,
}

PRESETS = {
    "FCC": {"[100]  (ejercicio p. 42)": (1, 0, 0), "[110]": (1, 1, 0), "[111]": (1, 1, 1),
            "[123]": (1, 2, 3), "[2 1̅ 3̅]  (ejercicio p. 45)": (2, -1, -3)},
    "BCC": {"[123]  (ejercicio p. 43)": (1, 2, 3), "[121]  (ejercicio p. 44)": (1, 2, 1),
            "[101]  (ejercicio p. 56)": (1, 0, 1), "[100]": (1, 0, 0), "[111]": (1, 1, 1),
            "[2 1̅ 3̅]  (ejercicio p. 45)": (2, -1, -3)},
    "HCP": {"[001]  (eje c)": (0, 0, 1), "[100]": (1, 0, 0), "[101]": (1, 0, 1),
            "[102]": (1, 0, 2), "[111]": (1, 1, 1)},
}

# ----------------------------------------------------------------------------
# Estado y barra lateral
# ----------------------------------------------------------------------------
def _defaults():
    for k, v in dict(sigma=100.0, tau_c=40.0, phi=45.0, lam=45.0, u=1, v=2, w=3).items():
        st.session_state.setdefault(k, v)


_defaults()


def _load_preset(struct):
    name = st.session_state.get(f"preset_{struct}")
    if name in PRESETS[struct]:
        u, v, w = PRESETS[struct][name]
        st.session_state["u"], st.session_state["v"], st.session_state["w"] = u, v, w


with st.sidebar:
    st.title("Parámetros")
    struct = st.radio("Estructura cristalina", ["FCC", "BCC", "HCP"], horizontal=True)
    st.caption(f"Sistema preferencial de deslizamiento: **{FAMILY_LABEL[struct]}**")
    sigma = st.number_input("Esfuerzo aplicado σ (MPa)", min_value=0.0, max_value=10000.0,
                            step=1.0, format="%.2f", key="sigma")
    tau_c = st.number_input("Esfuerzo cortante crítico τ_c (MPa)", min_value=0.01, max_value=5000.0,
                            step=1.0, format="%.3f", key="tau_c",
                            help="τ_c (τ_CRSS): esfuerzo cortante resuelto necesario para iniciar el deslizamiento.")
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
st.caption("Materiales II · Tema 2: Deformación plástica a través del movimiento de las dislocaciones")

tab1, tab2, tab3 = st.tabs([
    "1 · Ley de Schmid (φ y λ)",
    "2 · ¿Qué sistema se activa primero?",
    "3 · Estructuras y familias preferenciales",
])

# ============================================================================
# TAB 1: ley de Schmid con φ y λ
# ============================================================================
with tab1:
    c_left, c_right = st.columns([1, 1.15])

    with c_left:
        st.latex(r"\tau_R = \sigma\,\cos\phi\,\cos\lambda \qquad m=\cos\phi\,\cos\lambda")
        phi = st.slider("φ : ángulo entre la normal al plano de deslizamiento y el esfuerzo aplicado (°)",
                        0.0, 90.0, step=0.1, key="phi")
        lam = st.slider("λ : ángulo entre la dirección de deslizamiento y el esfuerzo aplicado (°)",
                        0.0, 90.0, step=0.1, key="lam")

        m1 = np.cos(np.radians(phi)) * np.cos(np.radians(lam))
        tauR1 = sigma * m1
        sig_y1 = tau_c / m1 if m1 > 1e-9 else np.inf

        k1, k2 = st.columns(2)
        k1.metric("Factor de Schmid m", f"{m1:.4f}")
        k2.metric("Esfuerzo cortante resuelto τ_R", f"{tauR1:.3f} MPa")
        k3, k4 = st.columns(2)
        k3.metric("σ necesario = τ_c / m", "∞" if np.isinf(sig_y1) else f"{sig_y1:.2f} MPa")
        k4.metric("φ + λ", f"{phi + lam:.1f}°")

        if np.isinf(sig_y1):
            st.error("m = 0: el esfuerzo no produce cortante en la dirección de deslizamiento.")
        elif tauR1 >= tau_c:
            st.success(f"τ_R = {tauR1:.2f} MPa ≥ τ_c = {tau_c:.2f} MPa → **el cristal se deforma plásticamente**.")
        else:
            st.info(f"τ_R = {tauR1:.2f} MPa < τ_c = {tau_c:.2f} MPa → el cristal **no** se deforma (aún). "
                    f"Necesita σ ≥ {sig_y1:.2f} MPa.")

        if phi + lam < 90 - 1e-6:
            st.warning("Si la normal al plano y la dirección de deslizamiento son perpendiculares, siempre "
                       "φ + λ ≥ 90° (= 90° cuando el eje de esfuerzo, la normal y la dirección son coplanares). "
                       "El cálculo usa tus valores tal cual (como en los problemas de clase); "
                       "el dibujo muestra la geometría compatible más cercana.")

    with c_right:
        show_plot(schmid_diagram(phi, lam), key="schmid3d")

    # mapa de m(φ, λ)
    g = np.linspace(0, 90, 91)
    P, Lm = np.meshgrid(g, g)
    Z = np.cos(np.radians(P)) * np.cos(np.radians(Lm))
    Z = np.where(P + Lm >= 90 - 1e-9, Z, np.nan)
    fig_m = go.Figure()
    fig_m.add_trace(go.Contour(x=g, y=g, z=Z, colorscale="Viridis", zmin=0, zmax=0.5,
                               contours=dict(start=0, end=0.5, size=0.05, showlabels=True),
                               colorbar=dict(title="m"), hovertemplate="φ=%{x}°<br>λ=%{y}°<br>m=%{z:.3f}<extra></extra>"))
    fig_m.add_trace(go.Scatter(x=[0, 90], y=[90, 0], mode="lines", line=dict(color="white", dash="dash"),
                               name="φ + λ = 90° (coplanar)", hoverinfo="skip"))
    fig_m.add_trace(go.Scatter(x=[phi], y=[lam], mode="markers",
                               marker=dict(size=14, color="red", line=dict(color="white", width=2)),
                               name=f"Actual: m = {m1:.3f}"))
    fig_m.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10),
                        title="Factor de Schmid m en función de φ y λ (máximo: m = 0.5 con φ = λ = 45°)",
                        xaxis_title="φ (°)", yaxis_title="λ (°)", legend=dict(orientation="h", y=-0.2))
    show_plot(fig_m, key="mmap")
    st.caption("La zona en blanco (φ + λ < 90°) es geométricamente imposible cuando la normal al plano y la "
               "dirección de deslizamiento son perpendiculares.")

# ============================================================================
# TAB 2: orientación del eje de tracción -> sistema que se activa primero
# ============================================================================
with tab2:
    systems = build_systems(struct, ca)
    left, right = st.columns([1, 1.25])

    with left:
        st.subheader(f"{struct}: {FAMILY_LABEL[struct]}")
        ref_axis = "[0001]" if struct == "HCP" else "[001]"
        modo = st.radio(
            "¿Cómo defines el eje de tracción?",
            ["Ángulos (θ, ψ) respecto a la red",
             "Dirección cristalográfica [u v w]",
             "Rotar φ dentro del plano (normal, dirección) de un sistema"],
        )

        if modo.startswith("Ángulos"):
            theta = st.slider(f"θ : ángulo del esfuerzo respecto al eje {ref_axis} (°)", 0.0, 90.0, 35.0, 0.5)
            psi = st.slider("ψ : azimut del esfuerzo, medido desde "
                            + ("a₁" if struct == "HCP" else "[100]") + " (°)", 0.0, 360.0, 20.0, 0.5)
            th, ps = np.radians(theta), np.radians(psi)
            t_axis = np.array([np.sin(th) * np.cos(ps), np.sin(th) * np.sin(ps), np.cos(th)])
            axis_txt = None

        elif modo.startswith("Dirección"):
            st.selectbox("Cargar orientación de ejemplo", list(PRESETS[struct]), index=None,
                         placeholder="Elegir…", key=f"preset_{struct}", on_change=_load_preset, args=(struct,))
            cu, cv, cw = st.columns(3)
            u = cu.number_input("u", step=1, key="u")
            v = cv.number_input("v", step=1, key="v")
            w = cw.number_input("w", step=1, key="w")
            if u == 0 and v == 0 and w == 0:
                st.error("La dirección [0 0 0] no es válida.")
                st.stop()
            if struct == "HCP":
                vec = hex3_to_cart(u, v, w, ca)
                U4 = hex3_to_4(int(u), int(v), int(w))
                axis_txt = f"[{u} {v} {w}] → [UVTW] = {fmt(U4, '[]', ' ')}  (U=(2u−v)/3, V=(2v−u)/3, T=−(U+V), W=w)"
            else:
                vec = np.array([u, v, w], float)
                axis_txt = None
            t_axis = vec / np.linalg.norm(vec)

        else:
            opts = [f"{s['planeTxt']} {s['dirTxt']}" for s in systems]
            idx = st.selectbox("Sistema de deslizamiento", range(len(systems)), format_func=lambda i: opts[i])
            phi_s = st.slider("φ : ángulo entre la normal al plano de deslizamiento y el esfuerzo (°)",
                              0.0, 90.0, 45.0, 0.5)
            s0 = systems[idx]
            t_axis = np.cos(np.radians(phi_s)) * s0["n"] + np.sin(np.radians(phi_s)) * s0["d"]
            axis_txt = f"Esfuerzo coplanar con n y d ⇒ λ = 90° − φ = {90 - phi_s:.1f}°"

        if axis_txt:
            st.caption(axis_txt)

        df = evaluate(systems, t_axis, sigma, tau_c)
        m_max = float(df["m"].max())
        tol = 1e-6
        first_mask = df["m"] >= m_max - tol
        firsts = df[first_mask]

        st.divider()
        if m_max < 1e-6:
            st.error("m = 0 en todos los sistemas preferenciales: ninguno se activa con esta orientación.")
        else:
            sig_y = tau_c / m_max
            a, b = st.columns(2)
            if len(firsts) == 1:
                r0 = firsts.iloc[0]
                a.metric("Sistema que se activa primero", f"{r0.planeTxt} {r0.dirTxt}")
            else:
                a.metric("Sistemas que se activan primero", f"{len(firsts)} simultáneos")
            b.metric("Factor de Schmid máximo m", f"{m_max:.4f}")
            c, d_ = st.columns(2)
            c.metric("σ necesario = τ_c / m", f"{sig_y:.2f} MPa")
            tauR_first = sigma * m_max
            d_.metric("τ_R con σ aplicado", f"{tauR_first:.2f} MPa",
                      delta=f"{tauR_first - tau_c:+.2f} vs τ_c")
            if len(firsts) > 1:
                st.caption("Empate por simetría de la orientación: " +
                           ", ".join(f"{r.planeTxt}{r.dirTxt}" for r in firsts.itertuples()))
            rest = df[~first_mask & (df["m"] > 1e-6)]
            if len(rest):
                nxt = rest.sort_values("m", ascending=False).iloc[0]
                st.caption(
                    f"Deslizamiento único hasta σ = {sig_y:.1f} MPa; el siguiente sistema "
                    f"({nxt.planeTxt}{nxt.dirTxt}, m = {nxt.m:.3f}) se activaría a σ = {nxt.sig_c:.1f} MPa "
                    "(deslizamiento múltiple)."
                )
            if struct == "HCP" and m_max < 0.05:
                st.warning("Con m ≈ 0 en los tres sistemas basales, en HCP se activarían otros sistemas "
                           "(prismáticos o piramidales) con τ_c mayor; esta app solo considera la familia preferencial.")

    # --- selección de sistema a dibujar ---
    with right:
        order = df.sort_values("m", ascending=False)
        best_i = int(firsts.iloc[0]["i"]) - 1 if m_max >= 1e-6 else 0
        draw_mode = st.radio("Sistema a dibujar en la celda", ["El que se activa primero", "Elegir otro"],
                             horizontal=True)
        draw_i = best_i
        if draw_mode == "Elegir otro":
            labels = [f"{r.i}: {r.planeTxt} {r.dirTxt}  (m = {r.m:.3f})" for r in df.itertuples()]
            draw_i = st.selectbox("Sistema", range(len(systems)), format_func=lambda i: labels[i], key="draw_sel")
        show_plot(cell_figure(struct, systems[draw_i], t_axis, ca), key="cell2")

    # --- barras de τ_R por sistema ---
    colors = []
    for r in df.itertuples():
        if r.m >= m_max - tol and m_max >= 1e-6:
            colors.append("rgb(30,150,70)")
        elif r.tauR >= tau_c:
            colors.append("rgb(240,140,0)")
        else:
            colors.append("rgb(160,160,160)")
    fig_b = go.Figure(
        go.Bar(
            x=[f"S{i}" for i in df["i"]], y=df["tauR"], marker_color=colors,
            customdata=np.stack([df["planeTxt"], df["dirTxt"], df["phi"].round(2), df["lam"].round(2),
                                 df["m"].round(4)], axis=-1),
            hovertemplate="%{customdata[0]} %{customdata[1]}<br>φ=%{customdata[2]}°  λ=%{customdata[3]}°"
                          "<br>m=%{customdata[4]}<br>τ_R=%{y:.2f} MPa<extra></extra>",
        )
    )
    fig_b.add_hline(y=tau_c, line_dash="dash", line_color="red",
                    annotation_text=f"τ_c = {tau_c:.2f} MPa", annotation_position="top left")
    fig_b.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                        title="Esfuerzo cortante resuelto τ_R en cada sistema preferencial (verde: se activa primero · "
                              "naranja: τ_R ≥ τ_c con σ aplicado)",
                        xaxis_title="Sistema de deslizamiento", yaxis_title="τ_R (MPa)")
    show_plot(fig_b, key="bars")

    # --- tabla ---
    tbl = order.copy()
    estados = []
    for r in tbl.itertuples():
        if r.m >= m_max - tol and m_max >= 1e-6:
            estados.append("🟢 Primero (activo)" if r.tauR >= tau_c else "🟢 Primero")
        elif r.tauR >= tau_c:
            estados.append("🟠 Activo con σ aplicado")
        else:
            estados.append("⚪ No activo")
    out = pd.DataFrame({
        "Sistema": [f"S{i}" for i in tbl["i"]],
        "Plano de deslizamiento": tbl["planeTxt"],
        "Dirección de deslizamiento": tbl["dirTxt"],
        "φ (°)": tbl["phi"].round(2),
        "λ (°)": tbl["lam"].round(2),
        "m = cos φ cos λ": tbl["m"].round(4),
        "τ_R (MPa)": tbl["tauR"].round(3),
        "σ necesario = τ_c/m (MPa)": tbl["sig_c"].replace(np.inf, np.nan).round(2),
        "Estado": estados,
    })
    show_table(out)
    st.caption("Las direcciones y planos con todos los signos invertidos son equivalentes (paralelos), "
               "por lo que φ y λ se reportan entre 0° y 90°.")

# ============================================================================
# TAB 3: estructuras y tabla de familias preferenciales
# ============================================================================
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
