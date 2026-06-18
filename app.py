"""
Interactive SVM Kernel Trick 3D Demo (Streamlit)

用 sklearn.svm.SVC 對同心圓資料 (make_circles) 做即時擬合, 展示:
  - 2D 決策邊界 (決策面 f=0 的等高線, 含 margin f=±1 虛線、support vectors)
  - 3D 決策函數曲面 f(x, y), 對應到 2D 邊界
  - 訓練準確率與模型統計

本地執行:
  pip install -r requirements.txt
  streamlit run app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import make_circles
from sklearn.svm import SVC

SCALE = 2.0   # 放大資料座標, 讓圖形比例好看
FACTOR = 0.3  # 內圈半徑 / 外圈半徑

KERNEL_LABELS = {"rbf": "RBF", "poly": "Polynomial", "linear": "Linear"}

INNER_COLOR = "#2563EB"   # 內圈 (class 0)
OUTER_COLOR = "#E11D48"   # 外圈 (class 1)
LINE_COLOR = "#334155"    # 決策邊界 / margin 線

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2.2rem; max-width: 1200px; }
    h1 { font-size: 2.4rem !important; font-weight: 700 !important; }
    h2, h3 { font-size: 1.55rem !important; font-weight: 600 !important; margin-top: 0.5rem !important; }
    [data-testid="stSidebar"] h2 { font-size: 1.3rem !important; }
    [data-testid="stSidebar"] label p { font-size: 1.05rem !important; }
    .stMarkdown p, [data-testid="stExpander"] p { font-size: 1.05rem !important; line-height: 1.6 !important; }
    [data-testid="stMetricValue"] { font-size: 1.9rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; font-weight: 600 !important; color: #475569 !important; }
</style>
"""

PLOT_FONT = dict(family="Segoe UI, Arial, sans-serif", size=15, color="#1E293B")


@st.cache_data
def generate_dataset(n_points, noise, seed):
    X, y = make_circles(n_samples=n_points, noise=noise, factor=FACTOR, random_state=seed)
    # make_circles 預設把外圈標為 0、內圈標為 1, 跟我們想要的「0=內、1=外」相反, 這裡依半徑校正
    if np.linalg.norm(X[y == 0], axis=1).mean() > np.linalg.norm(X[y == 1], axis=1).mean():
        y = 1 - y
    return X * SCALE, y


@st.cache_resource
def fit_model(X, y, kernel, C, gamma, degree):
    kwargs = dict(kernel=kernel, C=C)
    if kernel in ("rbf", "poly"):
        kwargs["gamma"] = gamma
    if kernel == "poly":
        kwargs["degree"] = degree
    clf = SVC(**kwargs)
    clf.fit(X, y)
    return clf


st.set_page_config(page_title="Interactive SVM Kernel Trick 3D Demo", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("Interactive SVM Kernel Trick 3D Demo")

with st.expander("Concept (核心概念)"):
    st.write(
        "同心圓資料在 2D 平面上無法用直線分開。"
        "Kernel 函數隱含地把資料映射到更高維空間, 在那裡一個超平面就能把兩類分開; "
        "把超平面的方程式 f(x, y) = 0 投影回 2D, 就是這裡看到的決策邊界。"
    )

with st.sidebar:
    st.header("Parameters")
    kernel = st.selectbox("Kernel", ["rbf", "poly", "linear"])
    C = st.slider("C (regularization)", 0.01, 50.0, 10.0, 0.01)

    gamma = None
    degree = None
    if kernel in ("rbf", "poly"):
        gamma = st.slider("Gamma", 0.01, 5.0, 1.0, 0.01)
    if kernel == "poly":
        degree = st.slider("Degree", 2, 6, 3)

    noise = st.slider("Noise", 0.0, 0.3, 0.06, 0.01)
    n_points = st.slider("Number of points", 50, 300, 120, 10)
    seed = st.number_input("Random seed", min_value=0, value=7, step=1)

X, y = generate_dataset(n_points, noise, seed)
clf = fit_model(X, y, kernel, C, gamma, degree)
accuracy = clf.score(X, y)
sv = clf.support_vectors_

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

label = KERNEL_LABELS[kernel]

# ---- 2D 決策邊界 ----
st.subheader("2D Decision Boundary")
fig2d = go.Figure()
fig2d.add_trace(go.Contour(
    x=xx[0], y=yy[:, 0], z=Z,
    contours=dict(start=-1, end=-1, size=1, coloring="none"),
    line=dict(width=1.5, dash="dash", color=LINE_COLOR),
    showscale=False, showlegend=False, name="margin (f=-1)",
))
fig2d.add_trace(go.Contour(
    x=xx[0], y=yy[:, 0], z=Z,
    contours=dict(start=1, end=1, size=1, coloring="none"),
    line=dict(width=1.5, dash="dash", color=LINE_COLOR),
    showscale=False, showlegend=False, name="margin (f=+1)",
))
fig2d.add_trace(go.Contour(
    x=xx[0], y=yy[:, 0], z=Z,
    contours=dict(start=0, end=0, size=1, coloring="none"),
    line=dict(width=3, color=LINE_COLOR),
    showscale=False, name="boundary (f=0)",
))
fig2d.add_trace(go.Scatter(
    x=X[y == 0, 0], y=X[y == 0, 1], mode="markers",
    marker=dict(color=INNER_COLOR, size=8, line=dict(width=0.5, color="white")),
    name="Inner (class 0)",
))
fig2d.add_trace(go.Scatter(
    x=X[y == 1, 0], y=X[y == 1, 1], mode="markers",
    marker=dict(color=OUTER_COLOR, size=8, line=dict(width=0.5, color="white")),
    name="Outer (class 1)",
))
fig2d.add_trace(go.Scatter(
    x=sv[:, 0], y=sv[:, 1], mode="markers",
    marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color="#0F172A", width=2)),
    name="Support vectors",
))
fig2d.update_layout(
    template="plotly_white",
    font=PLOT_FONT,
    xaxis_title="X", yaxis_title="Y",
    legend=dict(x=0.02, y=0.02, bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0", borderwidth=1),
    height=520,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor="#E2E8F0"),
    xaxis=dict(gridcolor="#E2E8F0"),
    plot_bgcolor="white",
)
st.plotly_chart(fig2d, use_container_width=True)
st.divider()

# ---- 3D 決策函數曲面 ----
st.subheader("3D Decision Function Surface")
fig3d = go.Figure()
fig3d.add_trace(go.Surface(
    x=xx, y=yy, z=Z, colorscale="RdYlBu", opacity=0.85,
    colorbar=dict(title=dict(text="f(x, y)", font=PLOT_FONT), tickfont=PLOT_FONT),
))
fig3d.add_trace(go.Scatter3d(
    x=X[y == 0, 0], y=X[y == 0, 1], z=clf.decision_function(X[y == 0]),
    mode="markers", marker=dict(size=4, color=INNER_COLOR), name="Inner",
))
fig3d.add_trace(go.Scatter3d(
    x=X[y == 1, 0], y=X[y == 1, 1], z=clf.decision_function(X[y == 1]),
    mode="markers", marker=dict(size=4, color=OUTER_COLOR), name="Outer",
))
fig3d.add_trace(go.Scatter3d(
    x=sv[:, 0], y=sv[:, 1], z=clf.decision_function(sv),
    mode="markers",
    marker=dict(size=6, color="rgba(0,0,0,0)", line=dict(color="#0F172A", width=3)),
    name="Support vectors",
))
fig3d.update_layout(
    template="plotly_white",
    font=PLOT_FONT,
    scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="f(x, y)",
        xaxis=dict(backgroundcolor="white", gridcolor="#E2E8F0"),
        yaxis=dict(backgroundcolor="white", gridcolor="#E2E8F0"),
        zaxis=dict(backgroundcolor="white", gridcolor="#E2E8F0"),
    ),
    height=600, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig3d, use_container_width=True)
st.divider()

# ---- 統計資訊 ----
with st.container(border=True):
    cols = st.columns(5)
    cols[0].metric("Support vectors", len(sv))
    cols[1].metric("Training accuracy", f"{accuracy:.3f}")
    cols[2].metric("Kernel", label)
    cols[3].metric("C", f"{C:.2f}")
    cols[4].metric("Gamma", f"{gamma:.2f}" if gamma is not None else "N/A")
