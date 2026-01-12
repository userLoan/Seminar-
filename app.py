
import sys
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Repo root discovery
# ---------------------------
def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "datasets").exists() and ((p / "notebooks").exists() or (p / "models").exists()):
            return p
    return start

REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="SP100 – Optimal Portfolio Selection (TGCN)", layout="wide")

# ---------------------------
# Imports with friendly errors
# ---------------------------
def _safe_imports():
    try:
        from datasets.SP100Stocks import SP100Stocks  # type: ignore
    except Exception as e:
        st.error(
            "Không import được `datasets.SP100Stocks`.\n\n"
            "Lý do thường gặp:\n"
            "- Bạn chưa chạy pipeline notebooks để tạo dữ liệu trong `data/SP100/raw/`.\n"
            "- Hoặc bạn chạy app không ở repo root.\n\n"
            f"Lỗi chi tiết: {e}"
        )
        st.stop()

    # Try import TGCN
    TGCN = None
    tried = []
    for mod in ["notebooks.models", "models", "models.tgcn", "notebooks.models.tgcn"]:
        try:
            m = __import__(mod, fromlist=["TGCN"])
            if hasattr(m, "TGCN"):
                TGCN = getattr(m, "TGCN")
                break
            tried.append(f"{mod}.TGCN (symbol not found)")
        except Exception as e:
            tried.append(f"{mod}.TGCN ({e})")

    if TGCN is None:
        st.error(
            "Không import được `TGCN`.\n\n"
            "Hãy mở notebook `9-optimal_portfolio_selection.ipynb` để xem `TGCN` import từ đâu,\n"
            "sau đó chỉnh lại danh sách module trong app.\n\n"
            "Đã thử:\n- " + "\n- ".join(tried[:6])
        )
        st.stop()

    return SP100Stocks, TGCN


# ---------------------------
# Pipeline checks & runner
# ---------------------------
RAW_DIR = REPO_ROOT / "data" / "SP100" / "raw"
REQUIRED_FILES = {
    "stocks.csv": RAW_DIR / "stocks.csv",
    "fundamentals.csv": RAW_DIR / "fundamentals.csv",
    "values.csv": RAW_DIR / "values.csv",
    "adj.npy": RAW_DIR / "adj.npy",
}

def missing_raw_files() -> list[str]:
    missing = []
    for name, p in REQUIRED_FILES.items():
        if not p.exists():
            missing.append(str(p))
    return missing

def find_notebook(patterns: list[str]) -> Path | None:
    nb_dir = REPO_ROOT / "notebooks"
    if not nb_dir.exists():
        return None
    cands = list(nb_dir.glob("*.ipynb"))
    # prioritize exact prefix matches (e.g., "1-", "2-", "3-")
    for pref in patterns:
        for p in cands:
            if p.name.startswith(pref):
                return p
    # fallback: keyword match
    for kw in patterns:
        kw_low = kw.lower()
        for p in cands:
            if kw_low in p.name.lower():
                return p
    return None

def run_nb(nb_path: Path) -> tuple[int, str]:
    """
    Execute a notebook in-place using nbconvert.
    Returns (returncode, combined_output).
    """
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=0",
        str(nb_path),
        "--output", nb_path.name.replace(".ipynb", ".executed.ipynb"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out

def run_pipeline():
    """
    Run notebooks 1 -> 2 -> 3 sequentially.
    This matches the dependency chain:
    1 creates raw csv; 2 creates adj.npy; 3 validates/creates dataset objects.
    """
    nb1 = find_notebook(["1-", "data_collection", "preprocessing"])
    nb2 = find_notebook(["2-", "graph_creation", "graph"])
    nb3 = find_notebook(["3-", "torch_geometric_dataset", "dataset"])

    if nb1 is None or nb2 is None or nb3 is None:
        st.error(
            "Không tìm thấy đủ 3 notebooks trong thư mục `notebooks/`.\n\n"
            "Yêu cầu:\n"
            "- 1-data_collection_and_preprocessing.ipynb\n"
            "- 2-graph_creation.ipynb\n"
            "- 3-torch_geometric_dataset.ipynb\n\n"
            "Hãy kiểm tra bạn đang chạy app ở repo root và các notebook nằm đúng vị trí."
        )
        st.stop()

    logs = []
    for nb in [nb1, nb2, nb3]:
        code, out = run_nb(nb)
        logs.append((nb.name, code, out))
        if code != 0:
            return False, logs
    return True, logs


# ---------------------------
# Portfolio logic (same as notebook 9)
# ---------------------------
def get_topk(model_out, k: int, largest: bool = True):
    import torch
    return torch.topk(model_out, k, largest=largest).indices

def compute_performance(cum_curve: list[float], periods_per_year: int = 52) -> dict:
    arr = np.asarray(cum_curve, dtype=float)
    if len(arr) < 2:
        return {}
    rets = arr[1:] / arr[:-1] - 1.0
    mean = rets.mean()
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = (np.sqrt(periods_per_year) * mean / std) if std > 0 else np.nan
    peak = np.maximum.accumulate(arr)
    dd = arr / peak - 1.0
    mdd = dd.min()
    total_return = arr[-1] - 1.0
    years = (len(arr) - 1) / periods_per_year
    cagr = (arr[-1] ** (1 / years) - 1.0) if years > 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Volatility": std * np.sqrt(periods_per_year),
        "Max Drawdown": mdd,
        "Periods": len(arr) - 1,
    }

@st.cache_resource(show_spinner=False)
def load_dataset_cached(weeks_ahead: int):
    SP100Stocks, _ = _safe_imports()
    return SP100Stocks(future_window=weeks_ahead * 5)

@st.cache_resource(show_spinner=False)
def load_model_cached(in_channels: int, out_channels: int, hidden_size: int, layers_nb: int, ckpt_bytes: bytes | None, ckpt_path: str | None):
    import torch
    _, TGCN = _safe_imports()
    model = TGCN(in_channels, out_channels, hidden_size, layers_nb)

    if ckpt_bytes is not None:
        tmp = Path(".streamlit_tmp_ckpt.pt")
        tmp.write_bytes(ckpt_bytes)
        state = torch.load(tmp, map_location="cpu")
    elif ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
    else:
        raise ValueError("Missing checkpoint.")

    model.load_state_dict(state)
    model.eval()
    return model

def find_checkpoints(repo_root: Path) -> list[str]:
    dirs = [
        repo_root / "notebooks" / "models" / "saved_models",
        repo_root / "models" / "saved_models",
    ]
    ckpts = []
    for d in dirs:
        if d.exists():
            ckpts.extend([str(p) for p in d.glob("*.pt")])
    return sorted(set(ckpts))

def try_get_tickers(dataset) -> list[str]:
    for attr in ["symbols", "tickers", "tickers_list", "stock_symbols", "stocks"]:
        if hasattr(dataset, attr):
            v = getattr(dataset, attr)
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
                return list(v)
    return []

def run_backtest(dataset, model, tickers: list[str], topks: list[int], largest: bool, train_part: float, step_days: int = 5):
    import torch
    test_data = dataset[int(len(dataset) * train_part):]
    test_data = [test_data[idx] for idx in range(0, len(test_data), step_days)]
    if len(test_data) < 2:
        raise RuntimeError("Test segment quá ngắn. Hãy tăng dữ liệu hoặc giảm train_part.")

    portfolio_curves = [[1.0] for _ in topks]
    market_curve = [1.0]
    selections = {k: [] for k in topks}

    with torch.no_grad():
        model_out = model(test_data[0].x, test_data[0].edge_index, test_data[0].edge_weight).squeeze(1)
    last_close = test_data[0].close_price[:, -1]

    for t in range(1, len(test_data)):
        close_now = test_data[t].close_price[:, -1]
        period_returns = close_now / last_close

        for j, k in enumerate(topks):
            idxs = get_topk(model_out, k, largest=largest)
            idxs_list = idxs.detach().cpu().numpy().tolist()
            if tickers:
                selections[k].append([tickers[i] if i < len(tickers) else f"Stock_{i}" for i in idxs_list])
            else:
                selections[k].append([f"Stock_{i}" for i in idxs_list])

            gross = period_returns[idxs].mean().item()
            portfolio_curves[j].append(gross * portfolio_curves[j][-1])

        market_curve.append(period_returns.mean().item() * market_curve[-1])

        last_close = close_now
        with torch.no_grad():
            model_out = model(test_data[t].x, test_data[t].edge_index, test_data[t].edge_weight).squeeze(1)

    return portfolio_curves, market_curve, selections


# ---------------------------
# UI
# ---------------------------
st.title("SP100 – Optimal Portfolio Selection (Streamlit)")

missing = missing_raw_files()
if missing:
    st.warning(
        "Bạn đang thiếu dữ liệu trong `data/SP100/raw/` nên `datasets.SP100Stocks` sẽ không chạy được.\n\n"
        "Về bản chất, 3 notebooks bạn nói phải chạy là do chúng tạo ra các file sau:\n"
        "- stocks.csv\n- fundamentals.csv\n- values.csv\n- adj.npy\n\n"
        "Bạn chỉ cần chạy chúng *một lần theo thứ tự* (1 → 2 → 3). Không cần chạy 'cùng lúc'."
    )
    st.code("\n".join(missing), language="text")

    st.subheader("Chạy pipeline ngay trong app (tùy chọn)")
    st.caption("Cách này dùng `jupyter nbconvert --execute`. Bạn cần cài jupyter: `pip install jupyter`.")
    if st.button("Run notebooks 1 → 2 → 3", type="primary"):
        with st.spinner("Executing notebooks... (có thể mất thời gian do tải dữ liệu yfinance)"):
            ok, logs = run_pipeline()
        if ok:
            st.success("Pipeline chạy xong. Bạn có thể rerun app và chạy backtest.")
        else:
            st.error("Pipeline lỗi ở một notebook. Xem log dưới đây.")
        for name, code, out in logs:
            with st.expander(f"Log: {name} (returncode={code})", expanded=False):
                st.text(out[:20000])  # cap to keep UI responsive
    st.stop()

with st.sidebar:
    st.header("Backtest config")
    weeks_ahead = st.number_input("Weeks ahead (horizon)", min_value=1, max_value=12, value=1, step=1)
    train_part = st.slider("Train proportion", min_value=0.5, max_value=0.95, value=0.9, step=0.01)
    step_days = st.selectbox("Sampling", options=[5, 1, 10], index=0, help="5=weekly giống notebook")
    largest = st.selectbox(
        "Top‑K direction",
        options=[("smallest (largest=False, giống notebook)", False), ("largest", True)],
        format_func=lambda x: x[0]
    )[1]
    topks_txt = st.text_input("Top‑K list", value="5,10,20")
    try:
        topks = sorted({int(x.strip()) for x in topks_txt.split(",") if x.strip()})
        if not topks:
            raise ValueError()
    except Exception:
        topks = [5, 10, 20]

    hidden_size = st.number_input("hidden_size", min_value=4, max_value=256, value=16, step=4)
    layers_nb = st.number_input("layers_nb", min_value=1, max_value=8, value=2, step=1)

    st.subheader("Checkpoint")
    ckpts = find_checkpoints(REPO_ROOT)
    ckpt_mode = st.radio("Source", ["Auto-detect in repo", "Upload .pt"], index=0)

    ckpt_path = None
    ckpt_bytes = None
    if ckpt_mode == "Auto-detect in repo":
        if ckpts:
            default_idx = 0
            for i, p in enumerate(ckpts):
                if "UpDownTrend_TGCN.pt" in p.replace("\\", "/"):
                    default_idx = i
                    break
            ckpt_path = st.selectbox("Checkpoint (.pt)", options=ckpts, index=default_idx)
        else:
            st.info("Không tìm thấy .pt trong repo. Chọn Upload.")
            ckpt_mode = "Upload .pt"

    if ckpt_mode == "Upload .pt":
        up = st.file_uploader("Upload checkpoint", type=["pt"])
        if up is not None:
            ckpt_bytes = up.read()

    run_btn = st.button("Run backtest", type="primary")

SP100Stocks, TGCN = _safe_imports()

with st.spinner("Loading dataset..."):
    dataset = load_dataset_cached(int(weeks_ahead))
tickers = try_get_tickers(dataset)

c1, c2, c3, c4 = st.columns(4)
try:
    sample = dataset[0]
    c1.metric("Nodes", int(sample.x.size(0)))
    c2.metric("Features", int(sample.x.size(-2)))
    c3.metric("Past window", int(sample.x.size(-1)))
    c4.metric("Len (days)", int(len(dataset)))
except Exception:
    st.info("Không đọc được dataset shape (schema có thể khác).")

if run_btn:
    if ckpt_path is None and ckpt_bytes is None:
        st.error("Chọn hoặc upload checkpoint trước khi chạy.")
        st.stop()

    try:
        in_channels = int(dataset[0].x.shape[-2])
    except Exception as e:
        st.error(f"Không xác định được in_channels: {e}")
        st.stop()

    with st.spinner("Loading model..."):
        model = load_model_cached(in_channels, 1, int(hidden_size), int(layers_nb), ckpt_bytes, ckpt_path)

    with st.spinner("Running backtest..."):
        curves, market, selections = run_backtest(dataset, model, tickers, topks, largest, float(train_part), int(step_days))

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, k in enumerate(topks):
        ax.plot(curves[j], label=f"Top-{k}")
    ax.plot(market, label="Market", linewidth=3)
    ax.grid(which="major", linestyle="-", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("Market vs Portfolio")
    ax.set_xlabel("Periods")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Metrics")
    rows = [{"Strategy": "Market", **compute_performance(market)}]
    for j, k in enumerate(topks):
        rows.append({"Strategy": f"Top-{k}", **compute_performance(curves[j])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Selections")
    k_show = st.selectbox("Choose K", options=topks, index=0)
    sel = selections.get(k_show, [])
    if sel:
        df = pd.DataFrame({"Period": np.arange(1, len(sel) + 1), "Selected": [", ".join(x) for x in sel]})
        st.dataframe(df, use_container_width=True, height=400)
