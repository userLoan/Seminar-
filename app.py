
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# ---------------------------
# Repo root discovery & imports
# ---------------------------
def find_repo_root(start: Path) -> Path:
    """
    Find repo root by looking for 'datasets/' plus either 'notebooks/' or 'models/'.
    You should run this app from the repo root (recommended).
    """
    for p in [start] + list(start.parents):
        if (p / "datasets").exists() and ((p / "notebooks").exists() or (p / "models").exists()):
            return p
    return start

REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="SP100 – Optimal Portfolio Selection (TGCN)", layout="wide")


# ---------------------------
# Helper functions
# ---------------------------
def _safe_imports():
    """Import project modules with helpful, user-facing errors."""
    try:
        from datasets.SP100Stocks import SP100Stocks  # type: ignore
    except Exception as e:
        st.error(
            "Không import được `datasets.SP100Stocks`.\n\n"
            "Cách sửa nhanh:\n"
            "1) Đảm bảo bạn đang chạy `streamlit run app.py` ở *repo root* (thư mục có `datasets/`).\n"
            "2) Kiểm tra bạn đã cài đủ dependencies.\n\n"
            f"Chi tiết lỗi: {e}"
        )
        st.stop()

    # Try several module paths for TGCN (repo/fork có thể khác nhau)
    TGCN = None
    tried = []
    candidates = [
        ("notebooks.models", "TGCN"),
        ("models", "TGCN"),
        ("models.tgcn", "TGCN"),
        ("notebooks.model", "TGCN"),
        ("notebooks.models.tgcn", "TGCN"),
    ]
    for mod, sym in candidates:
        try:
            m = __import__(mod, fromlist=[sym])
            if hasattr(m, sym):
                TGCN = getattr(m, sym)
                break
            tried.append(f"{mod}.{sym} (symbol not found)")
        except Exception as e:
            tried.append(f"{mod}.{sym} ({e})")

    if TGCN is None:
        st.error(
            "Không import được class `TGCN`.\n\n"
            "Cách sửa:\n"
            "- Mở notebook của bạn và xem dòng import `TGCN` nằm ở module nào, rồi chỉnh trong hàm `_safe_imports()`.\n\n"
            "Đã thử các đường dẫn:\n- " + "\n- ".join(tried[:6])
        )
        st.stop()

    return SP100Stocks, TGCN


def _try_get_tickers(dataset) -> list[str]:
    """Best-effort extraction of ticker symbols from dataset."""
    for attr in ["symbols", "tickers", "tickers_list", "stock_symbols", "stocks"]:
        if hasattr(dataset, attr):
            v = getattr(dataset, attr)
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                return list(v)

    # fallback: numeric labels
    try:
        n = int(dataset[0].x.size(0))
        return [f"Stock_{i}" for i in range(n)]
    except Exception:
        return []


def _find_checkpoints(repo_root: Path) -> list[str]:
    """
    Look for .pt checkpoints in common locations.
    Your path indicates: notebooks/models/saved_models/UpDownTrend_TGCN.pt
    """
    candidates = [
        repo_root / "models" / "saved_models",
        repo_root / "notebooks" / "models" / "saved_models",
        repo_root / "notebooks" / "models" / "saved_models",  # same but kept for clarity
    ]
    ckpts = []
    for d in candidates:
        if d.exists():
            ckpts.extend([str(p) for p in d.glob("*.pt")])

    # If still none, do a bounded recursive search (some forks rename folders)
    if not ckpts:
        # Avoid scanning huge trees: skip common virtualenv/cache folders
        skip = {"venv", ".venv", "__pycache__", ".git", ".idea", ".mypy_cache", ".pytest_cache"}
        for p in repo_root.rglob("*.pt"):
            if any(part in skip for part in p.parts):
                continue
            if "saved_models" in p.parts or "checkpoints" in p.parts:
                ckpts.append(str(p))

    ckpts = sorted(set(ckpts))
    return ckpts


def get_topk(model_out, k: int, largest: bool = True):
    import torch
    return torch.topk(model_out, k, largest=largest).indices


def compute_performance(cum_curve: list[float], periods_per_year: int = 52) -> dict:
    """Compute simple performance metrics from cumulative curve."""
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
    # Notebook dùng future_window = weeks_ahead * 5
    ds = SP100Stocks(future_window=weeks_ahead * 5)
    return ds


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


def run_backtest(dataset, model, tickers: list[str], topks: list[int], largest: bool, train_part: float, step_days: int = 5):
    """Replicate notebook logic: sampling and top-k selection based on model output."""
    import torch

    test_data = dataset[int(len(dataset) * train_part):]               # daily
    test_data = [test_data[idx] for idx in range(0, len(test_data), step_days)]  # sample

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
        period_returns = close_now / last_close  # gross return per stock

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
st.caption("Dùng checkpoint `UpDownTrend_TGCN.pt` để chọn Top‑K cổ phiếu và so sánh cumulative return với market.")

with st.sidebar:
    st.header("Cấu hình")
    weeks_ahead = st.number_input("Weeks ahead (horizon)", min_value=1, max_value=12, value=1, step=1)
    train_part = st.slider("Train proportion (split)", min_value=0.5, max_value=0.95, value=0.9, step=0.01)
    step_days = st.selectbox(
        "Sampling frequency",
        options=[5, 1, 10],
        index=0,
        help="5 = weekly (giống notebook). 1 = daily. 10 = 2 weeks."
    )

    largest = st.selectbox(
        "Chọn Top‑K theo score",
        options=[
            ("smallest (giống notebook: largest=False)", False),
            ("largest", True),
        ],
        format_func=lambda x: x[0]
    )[1]

    topks_txt = st.text_input("Top‑K list", value="5,10,20", help="Nhập danh sách K, phân tách bằng dấu phẩy.")
    try:
        topks = sorted({int(x.strip()) for x in topks_txt.split(",") if x.strip()})
        if not topks:
            raise ValueError()
    except Exception:
        st.warning("Top‑K list không hợp lệ. Dùng mặc định: 5,10,20.")
        topks = [5, 10, 20]

    hidden_size = st.number_input("TGCN hidden_size", min_value=4, max_value=256, value=16, step=4)
    layers_nb = st.number_input("TGCN layers_nb", min_value=1, max_value=8, value=2, step=1)

    st.subheader("Model checkpoint (.pt)")
    ckpt_candidates = _find_checkpoints(REPO_ROOT)

    ckpt_mode = st.radio("Nguồn checkpoint", options=["Chọn từ repo", "Upload .pt"], index=0)

    ckpt_path = None
    ckpt_bytes = None

    if ckpt_mode == "Chọn từ repo":
        if ckpt_candidates:
            # try to preselect UpDownTrend_TGCN.pt
            default_idx = 0
            for i, p in enumerate(ckpt_candidates):
                if "UpDownTrend_TGCN.pt" in p.replace("\\", "/"):
                    default_idx = i
                    break
            ckpt_path = st.selectbox("Checkpoint (.pt)", options=ckpt_candidates, index=default_idx)
            st.caption("Gợi ý: đặt file vào `notebooks/models/saved_models/UpDownTrend_TGCN.pt` để auto-detect.")
        else:
            st.info("Không tìm thấy file .pt trong repo. Hãy chọn Upload.")
            ckpt_mode = "Upload .pt"

    if ckpt_mode == "Upload .pt":
        up = st.file_uploader("Upload checkpoint", type=["pt"])
        if up is not None:
            ckpt_bytes = up.read()

    run_btn = st.button("Run backtest", type="primary")


# ---------------------------
# Main execution
# ---------------------------
SP100Stocks, TGCN = _safe_imports()

with st.spinner("Loading dataset..."):
    dataset = load_dataset_cached(int(weeks_ahead))

tickers = _try_get_tickers(dataset)

# Quick summary
c1, c2, c3, c4 = st.columns(4)
try:
    sample = dataset[0]
    c1.metric("Nodes (stocks)", int(sample.x.size(0)))
    c2.metric("Features", int(sample.x.size(-2)))
    c3.metric("Past window", int(sample.x.size(-1)))
    c4.metric("Dataset length (days)", int(len(dataset)))
except Exception:
    st.info("Không đọc được shape từ dataset[0].x; repo/fork của bạn có thể khác schema.")

st.divider()

if run_btn:
    if (ckpt_path is None) and (ckpt_bytes is None):
        st.error("Bạn cần chọn hoặc upload checkpoint (.pt) trước khi chạy.")
        st.stop()

    try:
        in_channels = int(dataset[0].x.shape[-2])
    except Exception as e:
        st.error(f"Không xác định được in_channels từ dataset[0].x: {e}")
        st.stop()

    out_channels = 1

    with st.spinner("Loading model..."):
        model = load_model_cached(in_channels, out_channels, int(hidden_size), int(layers_nb), ckpt_bytes, ckpt_path)

    with st.spinner("Running backtest..."):
        try:
            portfolio_curves, market_curve, selections = run_backtest(
                dataset=dataset,
                model=model,
                tickers=tickers,
                topks=topks,
                largest=largest,
                train_part=float(train_part),
                step_days=int(step_days),
            )
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, k in enumerate(topks):
        ax.plot(portfolio_curves[j], label=f"Top-{k}")
    ax.plot(market_curve, label="Market", linewidth=3)
    ax.grid(which="major", linestyle="-", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("Market vs Portfolio (Top‑K selection)")
    ax.set_xlabel("Periods")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Performance metrics (annualization = 52 when weekly)")
    rows = [{"Strategy": "Market", **compute_performance(market_curve, periods_per_year=52)}]
    for j, k in enumerate(topks):
        rows.append({"Strategy": f"Top-{k}", **compute_performance(portfolio_curves[j], periods_per_year=52)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Selected tickers per period")
    k_show = st.selectbox("Chọn K để xem danh sách", options=topks, index=0)
    sel = selections.get(k_show, [])
    if sel:
        sel_df = pd.DataFrame({
            "Period": np.arange(1, len(sel) + 1),
            "Selected": [", ".join(x) for x in sel],
        })
        st.dataframe(sel_df, use_container_width=True, height=400)
    else:
        st.info("Chưa có selections để hiển thị (test segment quá ngắn hoặc sampling quá thưa).")

else:
    st.info("Chọn cấu hình ở sidebar và bấm 'Run backtest'.")
