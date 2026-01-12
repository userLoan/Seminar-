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
    SP100Stocks = None
    tried_dataset = []
    for mod in ["datasets.SP100Stocks", "notebooks.datasets.SP100Stocks"]:
        try:
            m = __import__(mod, fromlist=["SP100Stocks"])
            if hasattr(m, "SP100Stocks"):
                SP100Stocks = getattr(m, "SP100Stocks")
                break
            tried_dataset.append(f"{mod}.SP100Stocks (symbol not found)")
        except Exception as e:
            tried_dataset.append(f"{mod}.SP100Stocks ({e})")

    if SP100Stocks is None:
        st.error(
            "Không import được `datasets.SP100Stocks`.\n\n"
            "Không import được `SP100Stocks`.\n\n"
            "Lý do thường gặp:\n"
            "- Bạn chưa chạy pipeline notebooks để tạo dữ liệu trong `data/SP100/raw/`.\n"
            "- Hoặc bạn chạy app không ở repo root.\n\n"
            f"Lỗi chi tiết: {e}"
            "- Bạn chưa chạy 3 notebooks để tạo dữ liệu trong `data/SP100/raw/`.\n"
            "- Hoặc bạn chạy app không ở repo root.\n"
            "- Module nằm trong `notebooks/datasets/SP100Stocks.py`.\n\n"
            "Đã thử:\n- " + "\n- ".join(tried_dataset[:6])
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

@@ -239,157 +250,174 @@ def run_backtest(dataset, model, tickers: list[str], topks: list[int], largest:
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
st.title("SP100 – Optimal Portfolio Selection")
st.markdown(
    """
Ứng dụng này triển khai lại notebook
`notebooks/9-optimal_portfolio_selection.ipynb` bằng Streamlit:

- Load dữ liệu SP100 từ `datasets.SP100Stocks`
- Load mô hình TGCN đã train
- Chọn top‑k cổ phiếu theo output của mô hình
- So sánh performance với market return
"""
)

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

st.caption("Dữ liệu đã sẵn sàng. Bên dưới là phần backtest theo notebook `9-optimal_portfolio_selection.ipynb`.")

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
        topks = sorted({int(x.strip()) for x in topks_txt.split(",") if x.strip() and int(x.strip()) > 0})
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

def plot_curves(curves, market_curve, topks, step_days):
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, k in enumerate(topks):
        ax.plot(curves[j], label=f"Top-{k}", linestyle=["--", "-.", ":"][j % 3])
    ax.plot(market_curve, label="Market", linewidth=3)
    ax.grid(which="major", linestyle="-", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("Market return vs Portfolio (top‑k)")
    ax.set_xlabel("Weeks" if step_days == 5 else "Periods")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{(x - 1) * 100:.0f}%"))
    ax.legend()
    return fig


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
    fig = plot_curves(curves, market, topks, int(step_days))
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
