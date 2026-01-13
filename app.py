import sys
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import streamlit as st

from torch_geometric.data import Dataset, Data

# =========================================================
# 0) LOCAL PATHS (your absolute Windows paths)
# =========================================================
VALUES_CSV_PATH = Path(r"C:\Users\LOAN\Downloads\SP100AnalysisWithGNNs-master\data\SP100\raw\values.csv")
ADJ_NPY_PATH    = Path(r"C:\Users\LOAN\Downloads\SP100AnalysisWithGNNs-master\data\SP100\raw\adj.npy")
CKPT_PATH       = Path(r"C:\Users\LOAN\Downloads\SP100AnalysisWithGNNs-master\notebooks\models\saved_models\UpDownTrend_TGCN.pt")

# Infer project root from values.csv location: ...\data\SP100\raw\values.csv -> project root = ...\SP100AnalysisWithGNNs-master
PROJECT_ROOT = VALUES_CSV_PATH.parents[4]  # raw -> SP100 -> data -> repo_root
# Safer fallback if structure differs:
if not (PROJECT_ROOT / "notebooks").exists():
    # try walking up until we find 'notebooks'
    for p in VALUES_CSV_PATH.parents:
        if (p / "notebooks").exists():
            PROJECT_ROOT = p
            break

# Make imports (notebooks.models) work no matter where you run app.py from
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Optimal Portfolio Selection", layout="wide")


# =========================================================
# 1) Your dataset code (adapted for absolute paths)
# =========================================================
def get_graph_in_pyg_format(values_path: str, adj_path: str):
    values = pd.read_csv(values_path).set_index(["Symbol", "Date"])
    adj = np.load(adj_path)

    nodes_nb = adj.shape[0]

    # ===== x: [N, Fin, T] =====
    x = torch.tensor(
        values.drop(columns=["Close"])
              .to_numpy()
              .reshape((nodes_nb, -1, values.shape[1] - 1)),
        dtype=torch.float32,
    ).transpose(1, 2)

    # ===== close prices =====
    close_prices = torch.tensor(
        values[["Close"]].to_numpy().reshape((nodes_nb, -1)),
        dtype=torch.float32,
    )

    # ===== edge_index (and edge_weight for models that accept it) =====
    src, dst = np.where(adj != 0)
    edge_index = torch.from_numpy(np.vstack([src, dst])).long()
    edge_weight = torch.from_numpy(adj[src, dst]).float()

    return x, close_prices, edge_index, edge_weight


class SP100Stocks(Dataset):
    """
    A PyG Dataset that uses absolute raw file paths directly.
    Processed files are stored under: <root>/processed/timestep_{idx}.pt
    """

    def __init__(
        self,
        values_path: Path,
        adj_path: Path,
        past_window: int = 25,
        future_window: int = 1,
        force_reload: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.values_path = Path(values_path)
        self.adj_path = Path(adj_path)
        self.past_window = int(past_window)
        self.future_window = int(future_window)

        # root = ...\data\SP100 (parent of raw/)
        root = str(self.values_path.parents[1])  # raw -> SP100
        super().__init__(root=root, force_reload=force_reload, transform=transform)

    @property
    def raw_dir(self) -> str:
        # not strictly needed, but keep PyG conventions
        return str(self.values_path.parent)

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed")

    @property
    def raw_file_names(self) -> list[str]:
        # PyG checks existence under raw_dir; we point raw_dir to values_path.parent
        return [self.values_path.name, self.adj_path.name]

    @property
    def processed_file_names(self) -> list[str]:
        return ["timestep_0.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        if not self.values_path.exists():
            raise FileNotFoundError(f"values.csv not found: {self.values_path}")
        if not self.adj_path.exists():
            raise FileNotFoundError(f"adj.npy not found: {self.adj_path}")

        x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
            values_path=str(self.values_path),
            adj_path=str(self.adj_path),
        )

        last = x.shape[2] - self.past_window - self.future_window
        os.makedirs(self.processed_dir, exist_ok=True)

        for idx in range(max(last, 0)):
            data = Data(
                x=x[:, :, idx:idx + self.past_window],      # [N, Fin, past_window]
                edge_index=edge_index,                      # [2, E]
                edge_weight=edge_weight,                    # [E]
                close_price=close_prices[:, idx:idx + self.past_window],
                y=x[:, 0, idx + self.past_window: idx + self.past_window + self.future_window],
                close_price_y=close_prices[:, idx + self.past_window: idx + self.past_window + self.future_window],
            )
            torch.save(data, os.path.join(self.processed_dir, f"timestep_{idx}.pt"))

    def len(self) -> int:
        if not os.path.isdir(self.processed_dir):
            return 0
        return sum(
            f.startswith("timestep_") and f.endswith(".pt")
            for f in os.listdir(self.processed_dir)
        )

    def get(self, idx: int) -> Data:
        path = os.path.join(self.processed_dir, f"timestep_{idx}.pt")
        return torch.load(path, map_location="cpu", weights_only=False)


# =========================================================
# 2) Model import (notebook 9 style)
# =========================================================
def import_tgcn():
    try:
        from notebooks.models import TGCN  # type: ignore
        return TGCN
    except Exception as e:
        st.error(
            "Không import được `TGCN` từ `notebooks.models`.\n\n"
            f"PROJECT_ROOT đang dùng: {PROJECT_ROOT}\n"
            "Hãy kiểm tra trong repo có `notebooks/models.py` (hoặc package tương đương) và class `TGCN`.\n\n"
            f"Lỗi chi tiết: {e}"
        )
        st.stop()


def forward_model(model, x, edge_index, edge_weight):
    try:
        return model(x, edge_index, edge_weight)
    except TypeError:
        return model(x, edge_index)


def get_topk(scores: torch.Tensor, k: int, largest: bool):
    return torch.topk(scores, k, largest=largest).indices


def compute_metrics(curve: list[float], periods_per_year: int = 52) -> dict:
    arr = np.asarray(curve, dtype=float)
    if len(arr) < 2:
        return {}
    rets = arr[1:] / arr[:-1] - 1.0
    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (np.sqrt(periods_per_year) * mean / std) if std > 0 else np.nan
    peak = np.maximum.accumulate(arr)
    dd = arr / peak - 1.0
    mdd = float(dd.min())
    total_return = float(arr[-1] - 1.0)
    years = (len(arr) - 1) / periods_per_year
    cagr = float(arr[-1] ** (1 / years) - 1.0) if years > 0 else np.nan
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Volatility": float(std * np.sqrt(periods_per_year)),
        "Max Drawdown": mdd,
    }


@st.cache_resource(show_spinner=False)
def load_dataset_cached(weeks_ahead: int, past_window: int, force_reload: bool):
    # notebook 9 uses: future_window = weeks_ahead * 5
    return SP100Stocks(
        values_path=VALUES_CSV_PATH,
        adj_path=ADJ_NPY_PATH,
        past_window=int(past_window),
        future_window=int(weeks_ahead) * 5,
        force_reload=bool(force_reload),
    )


@st.cache_resource(show_spinner=False)
def load_model_cached(in_channels: int, hidden_size: int, layers_nb: int, ckpt_path: Path):
    TGCN = import_tgcn()
    model = TGCN(in_channels, 1, hidden_size, layers_nb)
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def try_extract_tickers(values_csv: Path) -> list[str]:
    try:
        df = pd.read_csv(values_csv)
        if "Symbol" in df.columns:
            return sorted(df["Symbol"].unique().tolist())
    except Exception:
        pass
    return []


def run_backtest(dataset, model, topks: list[int], largest: bool, train_part: float, step_days: int):
    test_data = dataset[int(len(dataset) * train_part):]
    test_data = [test_data[i] for i in range(0, len(test_data), step_days)]
    if len(test_data) < 2:
        raise RuntimeError("Test segment quá ngắn. Giảm train_part hoặc tăng dữ liệu.")

    portfolio_curves = [[1.0] for _ in topks]
    market_curve = [1.0]
    selections = {k: [] for k in topks}

    tickers = try_extract_tickers(VALUES_CSV_PATH)

    with torch.no_grad():
        out = forward_model(
            model,
            test_data[0].x,
            test_data[0].edge_index,
            getattr(test_data[0], "edge_weight", None),
        )
        scores = out.squeeze()

    last_close = test_data[0].close_price[:, -1]

    for t in range(1, len(test_data)):
        close_now = test_data[t].close_price[:, -1]
        period_returns = close_now / last_close

        for j, k in enumerate(topks):
            idxs = get_topk(scores, k, largest=largest)
            idxs_list = idxs.detach().cpu().numpy().tolist()

            if tickers and len(tickers) >= max(idxs_list) + 1:
                selections[k].append([tickers[i] for i in idxs_list])
            else:
                selections[k].append([f"Stock_{i}" for i in idxs_list])

            gross = float(period_returns[idxs].mean().item())
            portfolio_curves[j].append(gross * portfolio_curves[j][-1])

        market_curve.append(float(period_returns.mean().item()) * market_curve[-1])

        last_close = close_now
        with torch.no_grad():
            out = forward_model(
                model,
                test_data[t].x,
                test_data[t].edge_index,
                getattr(test_data[t], "edge_weight", None),
            )
            scores = out.squeeze()

    return portfolio_curves, market_curve, selections


# =========================================================
# 3) UI
# =========================================================
st.title("Optimal Portfolio Selection")

# Quick validation
problems = []
if not VALUES_CSV_PATH.exists():
    problems.append(f"Missing values.csv: {VALUES_CSV_PATH}")
if not ADJ_NPY_PATH.exists():
    problems.append(f"Missing adj.npy: {ADJ_NPY_PATH}")
if not CKPT_PATH.exists():
    problems.append(f"Missing checkpoint .pt: {CKPT_PATH}")

if problems:
    st.error("Không tìm thấy file theo path bạn cung cấp:\n\n- " + "\n- ".join(problems))
    st.stop()

with st.sidebar:
    weeks_ahead = 1
    past_window = 25
    rebuild = False

    hidden_size = 16
    layers_nb = 2

    st.header("Backtest")
    train_part = st.slider("Train proportion", min_value=0.5, max_value=0.95, value=0.9, step=0.01)
    step_days = st.selectbox("Sampling step (days)", [5, 1, 10], index=0, help="Notebook 9 dùng 5 (weekly).")
    topks_txt = "5,10,20"
    largest = False
    run_btn = st.button("Run", type="primary")

with st.spinner("Loading dataset..."):
    ds = load_dataset_cached(int(weeks_ahead), int(past_window), bool(rebuild))

# Summary
c1, c2, c3, c4 = st.columns(4)
try:
    sample = ds[0]
    c1.metric("N (stocks)", int(sample.x.size(0)))
    c2.metric("F", int(sample.x.size(1)))
    c3.metric("T", int(sample.x.size(2)))
except Exception as e:
    st.warning(f"Không đọc được ds[0] để show shape: {e}")

if run_btn:
    # in_channels = F
    try:
        in_channels = int(ds[0].x.shape[1])
    except Exception as e:
        st.error(f"Không xác định được in_channels từ ds[0].x: {e}")
        st.stop()

    with st.spinner("Loading model..."):
        model = load_model_cached(in_channels, int(hidden_size), int(layers_nb), CKPT_PATH)

    with st.spinner("Running backtest..."):
        topks = [int(x.strip()) for x in topks_txt.split(",") if x.strip()]
        curves, market_curve, selections = run_backtest(
            model=model,
            dataset=ds,
            topks=topks,
            largest=bool(largest),
            train_part=float(train_part),
            step_days=int(step_days),
        )

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, k in enumerate(topks):
        ax.plot(curves[j], label=f"Top-{k}")
    ax.plot(market_curve, label="Market", linewidth=3)
    ax.grid(which="major", linestyle="-", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_title("Market vs Portfolio (Top-K selection)")
    ax.set_xlabel("Periods")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    # Metrics
    st.subheader("Performance metrics")
    rows = [{"Strategy": "Market", **compute_metrics(market_curve, periods_per_year=52)}]
    for j, k in enumerate(topks):
        rows.append({"Strategy": f"Top-{k}", **compute_metrics(curves[j], periods_per_year=52)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Selections
    st.subheader("Selections per period")
    k_show = st.selectbox("Show selections for K", options=topks, index=0)
    sel = selections.get(k_show, [])
    if sel:
        sel_df = pd.DataFrame({"Period": np.arange(1, len(sel) + 1), "Selected": [", ".join(x) for x in sel]})
        st.dataframe(sel_df, use_container_width=True, height=400)

    # Downloads
    st.subheader("Download outputs")
    curves_df = pd.DataFrame({"Market": market_curve})
    for j, k in enumerate(topks):
        curves_df[f"Top-{k}"] = curves[j]
    st.download_button(
        "Download curves CSV",
        data=curves_df.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_curves.csv",
        mime="text/csv",
    )
