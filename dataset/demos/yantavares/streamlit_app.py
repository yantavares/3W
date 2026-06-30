"""ThreeWChart — 3W Dataset Explorer.

A self-contained Streamlit dashboard that replaces the old (now-removed)
``toolkit.ThreeWChart`` helper. Pick a well/group, then an instance file, then a
sensor, and view that sensor's time series with the event-class background shading
and a legend — a live web app version of the original notebook demo.

Run from the repo root::

    streamlit run overviews/yantavares/streamlit_app.py

The dataset is discovered relative to this file (``<repo>/dataset``); a sidebar
override is available if your copy lives elsewhere.
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset"
DEFAULT_SENSOR = "P-MON-CKP"  # the original ThreeWChart default y-axis

# A pleasant qualitative palette for the 10 base event classes (0-9). Transient
# variants (label + TRANSIENT_OFFSET) reuse the same hue, drawn fainter.
BASE_CLASS_COLORS: dict[int, str] = {
    0: "#2E7D32",  # normal     – green
    1: "#1E88E5",  # blue
    2: "#E53935",  # red
    3: "#8E24AA",  # purple
    4: "#FB8C00",  # orange
    5: "#00897B",  # teal
    6: "#D81B60",  # pink
    7: "#6D4C41",  # brown
    8: "#3949AB",  # indigo
    9: "#00ACC1",  # cyan
}
FALLBACK_COLOR = "#90A4AE"  # blue-grey for unknown labels
LINE_COLOR = "#263238"      # dark slate for the sensor trace


@dataclass(frozen=True)
class DatasetConfig:
    event_descriptions: dict[int, str]
    sensor_descriptions: dict[str, str]
    transient_offset: int


# --------------------------------------------------------------------------- #
# Config parsing (reuse dataset.ini — no hardcoded labels)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_dataset_config(ini_path: str) -> DatasetConfig:
    """Parse ``dataset.ini`` into description maps and the transient offset."""
    parser = configparser.ConfigParser()
    # Keep option keys case-sensitive so sensor names like ``P-MON-CKP`` survive.
    parser.optionxform = str  # type: ignore[assignment]
    parser.read(ini_path, encoding="utf-8")

    sensor_descriptions: dict[str, str] = {}
    if parser.has_section("PARQUET_FILE_PROPERTIES"):
        for key, value in parser.items("PARQUET_FILE_PROPERTIES"):
            sensor_descriptions[key] = value.strip().replace("%%", "%")

    transient_offset = 100
    if parser.has_option("EVENTS", "TRANSIENT_OFFSET"):
        transient_offset = parser.getint("EVENTS", "TRANSIENT_OFFSET")

    event_descriptions: dict[int, str] = {}
    for section in parser.sections():
        if parser.has_option(section, "LABEL") and parser.has_option(
            section, "DESCRIPTION"
        ):
            try:
                label = int(parser.get(section, "LABEL"))
            except ValueError:
                continue
            event_descriptions[label] = parser.get(section, "DESCRIPTION").strip()

    return DatasetConfig(event_descriptions, sensor_descriptions, transient_offset)


def class_description(label, cfg: DatasetConfig) -> str:
    """Human-readable description for a (possibly transient/NaN) class label."""
    if label is None or pd.isna(label):
        return "Unlabeled"
    label = int(label)
    if label >= cfg.transient_offset:
        base = label - cfg.transient_offset
        base_desc = cfg.event_descriptions.get(base, str(base))
        return f"Transient: {base_desc}"
    return cfg.event_descriptions.get(label, str(label))


def class_color(label, cfg: DatasetConfig) -> tuple[str, bool]:
    """Return (hex color, is_transient) for a class label; shares hue with base."""
    if label is None or pd.isna(label):
        return FALLBACK_COLOR, False
    label = int(label)
    transient = label >= cfg.transient_offset
    base = label - cfg.transient_offset if transient else label
    return BASE_CLASS_COLORS.get(base, FALLBACK_COLOR), transient


def sensor_unit(description: str | None) -> str:
    """Extract a trailing unit like ``[Pa]`` from a sensor description."""
    if not description:
        return ""
    match = re.search(r"\[([^\]]+)\]\s*$", description)
    return match.group(1) if match else ""


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

_FILENAME_RE = re.compile(
    r"^(?P<source>WELL|SIMULATED|DRAWN)[-_](?P<ident>\d+)(?:_(?P<ts>\d{14}))?$"
)


@st.cache_data(show_spinner="Scanning dataset…")
def discover_instances(dataset_root: str) -> list[dict]:
    """Walk ``<root>/<class>/<*.parquet>`` into a list of instance records."""
    root = Path(dataset_root)
    records: list[dict] = []
    for path in sorted(root.glob("*/*.parquet")):
        match = _FILENAME_RE.match(path.stem)
        if match is None:
            continue
        source = match.group("source")
        ident = match.group("ident")
        ts = match.group("ts")
        timestamp_str = ""
        if ts:
            try:
                timestamp_str = pd.to_datetime(ts, format="%Y%m%d%H%M%S").strftime(
                    "%Y-%m-%d %H:%M"
                )
            except ValueError:
                timestamp_str = ts
        records.append(
            {
                "path": str(path),
                "source": source,
                "group_id": f"{source}-{ident}" if source == "WELL"
                else f"{source}_{ident}",
                "ident": ident,
                "class_folder": path.parent.name,
                "timestamp_str": timestamp_str,
                "filename": path.name,
            }
        )
    return records


@st.cache_data(show_spinner="Loading instance…")
def load_instance(path: str) -> pd.DataFrame:
    """Load and tidy a single parquet instance (datetime index, sorted, unique)."""
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _class_runs(df: pd.DataFrame) -> list[tuple]:
    """Contiguous runs of equal ``class``: (x0, x1, label) with NaN as sentinel."""
    if "class" not in df.columns or df.empty:
        return []
    codes = df["class"].fillna(-1).astype("int64").to_numpy()
    times = df.index.to_numpy()
    change = np.empty(len(codes), dtype=bool)
    change[0] = True
    change[1:] = codes[1:] != codes[:-1]
    starts = np.flatnonzero(change)
    runs: list[tuple] = []
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(codes)
        code = int(codes[s])
        # Extend the rect to the next run's start so shading is gap-free.
        x1 = times[end] if end < len(times) else times[-1]
        label = None if code == -1 else code
        runs.append((times[s], x1, label))
    return runs


def build_figure(
    df: pd.DataFrame, sensor: str, cfg: DatasetConfig, title: str
) -> go.Figure:
    """Plotly line of ``sensor`` over time with event-class background shading."""
    fig = go.Figure()

    # Background shading: one rect per contiguous class run.
    runs = _class_runs(df)
    for x0, x1, label in runs:
        if label is None:
            continue
        color, transient = class_color(label, cfg)
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=0,
            y1=1,
            fillcolor=color,
            opacity=0.10 if transient else 0.18,
            line_width=0,
            layer="below",
        )

    # Sensor trace.
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[sensor],
            mode="lines",
            name=sensor,
            line=dict(color=LINE_COLOR, width=1.4),
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>%{y:.4g}<extra></extra>",
            showlegend=False,
        )
    )

    # Custom legend: one invisible marker per class present in the file.
    present = sorted(
        {int(x) for x in df["class"].dropna().unique()}
    ) if "class" in df.columns else []
    for label in present:
        color, _ = class_color(label, cfg)
        fig.add_trace(
            go.Scatter(
                x=[df.index[0]],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square", opacity=0.6),
                name=f"{label} — {class_description(label, cfg)}",
                hoverinfo="skip",
            )
        )

    unit = sensor_unit(cfg.sensor_descriptions.get(sensor))
    y_title = f"{sensor} [{unit}]" if unit else sensor
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=560,
        margin=dict(l=60, r=30, t=70, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title="Event classes",
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="Timestamp", rangeslider_visible=True)
    return fig


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="ThreeWChart — 3W Dataset Explorer",
        page_icon="🛢️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        h1 { font-weight: 700; }
        [data-testid="stMetricValue"] { font-size: 1.15rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛢️ ThreeWChart — 3W Dataset Explorer")
    st.caption(
        "Browse the 3W dataset: pick a well, an instance, and a sensor to see its "
        "time series with event-class background shading."
    )

    # --- Sidebar: data source -------------------------------------------------
    st.sidebar.header("Data source")
    dataset_root = st.sidebar.text_input(
        "Dataset folder", value=str(DEFAULT_DATASET_ROOT)
    )
    root_path = Path(dataset_root)
    if not root_path.exists():
        st.error(f"Dataset folder not found: `{dataset_root}`")
        st.stop()

    ini_path = root_path / "dataset.ini"
    if not ini_path.exists():
        st.error(f"`dataset.ini` not found in `{dataset_root}`")
        st.stop()
    cfg = load_dataset_config(str(ini_path))

    records = discover_instances(str(root_path))
    if not records:
        st.error("No parquet instances found under the dataset folder.")
        st.stop()

    # --- Sidebar: selection flow ---------------------------------------------
    st.sidebar.header("Selection")
    all_sources = sorted({r["source"] for r in records})
    sources = st.sidebar.multiselect(
        "Source", options=all_sources, default=all_sources
    )
    if not sources:
        st.info("Select at least one source in the sidebar to begin.")
        st.stop()

    in_source = [r for r in records if r["source"] in sources]
    groups = sorted({r["group_id"] for r in in_source})
    group = st.sidebar.selectbox("Well / group", options=groups)

    instances = [r for r in in_source if r["group_id"] == group]
    instances.sort(key=lambda r: (r["class_folder"], r["timestamp_str"], r["filename"]))

    def _instance_label(r: dict) -> str:
        try:
            desc = cfg.event_descriptions.get(int(r["class_folder"]), r["class_folder"])
        except ValueError:
            desc = r["class_folder"]
        suffix = r["timestamp_str"] or r["filename"]
        return f"{r['class_folder']} · {desc} — {suffix}"

    chosen = st.sidebar.selectbox(
        "Instance", options=instances, format_func=_instance_label
    )

    df = load_instance(chosen["path"])
    if df.empty:
        st.warning("This instance has no valid timestamped rows.")
        st.stop()

    # Sensor list: numeric columns minus class/state, dropping all-NaN columns.
    excluded = {"class", "state"}
    sensor_cols = [
        c
        for c in df.columns
        if c not in excluded and not df[c].isna().all()
    ]
    if not sensor_cols:
        st.warning("No plottable sensor columns in this instance.")
        st.stop()

    default_index = (
        sensor_cols.index(DEFAULT_SENSOR) if DEFAULT_SENSOR in sensor_cols else 0
    )
    sensor = st.sidebar.selectbox(
        "Sensor",
        options=sensor_cols,
        index=default_index,
        help=cfg.sensor_descriptions.get(DEFAULT_SENSOR, ""),
    )
    sensor_desc = cfg.sensor_descriptions.get(sensor)
    if sensor_desc:
        st.sidebar.caption(sensor_desc)

    # --- Header metrics -------------------------------------------------------
    span = df.index.max() - df.index.min()
    total_seconds = int(span.total_seconds())
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    unit = sensor_unit(sensor_desc)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instance", chosen["filename"])
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Time span", f"{hours}h {minutes}m")
    c4.metric("Sensor unit", unit or "—")

    # --- Plot -----------------------------------------------------------------
    title = f"{sensor} — {chosen['group_id']} (event {chosen['class_folder']})"
    fig = build_figure(df, sensor, cfg, title)
    st.plotly_chart(fig, use_container_width=True)

    # --- Raw preview ----------------------------------------------------------
    with st.expander("Preview raw data (first rows)"):
        st.dataframe(df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
