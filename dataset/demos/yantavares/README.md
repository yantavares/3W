# ThreeWChart — 3W Dataset Explorer

An interactive [Streamlit](https://streamlit.io/) dashboard for browsing the
[3W dataset](../../dataset/). It replaces the old `toolkit.ThreeWChart` notebook
helper (the `toolkit/` package was removed), reproducing the same idea — a sensor
time series with the event-`class` background shading and legend — as a live web app.

## What it does

Pick, from the sidebar:

1. **Source** — real wells (`WELL-*`), `SIMULATED_*`, and/or `DRAWN_*`.
2. **Well / group** — e.g. `WELL-00001`, or a simulated/drawn id (which may appear
   across several event folders).
3. **Instance** — a single parquet file, labelled with its event class, description,
   and timestamp.
4. **Sensor** — any sensor column (defaults to `P-MON-CKP`, the original default).

The main panel shows the selected sensor over time, the time span / row metrics, and
a Plotly chart whose background is shaded by the labelled event `class` (transient
labels share the base event's hue, drawn fainter). Event names, sensor descriptions,
and the transient offset are all read from [`dataset.ini`](../../dataset/dataset.ini)
— nothing is hardcoded.

## Run it

From the repository root:

```bash
pip install -r overviews/yantavares/requirements.txt
streamlit run overviews/yantavares/streamlit_app.py
```

`streamlit` is also declared in the project's `pyproject.toml`, so an editable install
of the project (`pip install -e .`) pulls it in too.

The dataset is discovered relative to the app file (`<repo>/dataset`). If your copy
lives elsewhere, override the path in the sidebar's **Dataset folder** field.

## Notes

- The original `main.ipynb` is kept for reference but no longer runs (its
  `from toolkit import ThreeWChart` import was removed with the `toolkit/` package).
  This app supersedes it.
- The app is intentionally standalone — it does not depend on `ThreeWToolkit`.
