import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional

class ThreeWChart:
    """
    A class to generate interactive visualizations for 3W dataset files using Plotly.

    Attributes
    ----------
    file_path : str
        Path to the Parquet file containing the dataset.
    title : str, optional
        Title of the chart (default is "ThreeW Chart").
    y_axis : str, optional
        Column name to be plotted on the y-axis (default is "P-MON-CKP").
    class_mapping : Dict[int, str]
        Dictionary mapping class IDs to their respective descriptions.
    class_colors : Dict[int, str]
        Dictionary mapping class IDs to their respective colors for visualization.

    Methods
    -------
    _load_data(filename: str) -> pd.DataFrame
        Loads and preprocesses the dataset from a Parquet file.
    _get_background_shapes(df: pd.DataFrame) -> List[Dict]
        Creates background shapes for different classes based on class transitions.
    _add_custom_legend(fig: go.Figure) -> None
        Adds a custom legend to the chart based on the class mappings.
    plot() -> None
        Generates and displays the interactive chart using Plotly.
    """

    def __init__(self, file_path: str, title: str = "ThreeW Chart", y_axis: str = "P-MON-CKP"):
        """
        Initializes the ThreeWChart class with the given parameters.

        Parameters
        ----------
        file_path : str
            Path to the Parquet file containing the dataset.
        title : str, optional
            Title of the chart (default is "ThreeW Chart").
        y_axis : str, optional
            Column name to be plotted on the y-axis (default is "P-MON-CKP").
        """
        self.file_path: Optional[str] = file_path
        self.title: str = title
        self.y_axis: str = y_axis

        self.class_mapping: Dict[int, str] = {
            0: "Normal Operation",
            1: "Abrupt Increase of BSW",
            2: "Spurious Closure of DHSV",
            3: "Severe Slugging",
            4: "Flow Instability",
            5: "Rapid Productivity Loss",
            6: "Quick Restriction in PCK",
            7: "Scaling in PCK",
            8: "Hydrate in Production Line",
            9: "Hydrate in Service Line",
            101: "Transient: Abruption Increase of BSW",
            102: "Transient: Spurious Closure of DHSV",
            105: "Transient: Rapid Productivity Loss",
            106: "Transient: Quick Restriction in PCK",
            107: "Transient: Scaling in PCK",
            108: "Transient: Hydrate in Production Line",
            109: "Transient: Hydrate in Service Line",
        }

        self.class_colors: Dict[int, str] = {
            0: "white", 1: "blue", 2: "coral", 3: "green", 4: "yellow",
            5: "brown", 6: "pink", 7: "gray", 8: "salmon", 9: "red",
            101: "cyan", 102: "lightcoral", 105: "beige", 106: "lightpink",
            107: "lightgray", 108: "lightsalmon", 109: "orange",
        }

    def _load_data(self, filename: str) -> pd.DataFrame:
        """
        Loads and preprocesses the dataset from a Parquet file.

        Parameters
        ----------
        filename : str
            Path to the Parquet file.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame with sorted timestamps and no missing values.
        """
        try:
            df = pd.read_parquet(filename)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filename}")
        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates("timestamp").fillna(0)
        return df.sort_values(by="timestamp")

    def _get_background_shapes(self, df: pd.DataFrame) -> List[Dict]:
        """
        Creates background shapes to highlight class transitions in the chart.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the class data.

        Returns
        -------
        List[Dict]
            List of shape dictionaries for Plotly.
        """
        shapes = []
        prev_class = None
        start_idx = 0

        for i in range(len(df)):
            current_class = df.iloc[i]['class']

            if pd.isna(current_class):
                print(f"Warning: NaN class value at index {i}")
                continue

            if prev_class is not None and current_class != prev_class:
                shapes.append(dict(
                    type="rect",
                    x0=df.iloc[start_idx]['timestamp'],
                    x1=df.iloc[i - 1]['timestamp'],
                    y0=0, y1=1,
                    xref='x', yref='paper',
                    fillcolor=self.class_colors.get(prev_class, "white"),
                    opacity=0.2, line_width=0
                ))
                start_idx = i

            prev_class = current_class

        if prev_class is not None:
            shapes.append(dict(
                type="rect",
                x0=df.iloc[start_idx]['timestamp'],
                x1=df.iloc[len(df) - 1]['timestamp'],
                y0=0, y1=1,
                xref='x', yref='paper',
                fillcolor=self.class_colors.get(prev_class, "white"),
                opacity=0.2, line_width=0
            ))

        return shapes

    def _add_custom_legend(self, fig: go.Figure) -> None:
        """
        Adds a custom legend to the chart based on the class mappings.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to which the legend will be added.
        """
        for class_value, event_name in self.class_mapping.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], 
                mode='markers',
                marker=dict(size=12, color=self.class_colors[class_value]),
                name=f"{class_value}: {event_name}",
                showlegend=True,
            ))

    def plot(self) -> None:
        """
        Generates and displays the interactive chart using Plotly.
        """
        df = self._load_data(self.file_path)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[self.y_axis], mode='lines', name=self.y_axis))
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            shapes=self._get_background_shapes(df),
            xaxis_title='Timestamp',
            yaxis_title=self.y_axis,
            title=self.title,
        )
        self._add_custom_legend(fig)
        fig.update_layout(legend=dict(x=1.05, y=1, title="Class Events"))
        fig.show(config={'displaylogo': False})


if __name__ == "__main__":
    chart = ThreeWChart(file_path="dataset/0/WELL-00001_20170201010207.parquet")
    chart.plot()
