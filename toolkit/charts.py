import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional
import ipywidgets as widgets
from IPython.display import display

class ThreeWChart:
    """
    A class to generate interactive visualizations for 3W dataset files using Plotly.
    Optionally, it allows the selection of a column to plot via a dropdown if enabled.

    Attributes
    ----------
    file_path : str
        Path to the Parquet file containing the dataset.
    title : str, optional
        Title of the chart (default is "ThreeW Chart").
    y_axis : str, optional
        Default column to plot on the y-axis (default is "P-MON-CKP").
    use_dropdown : bool, optional
        Whether to enable the dropdown for column selection (default is False).
    class_mapping : Dict[int, str]
        Dictionary mapping class IDs to their respective descriptions.
    class_colors : Dict[int, str]
        Dictionary mapping class IDs to their respective colors for visualization.

    Methods
    -------
    _load_data(filename: str) -> pd.DataFrame
        Loads and preprocesses the dataset from a Parquet file.
    _get_valid_cols(df: pd.DataFrame) -> List[str]
        Retrieves numeric columns that contain at least one non-zero value.
    _create_dropdown(options: List[str]) -> widgets.Dropdown
        Creates a dropdown widget for selecting columns to plot.
    _update_chart(change: dict) -> None
        Updates the chart when a new column is selected from the dropdown.
    _get_background_shapes(df: pd.DataFrame) -> List[Dict]
        Creates background shapes for different classes based on class transitions.
    _add_custom_legend(fig: go.Figure) -> None
        Adds a custom legend to the chart based on the class mappings.
    plot() -> None
        Generates and displays the interactive chart using Plotly.
    """

    def __init__(self, file_path: str, title: str = "ThreeW Chart", y_axis: str = "P-MON-CKP", use_dropdown: bool = False):
        """
        Initializes the ThreeWChart class with the given parameters.

        Parameters
        ----------
        file_path : str
            Path to the Parquet file containing the dataset.
        title : str, optional
            Title of the chart (default is "ThreeW Chart").
        y_axis : str, optional
            Default column name to be plotted on the y-axis (default is "P-MON-CKP").
        use_dropdown : bool, optional
            Whether to enable the dropdown for column selection (default is False).
        """
        self.file_path: Optional[str] = file_path
        self.title: str = title
        self.y_axis: str = y_axis
        self.use_dropdown: bool = use_dropdown
        self.df: Optional[pd.DataFrame] = None  # To store the loaded DataFrame

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
        self.df = df.sort_values(by="timestamp")
        return self.df

    def _get_valid_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Retrieves numeric columns that contain at least one non-zero value.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.

        Returns
        -------
        List[str]
            List of numeric column names with non-zero values.
        """
        return [col for col in df.select_dtypes(include=[float, int]).columns if df[col].any()]

    def _create_dropdown(self, options: List[str]) -> widgets.Dropdown:
        """
        Creates a dropdown widget for selecting columns to plot.

        Parameters
        ----------
        options : List[str]
            List of column names to include in the dropdown.

        Returns
        -------
        widgets.Dropdown
            Dropdown widget for selecting columns.
        """
        dropdown = widgets.Dropdown(
            options=options,
            value=self.y_axis,
            description='Select Column:',
            disabled=False,
        )
        dropdown.observe(self._update_chart, names='value')
        return dropdown

    def _update_chart(self, change: dict) -> None:
        """
        Updates the chart when a new column is selected from the dropdown.

        Parameters
        ----------
        change : dict
            Dictionary containing information about the dropdown change event.
        """
        new_column = change['new']
        self.y_axis = new_column
        self._plot_chart()

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

        if self.use_dropdown:
            numeric_columns = self._get_valid_cols(df)
            dropdown = self._create_dropdown(numeric_columns)
            display(dropdown)

        self._plot_chart()

    def _plot_chart(self) -> None:
        """Helper method to plot the chart with the current y-axis column."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df["timestamp"], y=self.df[self.y_axis], mode='lines', name=self.y_axis))
        fig.update_layout(
            shapes=self._get_background_shapes(self.df),
            xaxis_title='Timestamp',
            yaxis_title=self.y_axis,
            title=self.title,
        )
        self._add_custom_legend(fig)
        fig.show(config={'displaylogo': False})

if __name__ == "__main__":
    chart = ThreeWChart(file_path="dataset/0/WELL-00001_20170201010207.parquet", use_dropdown=True)
    chart.plot()
