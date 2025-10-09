import pandas as pd
import plotly.graph_objects as go
import plotly.express as px              # only for the colour palette

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(r"full_results_EHNN.csv")

# ------------------------------------------------------------------
# 2. Coerce numeric columns
# ------------------------------------------------------------------
num_cols = [
    'roc_auc', 'pr_auc', 'f1', 'recall', 'precision',
    'accuracy', 'mcc', 'brier', 'tpr@1%fpr'
]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

# ------------------------------------------------------------------
# 3. Sort by dataset so the line connects in the right order
# ------------------------------------------------------------------
df = df.sort_values('dataset')

# ------------------------------------------------------------------
# 4. Build the figure
# ------------------------------------------------------------------
metrics = ['roc_auc', 'pr_auc', 'f1', 'recall', 'precision',
           'accuracy', 'mcc', 'tpr@1%fpr']
palette = px.colors.qualitative.Set2

fig = go.Figure()

for m, c in zip(metrics, palette):
    fig.add_trace(
        go.Scatter(
            x=df['dataset'],
            y=df[m],
            mode='lines+markers',
            name=m.upper(),
            line=dict(color=c, width=2),
            marker=dict(size=7)
        )
    )

fig.update_layout(
    title='Performance metrics across datasets',
    xaxis_title='Dataset',
    yaxis_title='Score',
    template='plotly_white',
    height=600,
    margin=dict(l=40, r=40, t=60, b=100),
    legend_title_text='Metric'
)

# categorical x-axis keeps datasets in the order we supplied
fig.update_xaxes(type='category', tickangle=-45)
fig.write_html("performance_metrics.html")   #
# 2. Static raster image (requires kaleido: pip install -U kaleido)
fig.write_image("performance_metrics.png", width=900, height=600, scale=2)

fig.show()