import plotly.graph_objects as go

def dark_theme(fig: go.Figure):

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor="#0E1117",

        plot_bgcolor="#0E1117",

        font=dict(
            color="black",
            family="Poppins"
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        )
    )

    return fig