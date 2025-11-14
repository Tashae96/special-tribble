import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.express as px
import base64
import io

# Initialize app
app = dash.Dash(__name__)
app.title = "Pay Gap & Communication Bias Tracker"

# --------------------------
# APP LAYOUT
# --------------------------
app.layout = html.Div(style={'font-family': 'Arial, sans-serif', 'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1("🔐 Pay Gap & Communication Bias Tracker", style={'color': '#4CAF50'}),
    html.H3("Cybersecurity + Fairness Analytics Capstone Project", style={'color': '#2196F3'}),
    html.P("Upload anonymized HR and communication datasets to detect pay gaps, promotion disparities, and communication bias."),

    html.H2("📁 Upload Data"),
    dcc.Upload(id='upload-hr', children=html.Button('Upload HR CSV', style={'backgroundColor':'#4CAF50', 'color':'white'})),
    html.Div(id='hr-preview', style={'margin-top': '10px'}),
    
    dcc.Upload(id='upload-comm', children=html.Button('Upload Communication CSV', style={'backgroundColor':'#2196F3', 'color':'white'})),
    html.Div(id='comm-preview', style={'margin-top': '10px'}),

    html.H2("💰 Raw Pay Gap Analysis"),
    html.Div(id='raw-paygap'),

    html.H2("⚖️ Adjusted Pay Gap (Regression)"),
    html.Div([
        html.Label("Select gender column:"),
        dcc.Dropdown(id='gender-col', placeholder="Select gender column"),
        html.Label("Select salary column:"),
        dcc.Dropdown(id='salary-col', placeholder="Select salary column"),
        html.Label("Select control variables:"),
        dcc.Dropdown(id='control-cols', multi=True, placeholder="Select control variables"),
        html.Button("Run Adjusted Pay Gap Model", id='run-regression', style={'backgroundColor':'#FF5722', 'color':'white', 'margin-top':'10px'}),
    ]),
    html.Div(id='regression-output', style={'margin-top': '20px'}),

    html.H2("💬 Communication Bias Analysis"),
    html.Div(id='comm-bias', style={'margin-top': '10px'}),

    html.H2("📊 Insights Summary"),
    html.P("""
        This tool helps identify potential disparities:
        - Pay gap between demographic groups  
        - Adjusted pay gap controlling for job-related factors  
        - Communication bias (slow response times, engagement gaps)  
        - Promotion fairness (if included in dataset)
    """)
])

# --------------------------
# CALLBACKS
# --------------------------

# Store uploaded data globally
df_hr_global = None
df_comm_global = None

# HR file upload
@app.callback(
    Output('hr-preview', 'children'),
    Output('gender-col', 'options'),
    Output('salary-col', 'options'),
    Output('control-cols', 'options'),
    Input('upload-hr', 'contents'),
    State('upload-hr', 'filename')
)
def handle_hr_upload(contents, filename):
    global df_hr_global
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_hr_global = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        options = [{'label': col, 'value': col} for col in df_hr_global.columns]
        preview = html.Div([
            html.H4(f"HR Data Preview: {filename}"),
            html.Pre(str(df_hr_global.head()))
        ])
        return preview, options, options, options
    return "", [], [], []

# Communication file upload
@app.callback(
    Output('comm-preview', 'children'),
    Input('upload-comm', 'contents'),
    State('upload-comm', 'filename')
)
def handle_comm_upload(contents, filename):
    global df_comm_global
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_comm_global = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        preview = html.Div([
            html.H4(f"Communication Data Preview: {filename}"),
            html.Pre(str(df_comm_global.head()))
        ])
        return preview
    return ""

# Raw pay gap calculation
@app.callback(
    Output('raw-paygap', 'children'),
    Input('gender-col', 'value'),
    Input('salary-col', 'value')
)
def raw_pay_gap(gender_col, salary_col):
    if df_hr_global is not None and gender_col and salary_col:
        median_salary = df_hr_global.groupby(gender_col)[salary_col].median()
        if len(median_salary) >= 2:
            group1 = median_salary.index[0]
            group2 = median_salary.index[1]
            gap = 1 - (median_salary.iloc[0] / median_salary.iloc[1])
            return html.Div([
                html.H4("Median Salary by Group"),
                html.Pre(str(median_salary)),
                html.P(f"Raw Pay Gap ({group1} vs {group2}): {gap:.2%}", style={'color':'red', 'font-weight':'bold'})
            ])
    return "Select gender and salary columns to see raw pay gap."

# Adjusted pay gap (regression)
@app.callback(
    Output('regression-output', 'children'),
    Input('run-regression', 'n_clicks'),
    State('gender-col', 'value'),
    State('salary-col', 'value'),
    State('control-cols', 'value')
)
def adjusted_pay_gap(n_clicks, gender_col, salary_col, controls):
    if n_clicks and df_hr_global is not None and gender_col and salary_col:
        formula = f"{salary_col} ~ C({gender_col})"
        if controls:
            for c in controls:
                if c != salary_col:
                    if np.issubdtype(df_hr_global[c].dtype, np.number):
                        formula += f" + {c}"
                    else:
                        formula += f" + C({c})"
        try:
            model = smf.ols(formula, data=df_hr_global).fit()
            return html.Pre(model.summary().as_text())
        except Exception as e:
            return html.Div(f"Error running regression: {e}", style={'color':'red'})
    return "Click 'Run Adjusted Pay Gap Model' after selecting columns."

# Communication bias analysis
@app.callback(
    Output('comm-bias', 'children'),
    Input('upload-comm', 'contents'),
)
def communication_bias(contents):
    if df_comm_global is not None and df_hr_global is not None:
        if 'pseud_id' in df_hr_global.columns and 'receiver_pseud' in df_comm_global.columns:
            # Merge gender
            gender_col = df_hr_global.columns[0]  # just pick first column for demo
            df_merge = df_comm_global.merge(
                df_hr_global[['pseud_id', gender_col]],
                left_on='receiver_pseud',
                right_on='pseud_id',
                how='left'
            )
            if 'response_time_seconds' in df_merge.columns:
                median_response = df_merge.groupby(gender_col)['response_time_seconds'].median().reset_index()
                fig = px.bar(median_response, x=gender_col, y='response_time_seconds', color=gender_col,
                             labels={'response_time_seconds': 'Median Response Time (s)'})
                return dcc.Graph(figure=fig)
            return "Column 'response_time_seconds' missing in communication data."
        return "Missing pseudonym columns. Ensure HR has 'pseud_id' and comm data has 'receiver_pseud'."
    return "Upload HR and communication datasets to analyze."

# --------------------------
# RUN SERVER
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
