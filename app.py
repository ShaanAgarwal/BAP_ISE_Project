import pandas as pd
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
import io
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("Updated Dataset.csv")
df['Start Year'] = df['Year'].str.split('-').str[0]
app = dash.Dash(__name__)
guide_names = df[['Guid 1 Name', 'Guid 2 Name']].stack().unique()
all_years = sorted(df['Start Year'].unique())
application_types = df['Application Type'].unique()

publication_counts_all_years = df['Start Year'].value_counts().reset_index()
publication_counts_all_years.columns = ['Year', 'Publication Count']
publication_counts_all_years = publication_counts_all_years.sort_values(by='Year')
fig_timeline = px.line(publication_counts_all_years, x='Year', y='Publication Count', title='Timeline of Paper Publications by Year')

department_distribution = df['Application Department'].value_counts().reset_index()
department_distribution.columns = ['Department', 'Paper Count']
fig_department_distribution = px.bar(department_distribution, x='Department', y='Paper Count', title='Department-wise Distribution of Papers')

app.layout = html.Div([
    html.H1("Dashboard"),
    html.Div([
        html.Div([
            dcc.Graph(figure=fig_timeline)
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(figure=fig_department_distribution)
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    dcc.Dropdown(
        id='application-type-dropdown',
        options=[{'label': i, 'value': i} for i in application_types],
        value=application_types[0]
    ),
    html.Div([
        html.Div([
            html.Div([
                html.H3("Bar Chart"),
                dcc.Graph(id='bar-graph')
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.H3("Sunburst Chart"),
                dcc.Graph(id='sunburst-chart')
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.H3("Word Cloud of Paper Titles"),
                html.Div(id='wordcloud-container')
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
            dcc.Graph(id='sankey-diagram')
            ], style={'width': '50%', 'display': 'inline-block'}),
        ]),
    ]),
    dcc.Dropdown(
        id='guide-dropdown',
        options=[{'label': guide, 'value': guide} for guide in guide_names],
        value=guide_names[0],
        clearable=False,
    ),
    html.Div([
        html.H3("Publications Over Time"),
        dcc.Graph(id='publication-graph')
    ]),
    html.Div([
        html.Div([
            html.Div([
                html.H3("Paper Count by Type of Paper"),
                dcc.Graph(id='paper-type-count', style={'width': '100%', 'height': '400px'})
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.H3("Paper Count by Application Department and Type of Publication"),
                dcc.Graph(id='paper-department-count', style={'width': '100%', 'height': '400px'})
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ]),
    html.Div([
        html.H3("Top Guides with Most Papers"),
        dcc.Input(
            id='top-guides-input',
            type='number',
            placeholder='Enter number of top guides',
            min=1,
            max=len(guide_names),
            step=1,
            value=5
        ),
        dcc.Graph(id='top-guides-bar-chart', style={'width': '100%', 'height': '400px'})
    ]),
])

@app.callback(
    [Output('bar-graph', 'figure'), Output('sunburst-chart', 'figure')],
    [Input('application-type-dropdown', 'value')]
)
def update_charts(selected_application_type):
    filtered_df = df[df['Application Type'] == selected_application_type]
    filtered_df.loc[:, 'Application Course'] = filtered_df['Application Course'].fillna('Unknown')
    filtered_df.loc[:, 'Application Department'] = filtered_df['Application Department'].fillna('Unknown')
    filtered_df.loc[:, 'Application Class'] = filtered_df['Application Class'].fillna('Unknown')
    counts = filtered_df['Start Year'].value_counts().reset_index()
    counts.columns = ['Year', 'Count']
    counts = counts.sort_values(by='Year')
    fig_bar_updated = px.bar(counts, x='Year', y='Count', title=f'Count of Each Year for {selected_application_type}')
    fig_sunburst_updated = px.sunburst(filtered_df, path=['Application Course', 'Application Department', 'Application Class'])
    fig_sunburst_updated.update_traces(
        marker=dict(
            colors=px.colors.qualitative.Set3
        ),
        textinfo='label+percent entry',
        insidetextorientation='radial'
    )
    return fig_bar_updated, fig_sunburst_updated

@app.callback(
    Output('publication-graph', 'figure'),
    [Input('guide-dropdown', 'value')]
)
def update_graph(selected_guide):
    filtered_df = df[(df['Guid 1 Name'] == selected_guide) | (df['Guid 2 Name'] == selected_guide)]
    filtered_df = filtered_df.dropna(subset=['Start Year'])
    publication_counts = filtered_df['Start Year'].value_counts().reset_index()
    publication_counts.columns = ['Year', 'Publication Count']
    publication_counts = publication_counts.set_index('Year').reindex(all_years).fillna(0).reset_index()
    hover_text = []
    customdata = []
    for year in publication_counts['Year']:
        papers = filtered_df[filtered_df['Start Year'] == year]['Title of Paper'].tolist()
        hover_text.append(f"Year: {year}<br>Publication Count: {publication_counts[publication_counts['Year'] == year]['Publication Count'].values[0]}")
        customdata.append('<br>'.join(papers))
    fig = px.line(publication_counts, x='Year', y='Publication Count', title=f'Publications Over Time for {selected_guide}', hover_data={'Year': False, 'Publication Count': True, 'hovertext': hover_text, 'customdata': customdata})
    fig.update_traces(hovertemplate='<b>Year</b>: %{x}<br><b>Publication Count</b>: %{y}<br><br>%{customdata}')
    
    return fig

@app.callback(
    [Output('paper-type-count', 'figure'), Output('paper-department-count', 'figure')],
    [Input('guide-dropdown', 'value')]
)
def update_paper_count_graphs(selected_guide):
    filtered_df = df[(df['Guid 1 Name'] == selected_guide) | (df['Guid 2 Name'] == selected_guide)]
    paper_type_counts = filtered_df['Type of Paper'].value_counts().reset_index()
    paper_type_counts.columns = ['Type of Paper', 'Count']
    fig_paper_type_count = px.bar(paper_type_counts, x='Type of Paper', y='Count', title='Paper Count by Type of Paper')
    publication_department_counts = filtered_df.groupby(['Type of Publication', 'Application Department']).size().reset_index(name='Count')
    fig_paper_department_count = px.bar(publication_department_counts, x='Type of Publication', y='Count', color='Application Department', barmode='group', title='Paper Count by Type of Publication and Application Department')
    return fig_paper_type_count, fig_paper_department_count

@app.callback(
    Output('top-guides-bar-chart', 'figure'),
    [Input('top-guides-input', 'value')]
)
def update_top_guides_bar_chart(top_count):
    if top_count is None or top_count <= 0:
        top_count = 5
    top_guides = pd.concat([df['Guid 1 Name'], df['Guid 2 Name']]).value_counts().nlargest(top_count)
    fig = px.bar(x=top_guides.index, y=top_guides.values, title=f'Top {top_count} Guides with Most Papers')
    fig.update_xaxes(title='Guide Names')
    fig.update_yaxes(title='Paper Count')
    return fig

@app.callback(
    Output('wordcloud-container', 'children'),
    [Input('application-type-dropdown', 'value')]
)
def generate_wordcloud(selected_application_type):
    filtered_df = df[df['Application Type'] == selected_application_type]
    filtered_df = filtered_df.dropna(subset=['Title of Paper'])
    stop_words = set(stopwords.words('english'))
    titles = ' '.join(filtered_df['Title of Paper'])
    titles_no_stopwords = ' '.join([word for word in titles.split() if word.lower() not in stop_words])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_no_stopwords)
    img_stream = io.BytesIO()
    wordcloud.to_image().save(img_stream, format='PNG')
    encoded_image = base64.b64encode(img_stream.getvalue()).decode()
    return html.Img(src=f'data:image/png;base64,{encoded_image}')

@app.callback(
    Output('sankey-diagram', 'figure'),
    [Input('application-type-dropdown', 'value')]
)
def update_sankey_diagram(selected_application_type):
    filtered_df = df[df['Application Type'] == selected_application_type]
    department_type_counts = filtered_df.groupby(['Application Department', 'Type of Publication']).size().reset_index(name='Count')
    sources = department_type_counts['Application Department']
    targets = department_type_counts['Type of Publication']
    values = department_type_counts['Count']
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sources.tolist() + targets.tolist(),
            color="blue"
        ),
        link=dict(
            source=[sources.tolist().index(source) for source in sources],
            target=[targets.tolist().index(target) + len(sources) for target in targets],
            value=values.tolist(),
        )
    )])
    fig_sankey.update_layout(title_text="Flow of Publications by Department and Type", font=dict(size=10))
    return fig_sankey

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=False)