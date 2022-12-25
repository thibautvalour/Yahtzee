from dash.dependencies import Input, Output, State
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy.random as random
from PIL import Image

from compute_turn_score import score

pil_images = [Image.open(f"yahtzee_dash_app/images/dice_{idx}.png") for idx in range(1,7)]

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# initialize game state with all dices as 0
game_state = [0, 3, 2, 5, 0]

# list of possible choices for the player
choices = ['aces','twos','threes','fours','fives','sixes','three_of_a_kind','four_of_a_kind','full_house','small_straight','large_straight','yahtzee','chance']

#init called only once when the app is launched
@app.callback(Output('header', 'children'),
              [Input('dummy', 'value')]) #dummy input
def update_output(input_value):
    # print('LAUNCHED THE APP')
    return html.Div([
        html.Div([html.Button('Roll Dices', id='button', style={"align": "center"})], style={"text-align": "center"}),
        html.Div(id='rolls_left', style={"text-align": "center"})
    ], id='header')


# layout for the website
app.layout = html.Div([
    html.H1("AI Yahtzee", style={"text-align": "center"}, id='dummy'),
    html.H6("Are you ready to challenge the Machine?", style={"text-align": "center"}),
    html.Div([
        html.Div(id='header')
    ]),
    html.Div([
        html.Div([
            html.Img(id='dice1', src=pil_images[0], width='150', height='150'),
            dcc.Checklist(id='keep1', options=[{'label': 'Keep Dice 1', 'value': 1}]),
        ], style={'display': 'inline-block', 'background-image': pil_images[0]}),
        html.Div([
            html.Img(id='dice2', src=pil_images[0], width='150', height='150'),
            dcc.Checklist(id='keep2', options=[{'label': 'Keep Dice 2', 'value': 2}]),
        ], style={'display': 'inline-block'}),
        html.Div([
            html.Img(id='dice3', src=pil_images[0], width='150', height='150'),
            dcc.Checklist(id='keep3', options=[{'label': 'Keep Dice 3', 'value': 3}]),
        ], style={'display': 'inline-block'}),
        html.Div([
            html.Img(id='dice4', src=pil_images[0], width='150', height='150'),
            dcc.Checklist(id='keep4', options=[{'label': 'Keep Dice 4', 'value': 4}]),
        ], style={'display': 'inline-block'}),
        html.Div([
            html.Img(id='dice5', src=pil_images[0], width='150', height='150'),
            dcc.Checklist(id='keep5', options=[{'label': 'Keep Dice 5', 'value': 5}]),
        ], style={'display': 'inline-block'}),
        html.Div(id='choice')
    ], style={"text-align": "center"}),
    html.Div(id='score'),
    html.Div('Total player score :', style={"text-align": "center"}),
    html.Div(id='total_score'),
    html.Div(0, id='score_total', style={"text-align":'center'}),
    dcc.Checklist(id='available_choices', options=choices, style={'display': 'none'}),
], style={'background-image': 'url("/images/background.jpeg")'}
)

@app.callback(
    [Output('choice', 'children'), Output('total_score', 'children')],
    [Input('rule', 'value')],
    [State('total_score', 'children')]
)
def display_rule(value, total_score):
    print('display_rule')
    print(value, total_score)
    print()
    s = score(value, game_state)
    return html.Div(f"With this rule, you would score {s} points at this turn"), total_score


# Update the total score and the remaining choices when 'Submit' is hit
@app.callback(
    [Output('score_total', 'children'), Output('available_choices', 'options')],
    [Input('button', 'n_clicks')],
    [State('score_total', 'children'), State('rule', 'value'), State('available_choices', 'options')]
)
def printer(clicks, totscore, rule, avachoices):
    if clicks is not None and clicks%3==0:
        updated_choices = [choice for choice in avachoices if not choice==rule]
        current_score = totscore[0] if type(totscore)==list else totscore
        return [current_score+score(rule, game_state)], updated_choices
    return [totscore], avachoices


# Update the number of rolls left when playing the turns
@app.callback(
    Output('rolls_left', 'children'),
    [Input('button', 'n_clicks'), State('available_choices', 'options')]
)
def update_rolls_left(n_clicks, avachoices):
    if n_clicks is None:
        return html.Div(['🎲']*2)

    n_clicks_left = 2-n_clicks%3
    if n_clicks_left > 0 :
        return html.Div(['🎲']*n_clicks_left)
    else :
        return html.Div([
            'Choose your rule:',
            dcc.Dropdown(id='rule', options=avachoices, value=choices[0],
                style={
                    'display': 'block','margin-left': 'calc((100% - 550px) / 2)','margin-right': 'calc((100% - 550px) / 2)'}
            )])


# callback to update dices and game state when button is clicked
@app.callback(
    [Output('dice1', 'src'), Output('dice2', 'src'), Output('dice3', 'src'), Output('dice4', 'src'), Output('dice5', 'src'), Output('score', 'children')],
    [Input('button', 'n_clicks')],
    [State('keep1', 'value'), State('keep2', 'value'), State('keep3', 'value'), State('keep4', 'value'), State('keep5', 'value'), State('choice', 'value')]
)
def update_dices(n_clicks, keep1, keep2, keep3, keep4, keep5, choice):
    # roll dices that are not kept
    if keep1 is None:
        game_state[0] = random.randint(1, 7)
    if keep2 is None:
        game_state[1] = random.randint(1, 7)
    if keep3 is None:
        game_state[2] = random.randint(1, 7)
    if keep4 is None:
        game_state[3] = random.randint(1, 7)
    if keep5 is None:
        game_state[4] = random.randint(1, 7)
    print('game state', game_state, score('aces', game_state))

    # update dice images
    dice1_src = pil_images[game_state[0]-1]
    dice2_src = pil_images[game_state[1]-1]
    dice3_src = pil_images[game_state[2]-1]
    dice4_src = pil_images[game_state[3]-1]
    dice5_src = pil_images[game_state[4]-1]

    # compute score for this turn
    if choice:
        score_text = f'Score: {score(choice, game_state)}'
    else:
        score_text = ''

    return dice1_src, dice2_src, dice3_src, dice4_src, dice5_src, score_text

@app.callback(
    Output('button', 'children'),
    [Input('button', 'n_clicks')]
)
def switch_roll_submit(clicks):
    if clicks is None:
        return 'Roll Dices'
    return 'Roll Dices' if clicks%3!=2 else 'Submit'


if __name__ == '__main__':
    app.run_server()