from dash.dependencies import Input, Output, State
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy.random as random
from PIL import Image

from utils import score, bot_decision

pil_images = [Image.open(f"yahtzee_dash_app/images/dice_{idx}.png") for idx in range(1,13)]
bot_image = Image.open(f"yahtzee_dash_app/images/robot.jpeg")

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# initialize game state with all dices as 0
game_state = [0, 3, 2, 5, 0]

# list of possible choices for the player
choices = ['aces','twos','threes','fours','fives','sixes','three_of_a_kind','four_of_a_kind','full_house','small_straight','large_straight','yahtzee','chance']
choices_bot = ['aces','twos','threes','fours','fives','sixes','three_of_a_kind','four_of_a_kind','full_house','small_straight','large_straight','yahtzee','chance']

#init called only once when the app is launched
@app.callback(Output('header', 'children'),
              [Input('dummy', 'value')]) #dummy input
def update_output(input_value):
    # print('LAUNCHED THE APP')
    return html.Div([
        html.Div([html.Button('Roll Dices', id='button', style={"align": "center"})], style={"text-align": "center"}),
        html.Div(id='rolls_left', style={"text-align": "center"})
    ], id='header')

display = html.Div(id='display')

# layout for the website
app.layout = html.Div([
    html.Div([
        html.Div([
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
            html.Div([
                html.H6('SCOREBOARD'),
                html.Div([
                    html.Div('Total player score :', style={'display':'inline-block'}),
                    html.Div(id='total_score', style={'display':'inline-block'}),
                    html.Div(0, id='score_total', style={"text-align":'center', 'display':'inline-block', 'margin-left':10}),
                ]),
                html.Div([
                    html.Div('Total bot score :', style={'display':'inline-block'}),
                    html.Div(id='total_score_bot', style={'display':'inline-block'}),
                    html.Div(0, id='score_total_bot', style={"text-align":'center', 'display':'inline-block', 'margin-left':10}),
                ])
            ], style={'position': 'absolute', 'top': '10%', 'left': '10%', 'transform': 'translate(-50%, -50%)'}),
            dcc.Checklist(id='available_choices', options=choices, style={'display': 'none'}),
        ]),
    ]),

    # html.Br(),
    
    html.Div([
        html.Div([
            html.Div([
                html.Img(id='bot-img', src=bot_image, style={'width': '40%', 'height': '40%', 'position': 'absolute', 'top': '70%', 'left': '50%', 'transform': 'translate(-50%, -50%)'})
        ], id='bot_portrait', ),
            html.Div(
                html.Div([
                    html.H6("Bot turn"),
                ],
                    id='bot-div', style={'display':'none'}),
                style={'display': 'inline-block', 'margin-left':100}, id='game_bot'),
                ])
            ])
    ], style={'text-align':'center'})


# PLAYER : Update the chosen rule & score
@app.callback(
    [Output('choice', 'children'), Output('total_score', 'children')],
    [Input('rule', 'value')],
    [State('total_score', 'children')]
)
def display_rule(value, total_score):
    s = score(value, game_state)
    return html.Div(f"With this rule, you would score {s} points at this turn"), total_score


# Update the total score and the remaining choices when 'Submit' is hit, also make the bot game appear
@app.callback(
    [Output('score_total', 'children'), Output('available_choices', 'options'), Output('bot-div', 'style'), 
        Output('bot-div', 'children'), Output('score_total_bot', 'children')],
    [Input('button', 'n_clicks')],
    [State('score_total', 'children'), State('rule', 'value'), State('available_choices', 'options'), 
        State('bot-div', 'style'), State('score_total_bot', 'children'),
        ]
)
def hit_submit(clicks, totscore, rule, avachoices, botstyle, score_tot_bot):
    # print('hit submit', botstyle)
    if clicks is not None and clicks%3==0:
        updated_choices = [choice for choice in avachoices if not choice==rule]
        current_score = totscore[0] if type(totscore)==list else totscore
        player_score = current_score+score(rule, game_state)

        mem_dices = []
        mem_held = []

        global choices_bot
        
        #first roll
        turn = 1
        dices = [random.randint(1, 7) for _ in range(5)]
        mem_dices.append(dices)
        keep_list = bot_decision(dices, choices_bot, turn)
        mem_held.append(keep_list)

        #second roll 
        turn = 2
        dices = [random.randint(1, 7) if not keep_list[idx] else dices[idx] for idx in range(5)]
        mem_dices.append(dices)
        keep_list = bot_decision(dices, choices_bot, turn)
        mem_held.append(keep_list)
        mem_held.append([True]*5)

        #last roll
        turn = 3
        dices = [random.randint(1, 7) if not keep_list[idx] else dices[idx] for idx in range(5)]
        mem_dices.append(dices)
        rule = bot_decision(dices, choices_bot, turn)

        choices_bot = [e for e in choices_bot if not e == rule]

        dices_idx = [
            [(1-int(not mem_held[0][idx]))*6+mem_dices[0][idx] for idx in range(len(mem_dices[0]))],
            [(1-int(not mem_held[1][idx]))*6+mem_dices[1][idx] for idx in range(len(mem_dices[1]))],
            [(1-int(not mem_held[2][idx]))*6+mem_dices[2][idx] for idx in range(len(mem_dices[2]))],
        ]

        s = score(rule, dices)

        # print("dices_idx", dices_idx)

        child = html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div('Turn 1 :', style={'display':'inline-block'}),
                        html.Div(list(map(lambda x: html.Img(src=pil_images[x-1], width='50', height='50'), 
                            dices_idx[0]
                            )), style={'display':'inline-block'}),
                    ]),
                    html.Div([
                        html.Div('Turn 2 :', style={'display':'inline-block'}),
                        html.Div(list(map(lambda x: html.Img(src=pil_images[x-1], width='50', height='50'), 
                            dices_idx[1]
                            )), style={'display':'inline-block'}),
                    ]),
                    html.Div([
                        html.Div('Turn 3 :', style={'display':'inline-block'}),
                        html.Div(list(map(lambda x: html.Img(src=pil_images[x-1], width='50', height='50'), 
                            dices_idx[2]
                            )), style={'display':'inline-block'}),
                    ]),
                    html.Div([
                        html.Div('Rule chosen :', style={'display':'inline-block'}),
                        html.Div(rule, style={'display':'inline-block', 'margin-left':10}),
                    ]),
                    html.Div([
                        html.Div('Turn score :', style={'display':'inline-block'}),
                        html.Div(s, style={'display':'inline-block', 'margin-left':10}),
                    ]),
                ]),
            ], style={'position': 'absolute', 'top': '65%', 'left': '80%', 'transform': 'translate(-50%, -50%)'})
        ])

        # print(mem_dices)
        # print(mem_held)
        # print(rule)
        score_tot_bot = score_tot_bot[0] if type(score_tot_bot) is list else score_tot_bot
        return [player_score], updated_choices, {'text-align':'center'}, child, [score_tot_bot+s]

    return [totscore], avachoices, {'text-align':'center'}, [], [score_tot_bot]

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
            dcc.Dropdown(id='rule', options=avachoices, value=avachoices[0],
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

# 
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