# Yahtzee bot

This project implements the Yathzee game. The object is to roll dice for scoring combinations and gate the highest total score.
For more informations about rules of the game please visit:
``` bash
https://www.hasbro.com/common/instruct/yahtzee.pdf
```

## Presentation of the project

The project contains the following files:
```
.
├── requirements.txt
├── README.md
├── LICENSE.md
├── app.py
├── yahtzee_game
│   ├── scoreboard.py
│   ├── rules.py
│   ├── main.py
│   └── hand.py 
│ 
├── yahtzee_dash_app
│   ├── images
│   └── utils.py
└── yahtzee_bot
    ├── weights
    ├── data.py
    ├── hash.py
    ├── network.py
    ├── ppo_loss.py
    ├── train.py
    └── training.py
```
The `yahtzee_bot/data` generates samples of parts played by the actions of the model.
The `yahtzee_bot/train` generate a neural network used to find the best choice.

The `app.py`  implements the final user interface of the game.

## Installation guide

To clone the repository:
```bash
 $ git clone https://github.com/thibautvalour/Yahtzee.git
```
To play the game:
```bash
 $ Python app.py
 ```
Training the bot with  desired parameters:
 ```bash
 $ Python yahtzee_bot/train.py -n_ite 100 -n_games 4096, use -h for more functionality.
 ```


 
## Contibutors
 * Augustin Combes
 * Nicolas Tollot
 * Thibaut Valour
 * Aurelie Ngalula
 
 ## License
 This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) for details
 
