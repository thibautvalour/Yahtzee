import numpy.random as random

def bot_decision(dices, choices_bot, turn):
    if turn in [1,2]:
        # we have to return a list of True and False depending of which dices the bot decides to keep
        keepList = [True if dice>3 else False for dice in dices]
        return keepList
    else: 
        #turn = 3, we have to return a choice of rule
        random_idx = random.randint(0, len(choices_bot))
        choice = choices_bot[random_idx]
        return choice


def score(rule_str, game_state):
    if rule_str=='aces':
        return sum([x for x in game_state if x==1])

    if rule_str=='twos':
        return sum([x for x in game_state if x==2])

    if rule_str=='threes':
        return sum([x for x in game_state if x==3])

    if rule_str=='fours':
        return sum([x for x in game_state if x==4])

    if rule_str=='fives':
        return sum([x for x in game_state if x==5])

    if rule_str=='sixes':
        return sum([x for x in game_state if x==6])

    if rule_str=='three_of_a_kind':
        for i in range(1,7):
            if game_state.count(i) >= 3:
                return sum(game_state)
        return 0

    if rule_str=='four_of_a_kind':
        for i in range(1,7):
            if game_state.count(i) >= 4:
                return sum(game_state)
        return 0

    if rule_str=='full_house':
        for i in game_state:
            x = game_state.count(i)
            if x == 3:
                for i in game_state:
                    y = game_state.count(i)
                    if y == 2 and x != y:
                        return 25
        return 0

    if rule_str=='small_straight':
        hand = list(set(sorted(game_state)))
        try:
            if len(hand) >= 4:
                for idx, val in enumerate(hand):
                    if hand[idx+1] == val+1 and \
                            hand[idx+2] == val+2 and \
                            hand[idx+3] == val+3:
                        return 30
        except IndexError:
            pass
        return 0

    if rule_str=='large_straight':
        hand = list(set(sorted(game_state)))
        try:
            if len(hand) >= 5:
                for idx, val in enumerate(hand):
                    if hand[idx+1] == val+1 and \
                            hand[idx+2] == val+2 and \
                            hand[idx+3] == val+3 and \
                            hand[idx+4] == val+4:
                        return 40
        except IndexError:
            pass
        return 0

    if rule_str=='yahtzee':
        return 50 if len(set(game_state)) == 1 else 0

    if rule_str=='chance':
        return sum(game_state)

    else :
        return 'Please select your rule'
