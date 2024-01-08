from torch import Tensor

from ann_Hearts import ANN_Hearts


def is_card_in_hand_player(card: int, player: ANN_Hearts) -> bool:
    if player.card_vectors[card][0] == 1:
        return True
    else:
        return False


def is_suit_in_hand_player(suit_range, player: ANN_Hearts) -> bool:
    suit_int_hand = False
    for card in suit_range:
        if player.card_vectors[card][0] == 1.0:
            return True
    return suit_int_hand


def get_cards_in_hand_within_suit(suit_range, player_object: ANN_Hearts) -> [float]:
    cards = []
    for card in suit_range:
        if player_object.card_vectors[card][0] == 1.0:
            cards.append(card)
    return cards


def get_best_card_in_hand_in_suit(suit_range, player: ANN_Hearts, output_layer: Tensor) -> float:
    allowed_cards_in_hand = get_cards_in_hand_within_suit(suit_range, player)
    highest_change = float('-inf')
    best_card = None
    for i in range(len(allowed_cards_in_hand)):
        if output_layer[allowed_cards_in_hand[i]] > highest_change:
            highest_change = output_layer[allowed_cards_in_hand[i]]
            best_card = allowed_cards_in_hand[i]
    return best_card
