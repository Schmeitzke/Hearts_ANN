from typing import List

import numpy as np
import torch
from torch import float32

from ann_Hearts import Utility
from ann_Hearts.ANN_Hearts import ANN_Hearts


class Game:

    def __init__(self):
        self.current_player = -1
        self.starting_player = -1

        self.HEARTS_RANGE = range(0, 13)  # Based on Card.java
        self.CLUBS_RANGE = range(13, 26)
        self.DIAMONDS_RANGE = range(26, 39)
        self.SPADES_RANGE = range(39, 52)

    @staticmethod
    def deal_cards(playerList: List[ANN_Hearts]):
        # Reset hands of all players before new deal
        for player in playerList:
            player.card_vectors = np.zeros((52, 5), dtype=float)
        cards = np.random.permutation(52)
        sliced_shuffled_cards = np.array_split(cards, 4)

        for player_index, player_cards in enumerate(sliced_shuffled_cards):
            for card in player_cards:
                # The card is in this player's hand
                playerList[player_index].card_vectors[card][0] = 1.0
                # Mark the card as not in the other players' hands
                for other_player_index in range(4):
                    if other_player_index != player_index:
                        playerList[other_player_index].card_vectors[card][1] = 1.0

        # counter = 0
        # print("HANDS DEALT!")
        # for player in playerList:
        #     print("Hand player:", counter)
        #     for row in player.card_vectors:
        #         print(row)
        #     print()
        #     counter += 1

    def play_round(self, playerList: List[ANN_Hearts]):
        trick = [-1] * 4
        corrected_card = [False] * 4

        for i in range(4):

            # print("\nPLAY ROUND PLAYER:", self.current_player, "(current player)")
            # Prepare and forward pass tensor
            input_tensor = torch.flatten(torch.tensor(playerList[self.current_player].card_vectors, dtype=float32))
            # print("Input tensor:", input_tensor)
            output_layer = playerList[self.current_player].forward(input_tensor)
            # print("Output layer:", output_layer)

            # Interpret output layer
            outputLayer = output_layer.detach()
            # print("Output layer detached:", output_layer)

            # Retrieve cards in hand agent
            cards_in_hand = []
            for card in range(52):
                if playerList[self.current_player].card_vectors[card][0] == 1:
                    cards_in_hand.append(card)
            # print("Cards in hand:", cards_in_hand)

            # Determine best card which is in hand
            chosen_card_prob = -1
            chosen_card = -1
            for card_in_hand in cards_in_hand:
                if outputLayer[card_in_hand] > chosen_card_prob:
                    chosen_card_prob = outputLayer[card_in_hand]
                    chosen_card = card_in_hand
            trick[self.current_player] = chosen_card
            # print("Chosen card:", chosen_card)

            suit_starting_card = self.determine_suit_range(trick[self.starting_player])
            # print("Suit range starting card:", suit_starting_card)

            # If not starting player and the suit is in hand player, play best card in hand and in suit range
            if (self.starting_player != self.current_player and
                    Utility.is_suit_in_hand_player(suit_starting_card, playerList[self.current_player]) and
                    chosen_card not in suit_starting_card):

                cards_in_hand_correct_suit = []
                # print("Cards in hand and in suit:", cards_in_hand_correct_suit)
                chosen_card_prob = -1
                chosen_card = -1
                for card_in_hand in cards_in_hand:
                    if card_in_hand in suit_starting_card:
                        cards_in_hand_correct_suit.append(card_in_hand)
                for card_in_hand in cards_in_hand_correct_suit:
                    if outputLayer[card_in_hand] > chosen_card_prob:
                        chosen_card_prob = outputLayer[card_in_hand]
                        chosen_card = card_in_hand
                corrected_card[self.current_player] = True
                # print("Chosen card looking at hand and starting suit:", chosen_card)

            self.update_player_vectors_during_round(playerList, chosen_card)

            trick[self.current_player] = chosen_card

            # Advance turn
            self.current_player += 1
            if self.current_player == 4:
                self.current_player = 0

        # print("Trick:", trick)
        # print("Corrected card:", corrected_card)

        # counter = 0
        # print("\nHANDS UPDATED BEFORE RESETTING TRICK!")
        # for player in playerList:
        #     print("Hand player:", counter)
        #     for row in player.card_vectors:
        #         print(row)
        #     print()
        #     counter += 1

        self.update_player_vectors_after_trick(playerList)

        # counter = 0
        # print("\nHANDS UPDATED AFTER RESETTING TRICK!")
        # for player in playerList:
        #     print("Hand player:", counter)
        #     for row in player.card_vectors:
        #         print(row)
        #     print()
        #     counter += 1

        losses, round_winner = self.calculate_loss(trick, corrected_card)

        # print("Losses:", losses)
        # print("Round winner:", round_winner)

        return losses, round_winner

    def update_player_vectors_during_round(self, playerList: List[ANN_Hearts], card_index: int):
        playerList[self.current_player].card_vectors[card_index][0] = 0.0  # not in playing agent’s hand
        playerList[self.current_player].card_vectors[card_index][2] = 1.0  # played in the current trick by the agent
        playerList[self.current_player].card_vectors[card_index][4] = 1.0  # card consumed

        for other_player in range(4):
            if self.current_player != other_player:
                playerList[other_player].card_vectors[card_index][1] = 0.0  # not in any other player’s hand
                playerList[other_player].card_vectors[card_index][
                    3] = 1.0  # played in the current trick by other player
                playerList[other_player].card_vectors[card_index][4] = 1.0  # card consumed

    @staticmethod
    def update_player_vectors_after_trick(playerList: List[ANN_Hearts]):
        for player in range(4):
            for card in range(52):
                playerList[player].card_vectors[card][2] = 0.0
                playerList[player].card_vectors[card][3] = 0.0

    def calculate_loss(self, trick: [float], corrected_cards: [bool]):
        loss = [0.0, 0.0, 0.0, 0.0]
        totalLoss = 0.0
        for i in range(4):
            if trick[i] in self.HEARTS_RANGE:
                totalLoss += 2.0
            if corrected_cards[i]:
                loss[i] += 20
        round_winner = self.determine_round_winner(trick)
        loss[round_winner] += totalLoss

        return loss, round_winner

    def determine_round_winner(self, trick):
        round_winner = self.starting_player
        starting_card = trick[self.starting_player]
        suit_of_starting_card = self.determine_suit_range(starting_card)

        for player in range(4):
            if starting_card < trick[player] <= max(suit_of_starting_card):
                starting_card = trick[player]
                round_winner = player
        return round_winner

    def determine_suit_range(self, card: int):
        suit = -1
        if card in self.HEARTS_RANGE:
            suit = self.HEARTS_RANGE
        elif card in self.CLUBS_RANGE:
            suit = self.CLUBS_RANGE
        elif card in self.DIAMONDS_RANGE:
            suit = self.DIAMONDS_RANGE
        elif card in self.SPADES_RANGE:
            suit = self.SPADES_RANGE
        return suit
