import random

import torch
import torch.optim as optim

from ann_Hearts.ANN_Hearts import ANN_Hearts
from ann_Hearts.Game import Game

# Number of epochs for training
num_epochs = 10000

# Initialize players (models)
player_1 = ANN_Hearts()
player_2 = ANN_Hearts()
player_3 = ANN_Hearts()
player_4 = ANN_Hearts()

player_1.train()
player_2.train()
player_3.train()
player_4.train()

playerList = [player_1, player_2, player_3, player_4]

# Optimizers for each player
optimizers = [
    optim.Adam(player_1.parameters()),
    optim.Adam(player_2.parameters()),
    optim.Adam(player_3.parameters()),
    optim.Adam(player_4.parameters())
]

played_wrong_cards = [0.0] * 4
total_losses = [0.0] * 4
total_games_won = [0.0] * 4

# Training loop
for epoch in range(num_epochs):
    # Deal cards and set up game
    game = Game()
    game.deal_cards(playerList)
    game.starting_player = random.randint(0, 3)
    game.current_player = game.starting_player
    game.round = 1
    losses = []
    round_losses = [0.0] * 4

    # Play all rounds in a game
    for _ in range(13):
        losses, round_winner = game.play_round(playerList)
        # Update each player's model after a trick
        for i, player in enumerate(playerList):
            if i == 0 or i == 2:
                optimizer = optimizers[i]
                optimizer.zero_grad()

                # Convert loss to a tensor and ensure it requires gradient
                player_loss = torch.tensor([losses[i]], requires_grad=True)
                player_loss.backward()

                optimizer.step()

        game.starting_player = round_winner
        game.current_player = round_winner

        for i in range(4):
            total_losses[i] += losses[i]
            round_losses[i] += losses[i]

    game_winner = round_losses.index(min(round_losses))
    total_games_won[game_winner] += 1

    # Optional: Print epoch number and losses for tracking
    if epoch % 50 == 0:

        print(f"Epoch {epoch}, Losses: {total_losses}")
        # print("Amount of wrongly played cards:", played_wrong_cards)
        print("Total games won:", total_games_won, "\n")
        # played_wrong_cards = [0.0] * 4
        total_losses = [0.0] * 4
        total_games_won = [0.0] * 4

# torch.save(player_1.state_dict(), 'player_1_weights.pth')
# torch.save(player_2.state_dict(), 'player_2_weights.pth')
# torch.save(player_3.state_dict(), 'player_3_weights.pth')
# torch.save(player_4.state_dict(), 'player_4_weights.pth')
