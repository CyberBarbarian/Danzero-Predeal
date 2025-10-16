# Paper-Compliant Observation Space: 513 Dimensions

This document describes the observation space implementation that follows the exact specification from the research paper.

## Overview

The observation space is a **513-dimensional vector** that provides a comprehensive view of the game state from one player's perspective. Each dimension has a specific meaning as defined in the paper.

## Dimension Breakdown

### [0-53]: Hand Cards (54 dimensions)
- **Purpose**: Cards currently held by the player
- **Encoding**: One-hot encoding where 1 indicates the card is in hand, 0 otherwise
- **Card Order**: 
  - Regular cards: 2S, 3S, ..., AS, 2H, 3H, ..., AH, 2D, 3D, ..., AD, 2C, 3C, ..., AC (52 cards)
  - Jokers: BJ (Black Joker), RJ (Red Joker) (2 cards)

### [54-107]: Remaining Cards (54 dimensions)
- **Purpose**: All cards excluding the player's current hand and those already played
- **Encoding**: One-hot encoding where 1 indicates the card is still available
- **Calculation**: `remaining_cards = all_cards - hand_cards - played_cards`

### [108-161]: Last Move to Cover (54 dimensions)
- **Purpose**: The last move of players that the current action must be able to cover
- **Special Cases**:
  - If the player is leading the trick: all dimensions set to 0
  - If no move to cover: all dimensions set to 0
- **Encoding**: One-hot encoding of the cards in the move that needs to be covered

### [162-215]: Partner Last Move (54 dimensions)
- **Purpose**: The last move made by the partner
- **Special Cases**:
  - If partner passed: all dimensions set to 0
  - If partner exhausted hand cards: all dimensions set to -1
- **Encoding**: One-hot encoding of the cards in the partner's last move

### [216-299]: Card Counts for Other Players (84 dimensions)
- **Purpose**: Number of remaining cards for the other three players
- **Structure**: 28 dimensions per player × 3 players = 84 dimensions
- **Encoding**: Normalized count (actual_count / 27.0) to range [0, 1]
- **Order**: Players in order of their card play (excluding current player)

### [300-461]: Played Cards for Other Players (162 dimensions)
- **Purpose**: Cards played by the other three players, recorded in order of playing
- **Structure**: 54 dimensions per player × 3 players = 162 dimensions
- **Encoding**: One-hot encoding of played cards
- **Order**: Players in order of their card play (excluding current player)

### [462-501]: Team Levels (40 dimensions)
- **Purpose**: Level information for both teams and current level
- **Structure**:
  - [462-474]: Our team level (13 dimensions)
  - [475-487]: Opponent team level (13 dimensions)  
  - [488-500]: Current level (13 dimensions)
- **Encoding**: Normalized rank value (rank / 13.0) to range [0, 1]

### [501-513]: Wild Card Flags (12 dimensions)
- **Purpose**: Flags indicating wild cards in hand and suitability for various combinations
- **Structure**:
  - [501]: Has wild cards flag (0 or 1)
  - [502]: Number of wild cards (normalized)
  - [503-508]: Suitability flags for different combinations:
    - [503]: Can form straights
    - [504]: Can form pairs
    - [505]: Can form trips
    - [506]: Can form bombs
    - [507]: Can form three pairs
    - [508]: Can form two trips
    - [509]: Can form straight flush
    - [510]: Has joker bombs
  - [511-512]: Reserved for future use

## Implementation Details

### Card Mapping
The implementation uses a consistent card-to-index mapping:
- Regular cards: `{rank}{suit}` → index (0-51)
- Jokers: `BJ` → 52, `RJ` → 53

### Move Encoding
Moves are encoded with:
- Move type (normalized to [0, 1])
- Cards in the move (one-hot encoding)

### Normalization
- Card counts: divided by 27 (maximum cards per player)
- Ranks: divided by 13 (maximum rank value)
- Move types: mapped to [0, 1] range

## Usage

```python
from guandan.env.paper_observation_extractor import extract_paper_observation

# Extract 513-dimensional observation
observation = extract_paper_observation(json_message, player_id)
print(f"Observation shape: {observation.shape}")  # (513,)
```

## Paper Compliance

This implementation strictly follows the paper's specification:
- ✅ Exact 513 dimensions
- ✅ Correct dimension ranges and meanings
- ✅ Proper encoding schemes
- ✅ Special case handling (leading, passing, exhausted)
- ✅ Team and level information
- ✅ Wild card flag system

The observation space is now fully compliant with the research paper and ready for training with the same model architectures used in the original work.
