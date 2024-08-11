# Blackjack Reinforcement Learning Agent

This project is an implementation template for a Blackjack agent using reinforcement learning, designed for the Gymnasium Blackjack environment. The CI script automatically sends your results to a ranking server for comparison. Only modify the `blackjack_agent.py` file, specifically the `BlackjackAgent` methods `action()` and `learn()`.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/aaviix/Blackjack-Reinforcement-Learning-Agent.git
    cd Blackjack-Reinforcement-Learning-Agent
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements-interactive.txt
    ```

## Usage

- Implement the `BlackjackAgent` methods in `blackjack_agent.py`.
- Use `blackjack_trainer.py` to train your agent.
- Run `blackjack_play_human.py` to interact with the trained agent as a human player.

## Files

- `blackjack_agent.py`: Implement your agent here.
- `blackjack_trainer.py`: Script to train the agent.
- `blackjack_play_human.py`: Script for human interaction with the agent.
- `benchmark.py`: Script for benchmarking agent performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.