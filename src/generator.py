import chess
import chess.engine
import numpy as np
from tools import get_date, to_bitboard, save


# TODO: Do not hardcode this
engine_path = "C:/Users/sd005/dev/misc/stockfish_14/stockfish_14_x64_avx2.exe"

WHITE_WON = '1-0'
BLACK_WON = '0-1'
DRAW = '1/2-1/2'

results = {WHITE_WON: False, BLACK_WON: True}


def generate_dataset(n_games, skill=20, threads=1):
    def play(engine, limit=0.01):
        board = chess.Board()
        # raw_history = []
        history = []
        while not board.is_game_over():
            result = engine.play(board, chess.engine.Limit(time=limit))
            # raw_history.append(board.copy())
            history.append(to_bitboard(board, pack=True))
            board.push(result.move)
        return history, board.result()

    def create_empty_games():
        return {WHITE_WON: [], BLACK_WON: []}

    engine = chess.engine.SimpleEngine.popen_uci(engine_path, timeout=None)
    engine.configure({"Threads": threads, "Skill Level": skill})

    games = create_empty_games()
    print('playing....')

    i = 1
    while i != n_games + 1:
        history, result = play(engine)

        if result != DRAW:
            print(f'.....{i}/{n_games}')
            games[result].append((history, results[result]))

            if i % 1000 == 0:
                filename = f'dataset-{i // 1000}'
                date = get_date()

                save(games[WHITE_WON],
                     f'{filename}-white-{len(games[WHITE_WON])}-{date}')
                save(games[BLACK_WON],
                     f'{filename}-black-{len(games[BLACK_WON])}-{date}')

                games = create_empty_games()

            i += 1

    engine.quit()


if __name__ == '__main__':
    generate_dataset(3000, threads=12)
