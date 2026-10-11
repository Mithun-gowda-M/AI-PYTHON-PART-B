"""Connect Four agent: iterative-deepening minimax with alpha-beta pruning."""

from dataclasses import dataclass
from time import monotonic

ROWS, COLS, CONNECT = 6, 7, 4
WIN_SCORE = 1_000_000


@dataclass
class Entry:
    depth: int
    score: int
    flag: str  # EXACT, LOWER, UPPER
    move: int | None


class ConnectFour:
    def __init__(self) -> None:
        self.board = [[0] * COLS for _ in range(ROWS)]
        self.heights = [0] * COLS
        self.nodes = 0
        self.table: dict[tuple[tuple[int, ...], int], Entry] = {}
        self.deadline = inf = float("inf")

    def key(self, player: int) -> tuple[tuple[int, ...], int]:
        return tuple(cell for row in self.board for cell in row), player

    def play(self, col: int, player: int) -> None:
        row = ROWS - 1 - self.heights[col]
        self.board[row][col] = player
        self.heights[col] += 1

    def undo(self, col: int) -> None:
        self.heights[col] -= 1
        self.board[ROWS - 1 - self.heights[col]][col] = 0

    def winning(self, player: int) -> bool:
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r][c] != player:
                    continue
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    if all(0 <= r + i * dr < ROWS and 0 <= c + i * dc < COLS
                           and self.board[r + i * dr][c + i * dc] == player for i in range(CONNECT)):
                        return True
        return False

    def evaluate(self, player: int) -> int:
        opponent = 3 - player
        score = 0
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r][c] == player:
                    score += max(0, 3 - abs(c - COLS // 2)) * 4
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(CONNECT)]
                    if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                        window = [self.board[rr][cc] for rr, cc in cells]
                        mine, theirs, empty = window.count(player), window.count(opponent), window.count(0)
                        if theirs == 0:
                            score += (0, 1, 8, 60, WIN_SCORE)[mine]
                        if mine == 0:
                            score -= (0, 2, 12, 90, WIN_SCORE)[theirs]
        return score

    def ordered_moves(self, preferred: int | None = None) -> list[int]:
        moves = sorted((c for c in range(COLS) if self.heights[c] < ROWS), key=lambda c: abs(c - COLS // 2))
        return ([preferred] if preferred in moves else []) + [c for c in moves if c != preferred]

    def search(self, depth: int, alpha: int, beta: int, player: int, root: int, ply: int) -> tuple[int, int | None]:
        self.nodes += 1
        if monotonic() >= self.deadline:
            raise TimeoutError
        if self.winning(3 - player):
            return -WIN_SCORE + ply, None
        if depth == 0 or all(h == ROWS for h in self.heights):
            return self.evaluate(root), None
        key = self.key(player)
        original_alpha, original_beta = alpha, beta
        entry = self.table.get(key)
        preferred = entry.move if entry else None
        if entry and entry.depth >= depth:
            if entry.flag == "EXACT":
                return entry.score, entry.move
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.score)
            else:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score, entry.move
        maximizing = player == root
        best_score, best_move = (-WIN_SCORE * 2, None) if maximizing else (WIN_SCORE * 2, None)
        for col in self.ordered_moves(preferred):
            self.play(col, player)
            try:
                score, _ = self.search(depth - 1, alpha, beta, 3 - player, root, ply + 1)
            finally:
                self.undo(col)
            if maximizing and score > best_score or not maximizing and score < best_score:
                best_score, best_move = score, col
            if maximizing:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)
            if alpha >= beta:
                break
        flag = "UPPER" if best_score <= original_alpha else "LOWER" if best_score >= original_beta else "EXACT"
        self.table[key] = Entry(depth, best_score, flag, best_move)
        return best_score, best_move

    def choose(self, player: int, max_depth: int = 7, time_limit: float = 2.0) -> tuple[int, int, int]:
        self.deadline = monotonic() + time_limit
        best_move = self.ordered_moves()[0]
        best_score, completed_depth = 0, 0
        for depth in range(1, max_depth + 1):
            try:
                score, move = self.search(depth, -WIN_SCORE * 2, WIN_SCORE * 2, player, player, 0)
            except TimeoutError:
                break
            if move is not None:
                best_move, best_score, completed_depth = move, score, depth
        return best_move, best_score, completed_depth


def demo() -> None:
    game = ConnectFour()
    # A tactical position: the agent must notice the immediate horizontal win.
    for col, player in [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (5, 2)]:
        game.play(col, player)
    move, score, depth = game.choose(player=1, max_depth=8, time_limit=3.0)
    print(f"AI chooses column {move}; score={score}; completed depth={depth}; nodes={game.nodes}")
    game.play(move, 1)
    print("Winning move found:", game.winning(1))


if __name__ == "__main__":
    demo()
