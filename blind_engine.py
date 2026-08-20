"""A second, independent Connect 4 engine. Verification artifact, not a deliverable.

Written from the referee's Java source in the referee's own idiom -- a 7x9 grid of
characters, a chosen column scanned downward for the first free row, and win
detection by walking outward from the cell that just changed in each of the four
directions -- so it shares no representation and no line table with napkin_c4.py's
bitboards. Two implementations agreeing on a million random plies is evidence; one
implementation agreeing with itself is not.

Interface deliberately matches napkin_c4.Engine so `fuzz --other` can drive both:
valid_actions() -> set, play(action), game_over, winner, current_player.
Action 9 is the STEAL (the referee's -2).
"""

NUM_ROWS = 7
NUM_COLS = 9
STEAL = 9
EMPTY = "."
DRAW = "d"


class Engine:
    def __init__(self):
        self.grid = [[EMPTY] * NUM_COLS for _ in range(NUM_ROWS)]
        self.turn_index = 0
        self.result = EMPTY          # EMPTY / "0" / "1" / "d"
        self.stole = False
        self.recent = (0, 0)

    # -- read interface ------------------------------------------------------

    @property
    def current_player(self):
        return self.turn_index % 2

    @property
    def game_over(self):
        return self.result != EMPTY

    @property
    def winner(self):
        if self.result in ("0", "1"):
            return int(self.result)
        return -1

    def valid_actions(self):
        if self.game_over:
            return set()
        out = {c for c in range(NUM_COLS) if self.grid[0][c] == EMPTY}
        if self.turn_index == 1:
            out.add(STEAL)
        return out

    # -- transition ----------------------------------------------------------

    def _bottom_free_row(self, col):
        for r in range(NUM_ROWS):
            if self.grid[r][col] != EMPTY:
                return r - 1
        return NUM_ROWS - 1

    def play(self, action):
        if action not in self.valid_actions():
            raise ValueError(f"invalid action {action}")

        if action == STEAL:
            self.stole = True
            r, c = self.recent
            self.grid[r][c] = "1"
        else:
            r = self._bottom_free_row(action)
            c = action
            self.grid[r][c] = "0" if self.turn_index % 2 == 0 else "1"

        self.recent = (r, c)
        if self._connected_four(r, c):
            self.result = self.grid[r][c]
        self.turn_index += 1
        if (self.turn_index >= NUM_ROWS * NUM_COLS + (1 if self.stole else 0)
                and self.result == EMPTY):
            self.result = DRAW

    def _run(self, r, c, dr, dc):
        """How many matching cells continue from (r, c) in one direction."""
        who = self.grid[r][c]
        n = 0
        while True:
            r += dr
            c += dc
            if not (0 <= r < NUM_ROWS and 0 <= c < NUM_COLS):
                return n
            if self.grid[r][c] != who:
                return n
            n += 1

    def _connected_four(self, r, c):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            fwd = self._run(r, c, dr, dc)
            back = self._run(r, c, -dr, -dc)
            if fwd + back >= 3:
                return True
        return False
