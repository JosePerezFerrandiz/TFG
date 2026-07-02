
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List
import chess


UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


class InteractiveBoard(ttk.Frame):
    """
    Tablero interactivo para Tkinter basado en python-chess.

    Uso típico:
        board_widget = InteractiveBoard(parent, on_position_changed=callback)
        board_widget.pack(fill="both", expand=True)
        board_widget.set_fen(fen_detectado)

    callback(new_fen, move_uci) se ejecuta cuando el usuario hace un movimiento legal.
    """

    def __init__(
        self,
        parent,
        on_position_changed: Optional[Callable[[str, str], None]] = None,
        square_size: int = 72,
        light_color: str = "#f0d9b5",
        dark_color: str = "#b58863",
        selected_color: str = "#d6f36a",
        legal_dot_color: str = "#2f855a",
        last_move_color: str = "#f6f669",
    ):
        super().__init__(parent)

        self.square_size = square_size
        self.light_color = light_color
        self.dark_color = dark_color
        self.selected_color = selected_color
        self.legal_dot_color = legal_dot_color
        self.last_move_color = last_move_color

        self.on_position_changed = on_position_changed

        self.board = chess.Board()
        self.selected_square: Optional[chess.Square] = None
        self.legal_targets: List[chess.Square] = []
        self.last_move: Optional[chess.Move] = None

        board_px = 8 * self.square_size
        self.canvas = tk.Canvas(
            self,
            width=board_px,
            height=board_px,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self._on_click)

        self.draw_board()

    def set_fen(self, fen: str) -> None:
        self.board = chess.Board(fen)
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.draw_board()

    def get_fen(self) -> str:
        return self.board.fen()

    def reset_start_position(self) -> None:
        self.board = chess.Board()
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.draw_board()

    def set_position_from_board(self, board: chess.Board) -> None:
        self.board = board.copy(stack=True)
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.draw_board()

    def apply_uci_move(self, move_uci: str) -> bool:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return False

        if move not in self.board.legal_moves:
            return False

        self.board.push(move)
        self.last_move = move
        self.selected_square = None
        self.legal_targets = []
        self.draw_board()

        if self.on_position_changed:
            self.on_position_changed(self.board.fen(), move.uci())

        return True

    def undo_last_move(self) -> bool:
        if not self.board.move_stack:
            return False

        self.board.pop()
        self.last_move = self.board.peek() if self.board.move_stack else None
        self.selected_square = None
        self.legal_targets = []
        self.draw_board()
        return True

    def draw_board(self) -> None:
        self.canvas.delete("all")

        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                x1, y1, x2, y2 = self._square_bbox(square)

                fill = self.light_color if (row + col) % 2 == 0 else self.dark_color

                if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
                    fill = self.last_move_color

                if self.selected_square == square:
                    fill = self.selected_color

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="")

                if col == 0:
                    self.canvas.create_text(
                        x1 + 10,
                        y1 + 12,
                        text=str(8 - row),
                        font=("Arial", 9, "bold"),
                        anchor="w",
                    )
                if row == 7:
                    self.canvas.create_text(
                        x2 - 10,
                        y2 - 10,
                        text=chr(ord("a") + col),
                        font=("Arial", 9, "bold"),
                        anchor="e",
                    )

                if square in self.legal_targets:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    radius = max(6, self.square_size // 9)
                    self.canvas.create_oval(
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                        fill=self.legal_dot_color,
                        outline="",
                    )

                piece = self.board.piece_at(square)
                if piece:
                    symbol = UNICODE_PIECES.get(piece.symbol(), piece.symbol())
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=symbol,
                        font=("Segoe UI Symbol", max(18, int(self.square_size * 0.52))),
                    )

    def _on_click(self, event) -> None:
        clicked_square = self._coords_to_square(event.x, event.y)
        if clicked_square is None:
            return

        clicked_piece = self.board.piece_at(clicked_square)

        if self.selected_square is None:
            if clicked_piece and clicked_piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.legal_targets = self._legal_targets_for(clicked_square)
                self.draw_board()
            return

        if clicked_square == self.selected_square:
            self.selected_square = None
            self.legal_targets = []
            self.draw_board()
            return

        if clicked_piece and clicked_piece.color == self.board.turn:
            self.selected_square = clicked_square
            self.legal_targets = self._legal_targets_for(clicked_square)
            self.draw_board()
            return

        move = self._build_move(self.selected_square, clicked_square)
        if move and move in self.board.legal_moves:
            self.board.push(move)
            self.last_move = move
            self.selected_square = None
            self.legal_targets = []
            self.draw_board()

            if self.on_position_changed:
                self.on_position_changed(self.board.fen(), move.uci())
        else:
            self.selected_square = None
            self.legal_targets = []
            self.draw_board()

    def _coords_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        board_px = 8 * self.square_size
        if not (0 <= x < board_px and 0 <= y < board_px):
            return None

        col = x // self.square_size
        row = y // self.square_size
        return chess.square(col, 7 - row)

    def _square_bbox(self, square: chess.Square):
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        x1 = col * self.square_size
        y1 = row * self.square_size
        x2 = x1 + self.square_size
        y2 = y1 + self.square_size
        return x1, y1, x2, y2

    def _legal_targets_for(self, from_square: chess.Square) -> List[chess.Square]:
        targets = []
        for move in self.board.legal_moves:
            if move.from_square == from_square:
                targets.append(move.to_square)
        return targets

    def _build_move(self, from_square: chess.Square, to_square: chess.Square) -> Optional[chess.Move]:
        piece = self.board.piece_at(from_square)
        if piece is None:
            return None

        if piece.piece_type == chess.PAWN:
            target_rank = chess.square_rank(to_square)
            if (piece.color == chess.WHITE and target_rank == 7) or (
                piece.color == chess.BLACK and target_rank == 0
            ):
                return chess.Move(from_square, to_square, promotion=chess.QUEEN)

        return chess.Move(from_square, to_square)


if __name__ == "__main__":
    def on_changed(new_fen: str, move_uci: str):
        print("Movimiento:", move_uci)
        print("Nueva FEN:", new_fen)

    root = tk.Tk()
    root.title("Demo InteractiveBoard")

    board_widget = InteractiveBoard(root, on_position_changed=on_changed, square_size=72)
    board_widget.pack(padx=10, pady=10)

    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=10, pady=(0, 10))

    ttk.Button(
        controls,
        text="Posición inicial",
        command=board_widget.reset_start_position
    ).pack(side="left", padx=(0, 8))

    ttk.Button(
        controls,
        text="Deshacer",
        command=board_widget.undo_last_move
    ).pack(side="left")

    root.mainloop()
