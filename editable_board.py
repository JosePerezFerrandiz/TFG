
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List, Tuple
import chess


UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

PIECE_MENU_ITEMS: List[Tuple[str, str]] = [
    ("Peón blanco", "P"),
    ("Caballo blanco", "N"),
    ("Alfil blanco", "B"),
    ("Torre blanca", "R"),
    ("Dama blanca", "Q"),
    ("Rey blanco", "K"),
    ("Peón negro", "p"),
    ("Caballo negro", "n"),
    ("Alfil negro", "b"),
    ("Torre negra", "r"),
    ("Dama negra", "q"),
    ("Rey negro", "k"),
]


class EditableBoard(ttk.Frame):
    '''
    Tablero responsive con dos modos:
    - analysis: movimientos legales de ajedrez
    - edit: edición libre de la posición

    En modo edit:
    - clic izquierdo: mover pieza libremente
    - clic derecho: menú contextual para borrar o cambiar la pieza de una casilla

    Callback:
        on_position_changed(new_fen, action)
    '''

    def __init__(
        self,
        parent,
        on_position_changed: Optional[Callable[[str, str], None]] = None,
        square_size: int = 72,
        min_square_size: int = 34,
        max_square_size: int = 96,
        light_color: str = "#f0d9b5",
        dark_color: str = "#b58863",
        selected_color: str = "#d6f36a",
        legal_dot_color: str = "#2f855a",
        last_move_color: str = "#f6f669",
        board_padding: int = 10,
    ):
        super().__init__(parent)

        self.base_square_size = square_size
        self.square_size = square_size
        self.min_square_size = min_square_size
        self.max_square_size = max_square_size
        self.board_padding = board_padding

        self.light_color = light_color
        self.dark_color = dark_color
        self.selected_color = selected_color
        self.legal_dot_color = legal_dot_color
        self.last_move_color = last_move_color

        self.on_position_changed = on_position_changed

        self.board = chess.Board()
        self.initial_fen = self.board.fen()

        self.edit_mode = False
        self.selected_square: Optional[chess.Square] = None
        self.legal_targets: List[chess.Square] = []
        self.last_move: Optional[chess.Move] = None
        self.edit_history: List[str] = []

        self.board_origin_x = self.board_padding
        self.board_origin_y = self.board_padding
        self.board_px = 8 * self.square_size

        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.context_menu = tk.Menu(self, tearoff=0)

        self.draw_board()

    def set_fen(self, fen: str) -> None:
        self.board = chess.Board(fen)
        self.initial_fen = fen
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.edit_history = []
        self.draw_board()

    def get_fen(self) -> str:
        return self.board.fen()

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = enabled
        self.selected_square = None
        self.legal_targets = []
        self.draw_board()

    def toggle_edit_mode(self) -> bool:
        self.set_edit_mode(not self.edit_mode)
        return self.edit_mode

    def reset_to_initial_position(self) -> None:
        self.board = chess.Board(self.initial_fen)
        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.edit_history = []
        self.draw_board()
        self._emit_change("reset")

    def undo_last_change(self) -> bool:
        if self.edit_mode:
            if not self.edit_history:
                return False
            fen = self.edit_history.pop()
            self.board = chess.Board(fen)
            self.selected_square = None
            self.legal_targets = []
            self.draw_board()
            self._emit_change("undo")
            return True

        if not self.board.move_stack:
            return False

        self.board.pop()
        self.last_move = self.board.peek() if self.board.move_stack else None
        self.selected_square = None
        self.legal_targets = []
        self.draw_board()
        self._emit_change("undo")
        return True

    def _emit_change(self, action: str) -> None:
        if self.on_position_changed:
            self.on_position_changed(self.board.fen(), action)

    def _save_edit_snapshot(self) -> None:
        self.edit_history.append(self.board.fen())

    def _on_canvas_resize(self, _event=None) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())

        usable = min(width, height) - 2 * self.board_padding
        usable = max(8 * self.min_square_size, usable)

        new_square_size = max(
            self.min_square_size,
            min(self.max_square_size, usable // 8)
        )

        self.square_size = new_square_size
        self.board_px = 8 * self.square_size
        self.board_origin_x = max(self.board_padding, (width - self.board_px) // 2)
        self.board_origin_y = max(self.board_padding, (height - self.board_px) // 2)
        self.draw_board()

    def draw_board(self) -> None:
        self.canvas.delete("all")

        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.board_px = 8 * self.square_size
        self.board_origin_x = max(self.board_padding, (width - self.board_px) // 2)
        self.board_origin_y = max(self.board_padding, (height - self.board_px) // 2)

        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                x1, y1, x2, y2 = self._square_bbox(square)

                fill = self.light_color if (row + col) % 2 == 0 else self.dark_color

                if not self.edit_mode and self.last_move and square in (
                    self.last_move.from_square,
                    self.last_move.to_square,
                ):
                    fill = self.last_move_color

                if self.selected_square == square:
                    fill = self.selected_color

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="")

                if col == 0:
                    self.canvas.create_text(
                        x1 + max(8, self.square_size * 0.14),
                        y1 + max(10, self.square_size * 0.16),
                        text=str(8 - row),
                        font=("Arial", max(8, int(self.square_size * 0.14)), "bold"),
                        anchor="w",
                    )
                if row == 7:
                    self.canvas.create_text(
                        x2 - max(8, self.square_size * 0.14),
                        y2 - max(8, self.square_size * 0.14),
                        text=chr(ord("a") + col),
                        font=("Arial", max(8, int(self.square_size * 0.14)), "bold"),
                        anchor="e",
                    )

                if square in self.legal_targets:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    radius = max(5, self.square_size // 9)
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
                        font=("Segoe UI Symbol", max(16, int(self.square_size * 0.52))),
                    )

        #mode_text = "Modo corrección" if self.edit_mode else "Modo análisis"
        #self.canvas.create_text(
        #    width - 10,
        #    10,
        #    text=mode_text,
        #    anchor="ne",
        #    font=("Segoe UI", 10, "bold"),
        #)

    def _on_left_click(self, event) -> None:
        clicked_square = self._coords_to_square(event.x, event.y)
        if clicked_square is None:
            return

        if self.edit_mode:
            self._handle_edit_left_click(clicked_square)
        else:
            self._handle_analysis_left_click(clicked_square)

    def _on_right_click(self, event) -> None:
        if not self.edit_mode:
            return

        clicked_square = self._coords_to_square(event.x, event.y)
        if clicked_square is None:
            return

        self._show_context_menu(clicked_square, event.x_root, event.y_root)

    def _handle_analysis_left_click(self, clicked_square: chess.Square) -> None:
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
            self._emit_change("legal_move")
        else:
            self.selected_square = None
            self.legal_targets = []
            self.draw_board()

    def _handle_edit_left_click(self, clicked_square: chess.Square) -> None:
        clicked_piece = self.board.piece_at(clicked_square)

        if self.selected_square is None:
            if clicked_piece:
                self.selected_square = clicked_square
                self.draw_board()
            return

        if clicked_square == self.selected_square:
            self.selected_square = None
            self.draw_board()
            return

        piece = self.board.piece_at(self.selected_square)
        if piece is None:
            self.selected_square = None
            self.draw_board()
            return

        self._save_edit_snapshot()
        self.board.remove_piece_at(self.selected_square)
        self.board.set_piece_at(clicked_square, piece)

        self.selected_square = None
        self.legal_targets = []
        self.last_move = None
        self.draw_board()
        self._emit_change("edit_move")

    def _show_context_menu(self, square: chess.Square, x_root: int, y_root: int) -> None:
        self.context_menu.delete(0, "end")

        self.context_menu.add_command(
            label="Borrar pieza",
            command=lambda sq=square: self._clear_square(sq),
        )
        self.context_menu.add_separator()

        for label, symbol in PIECE_MENU_ITEMS:
            self.context_menu.add_command(
                label=label,
                command=lambda sq=square, sym=symbol: self._set_piece_symbol(sq, sym),
            )

        self.context_menu.tk_popup(x_root, y_root)

    def _clear_square(self, square: chess.Square) -> None:
        self._save_edit_snapshot()
        self.board.remove_piece_at(square)
        self.selected_square = None
        self.draw_board()
        self._emit_change("edit_clear_square")

    def _set_piece_symbol(self, square: chess.Square, symbol: str) -> None:
        self._save_edit_snapshot()
        piece = chess.Piece.from_symbol(symbol)
        self.board.set_piece_at(square, piece)
        self.selected_square = None
        self.draw_board()
        self._emit_change("edit_set_piece")

    def _coords_to_square(self, x: int, y: int) -> Optional[chess.Square]:
        if not (
            self.board_origin_x <= x < self.board_origin_x + self.board_px and
            self.board_origin_y <= y < self.board_origin_y + self.board_px
        ):
            return None

        rel_x = x - self.board_origin_x
        rel_y = y - self.board_origin_y

        col = rel_x // self.square_size
        row = rel_y // self.square_size

        if not (0 <= col < 8 and 0 <= row < 8):
            return None

        return chess.square(int(col), 7 - int(row))

    def _square_bbox(self, square: chess.Square):
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        x1 = self.board_origin_x + col * self.square_size
        y1 = self.board_origin_y + row * self.square_size
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
    def on_changed(new_fen: str, action: str):
        print("Acción:", action)
        print("Nueva FEN:", new_fen)

    root = tk.Tk()
    root.title("Demo EditableBoard")
    root.geometry("760x820")

    board_widget = EditableBoard(root, on_position_changed=on_changed, square_size=72)
    board_widget.pack(fill="both", expand=True, padx=10, pady=10)

    controls = ttk.Frame(root)
    controls.pack(fill="x", padx=10, pady=(0, 10))

    ttk.Button(
        controls,
        text="Modo corrección",
        command=board_widget.toggle_edit_mode
    ).pack(side="left", padx=(0, 8))

    ttk.Button(
        controls,
        text="Volver a inicio",
        command=board_widget.reset_to_initial_position
    ).pack(side="left", padx=(0, 8))

    ttk.Button(
        controls,
        text="Deshacer",
        command=board_widget.undo_last_change
    ).pack(side="left")

    root.mainloop()
