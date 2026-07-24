
import os
import sys
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import chess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from editable_board import EditableBoard
from lichessAPI import LichessAPI


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = "extracted-data"


def read_text_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TFG Ajedrez - Análisis desde imagen")
        self.root.geometry("1500x950")

        self.image_path_var = tk.StringVar()
        self.turn_var = tk.StringVar(value="Automático")
        self.rot_display_var = tk.StringVar(value="Sin rotación")
        self.graph_source_var = tk.StringVar(value="Lichess")
        self.table_source_var = tk.StringVar(value="Lichess")
        self.status_var = tk.StringVar(value="Listo")
        self.opening_title_var = tk.StringVar(value="Apertura: -")

        self.rot_options = {
            "Sin rotación": "none",
            "90° horario": "cw",
            "90° antihorario": "ccw",
            "180°": "180",
        }

        self.current_report = None
        self.detected_report = None
        self.current_fen = None
        self.detected_fen = None
        self.current_image_path = None
        self.last_detected_opening_text = None
        self.preview_photo = None

        self.pie_canvas = None
        self.bar_canvas = None
        self.graphs_window_id = None
        self._resize_graph_job = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Imagen:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top_frame, textvariable=self.image_path_var, width=90).grid(
            row=0, column=1, padx=5, sticky="ew"
        )
        ttk.Button(top_frame, text="Seleccionar", command=self.select_image).grid(
            row=0, column=2, padx=5
        )

        ttk.Label(top_frame, text="Turno:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        turn_combo = ttk.Combobox(
            top_frame,
            textvariable=self.turn_var,
            values=["Automático", "Blancas", "Negras"],
            state="readonly",
            width=20,
        )
        turn_combo.grid(row=1, column=1, sticky="w", pady=(10, 0))
        turn_combo.current(0)

        ttk.Label(top_frame, text="Rotación del tablero:").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        rot_combo = ttk.Combobox(
            top_frame,
            textvariable=self.rot_display_var,
            values=list(self.rot_options.keys()),
            state="readonly",
            width=20,
        )
        rot_combo.grid(row=2, column=1, sticky="w", pady=(10, 0))
        rot_combo.current(0)

        self.analyze_image_button = ttk.Button(
            top_frame,
            text="Analizar imagen",
            command=self.run_analysis,
        )
        self.analyze_image_button.grid(row=1, column=2, rowspan=2, padx=5, pady=(10, 0))

        top_frame.columnconfigure(1, weight=1)

        opening_frame = ttk.Frame(self.root, padding=(10, 2, 10, 8))
        opening_frame.pack(fill="x")

        ttk.Label(
            opening_frame,
            text="Apertura detectada",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="center")

        self.opening_title_label = ttk.Label(
            opening_frame,
            textvariable=self.opening_title_var,
            font=("Segoe UI", 16, "bold"),
            anchor="center",
            justify="center",
        )
        self.opening_title_label.pack(fill="x", pady=(2, 0))

        middle_frame = ttk.Panedwindow(self.root, orient="horizontal")
        middle_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(middle_frame, padding=5)
        right_frame = ttk.Frame(middle_frame, padding=5)

        middle_frame.add(left_frame, weight=3)
        middle_frame.add(right_frame, weight=5)

        # ---------------- Izquierda: tablero editable/interactivo ----------------
        board_top = ttk.Frame(left_frame)
        board_top.pack(fill="x", pady=(0, 8))

        self.undo_button = ttk.Button(
            board_top,
            text="Deshacer",
            command=self.undo_board_change,
            state="disabled",
        )
        self.undo_button.pack(side="left")

        self.reset_board_button = ttk.Button(
            board_top,
            text="Volver a posición inicial",
            command=self.reset_interactive_board,
            state="disabled",
        )
        self.reset_board_button.pack(side="left", padx=(8, 0))

        self.correct_button = ttk.Button(
            board_top,
            text="Corregir posiciones",
            command=self.toggle_correction_mode,
            state="disabled",
        )
        self.correct_button.pack(side="left", padx=(8, 0))

        self.reanalyze_button = ttk.Button(
            board_top,
            text="Analizar nueva posición",
            command=self.reanalyze_current_board,
            state="disabled",
        )
        self.reanalyze_button.pack(side="left", padx=(8, 0))

        self.view_image_button = ttk.Button(
            board_top,
            text="Ver imagen",
            command=self.show_compare_window,
            state="disabled",
        )
        self.view_image_button.pack(side="left", padx=(8, 0))

        self.board_help_label = ttk.Label(
            left_frame,
            text="Modo análisis: clic en una pieza y luego en la casilla destino.",
            wraplength=520,
            justify="left",
        )
        self.board_help_label.pack(anchor="w", pady=(0, 6))

        self.board_container = ttk.Frame(left_frame)
        self.board_container.pack(fill="both", expand=True)

        self.preview_label = ttk.Label(
            self.board_container,
            anchor="center",
            text="Selecciona una imagen para comenzar",
        )
        self.preview_label.pack(fill="both", expand=True)

        self.interactive_board = EditableBoard(
            self.board_container,
            on_position_changed=self.on_board_position_changed,
            square_size=68,
        )

        self.current_fen_label = ttk.Label(
            left_frame,
            text="FEN actual: -",
            wraplength=520,
            justify="left",
        )
        self.current_fen_label.pack(fill="x", pady=(8, 0))

        # ---------------- Derecha: tablas y gráficas ----------------
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill="both", expand=True)

        self.tables_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tables_tab, text="Tablas")

        self.graphs_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.graphs_tab, text="Gráficas")

        self._build_tables_tab()
        self._build_graphs_tab()

        bottom_status = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        bottom_status.pack(fill="x")
        ttk.Label(bottom_status, textvariable=self.status_var).pack(side="left")

    def _build_tables_tab(self):
        top = ttk.Frame(self.tables_tab)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Fuente jugadas:").pack(side="left")
        source_combo = ttk.Combobox(
            top,
            textvariable=self.table_source_var,
            values=["Lichess", "Maestros"],
            state="readonly",
            width=15,
        )
        source_combo.pack(side="left", padx=(6, 0))
        source_combo.current(0)
        source_combo.bind("<<ComboboxSelected>>", lambda _e: self.render_tables())

        container = ttk.Frame(self.tables_tab)
        container.pack(fill="both", expand=True)

        self.tables_canvas = tk.Canvas(container, highlightthickness=0)
        self.tables_scrollbar = ttk.Scrollbar(
            container,
            orient="vertical",
            command=self.tables_canvas.yview,
        )
        self.tables_canvas.configure(yscrollcommand=self.tables_scrollbar.set)

        self.tables_scrollbar.pack(side="right", fill="y")
        self.tables_canvas.pack(side="left", fill="both", expand=True)

        self.tables_content = ttk.Frame(self.tables_canvas)
        self.tables_window_id = self.tables_canvas.create_window(
            (0, 0),
            window=self.tables_content,
            anchor="nw",
        )

        self.tables_content.bind("<Configure>", self._on_tables_content_configure)
        self.tables_canvas.bind("<Configure>", self._on_tables_canvas_configure)

        stats_frame = ttk.LabelFrame(self.tables_content, text="Resumen de estadísticas")
        stats_frame.pack(fill="x", expand=False, pady=(0, 8))

        self.stats_tree = ttk.Treeview(
            stats_frame,
            columns=("fuente", "total", "blancas", "tablas", "negras"),
            show="headings",
            height=2,
        )
        for col, text, width in [
            ("fuente", "Fuente", 110),
            ("total", "Total partidas", 120),
            ("blancas", "Vict. blancas", 150),
            ("tablas", "Tablas", 120),
            ("negras", "Vict. negras", 150),
        ]:
            self.stats_tree.heading(col, text=text)
            self.stats_tree.column(col, width=width, anchor="center")
        self.stats_tree.pack(fill="x", expand=True, padx=5, pady=5)

        frequent_frame = ttk.LabelFrame(self.tables_content, text="Jugadas más frecuentes")
        frequent_frame.pack(fill="x", expand=False, pady=(0, 8))

        self.frequent_tree = ttk.Treeview(
            frequent_frame,
            columns=("mov", "partidas", "blancas", "tablas", "negras"),
            show="headings",
            height=5,
        )
        for col, text, width in [
            ("mov", "Movimiento", 120),
            ("partidas", "Partidas", 100),
            ("blancas", "% Blancas", 100),
            ("tablas", "% Tablas", 100),
            ("negras", "% Negras", 100),
        ]:
            self.frequent_tree.heading(col, text=text)
            self.frequent_tree.column(col, width=width, anchor="center")
        self.frequent_tree.pack(fill="x", expand=True, padx=5, pady=5)

        best_frame = ttk.LabelFrame(self.tables_content, text="Mejores jugadas")
        best_frame.pack(fill="x", expand=False, pady=(0, 8))

        self.best_tree = ttk.Treeview(
            best_frame,
            columns=("mov", "partidas", "score"),
            show="headings",
            height=5,
        )
        for col, text, width in [
            ("mov", "Movimiento", 120),
            ("partidas", "Partidas", 100),
            ("score", "Score esperado", 130),
        ]:
            self.best_tree.heading(col, text=text)
            self.best_tree.column(col, width=width, anchor="center")
        self.best_tree.pack(fill="x", expand=True, padx=5, pady=5)

        games_frame = ttk.LabelFrame(
            self.tables_content,
            text="Partidas destacadas de grandes maestros",
        )
        games_frame.pack(fill="both", expand=True, pady=(0, 8))

        games_inner = ttk.Frame(games_frame)
        games_inner.pack(fill="both", expand=True, padx=5, pady=5)

        self.games_tree = ttk.Treeview(
            games_inner,
            columns=("white", "black", "fecha", "ganador"),
            show="headings",
            height=5,
        )
        for col, text, width in [
            ("white", "Blancas", 220),
            ("black", "Negras", 220),
            ("fecha", "Fecha", 100),
            ("ganador", "Ganador", 100),
        ]:
            self.games_tree.heading(col, text=text)
            self.games_tree.column(col, width=width, anchor="center")
        games_scroll = ttk.Scrollbar(
            games_inner,
            orient="vertical",
            command=self.games_tree.yview,
        )
        self.games_tree.configure(yscrollcommand=games_scroll.set)

        self.games_tree.pack(side="left", fill="both", expand=True)
        games_scroll.pack(side="right", fill="y")

        self._bind_tables_mousewheel()

    def _build_graphs_tab(self):
        graphs_top = ttk.Frame(self.graphs_tab)
        graphs_top.pack(fill="x", pady=(0, 10))

        ttk.Label(graphs_top, text="Fuente:").pack(side="left")
        source_combo = ttk.Combobox(
            graphs_top,
            textvariable=self.graph_source_var,
            values=["Lichess", "Maestros"],
            state="readonly",
            width=15,
        )
        source_combo.pack(side="left", padx=(5, 0))
        source_combo.current(0)
        source_combo.bind("<<ComboboxSelected>>", lambda _e: self.render_graphs())

        graphs_container = ttk.Frame(self.graphs_tab)
        graphs_container.pack(fill="both", expand=True)

        self.graphs_canvas = tk.Canvas(graphs_container, highlightthickness=0)

        self.graphs_scrollbar_y = ttk.Scrollbar(
            graphs_container,
            orient="vertical",
            command=self.graphs_canvas.yview,
        )

        self.graphs_scrollbar_x = ttk.Scrollbar(
            graphs_container,
            orient="horizontal",
            command=self.graphs_canvas.xview,
        )

        self.graphs_canvas.configure(
            yscrollcommand=self.graphs_scrollbar_y.set,
            xscrollcommand=self.graphs_scrollbar_x.set,
        )

        self.graphs_canvas.grid(row=0, column=0, sticky="nsew")
        self.graphs_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.graphs_scrollbar_x.grid(row=1, column=0, sticky="ew")

        graphs_container.rowconfigure(0, weight=1)
        graphs_container.columnconfigure(0, weight=1)
        self.graphs_content = ttk.Frame(self.graphs_canvas)
        self.graphs_window_id = self.graphs_canvas.create_window(
            (0, 0),
            window=self.graphs_content,
            anchor="nw",
        )

        self.graphs_content.bind("<Configure>", self._on_graphs_content_configure)
        self.graphs_canvas.bind("<Configure>", self._on_graphs_canvas_configure)

        self.pie_frame = ttk.LabelFrame(self.graphs_content, text="Distribución de jugadas")
        self.pie_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.bar_frame = ttk.LabelFrame(self.graphs_content, text="Resultados por jugada")
        self.bar_frame.pack(fill="both", expand=True)

        self._bind_graph_mousewheel()

    # ------------------------------------------------------------------
    # Scroll handlers
    # ------------------------------------------------------------------

    def _on_tables_content_configure(self, _event=None):
        self.tables_canvas.configure(scrollregion=self.tables_canvas.bbox("all"))

    def _on_tables_canvas_configure(self, event):
        self.tables_canvas.itemconfigure(self.tables_window_id, width=event.width)

    def _on_tables_mousewheel(self, event):
        if event.delta:
            self.tables_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            self.tables_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.tables_canvas.yview_scroll(1, "units")

    def _bind_tables_mousewheel(self):
        widgets = [self.tables_canvas, self.tables_content]
        for widget in widgets:
            widget.bind("<MouseWheel>", self._on_tables_mousewheel)
            widget.bind("<Button-4>", self._on_tables_mousewheel)
            widget.bind("<Button-5>", self._on_tables_mousewheel)

    def _on_graphs_content_configure(self, _event=None):
        self.graphs_canvas.configure(scrollregion=self.graphs_canvas.bbox("all"))

    def _on_graphs_canvas_configure(self, event):
        if self.graphs_window_id is not None:
            # El contenido se adapta al ancho real de la pestaña.
            self.graphs_canvas.itemconfigure(
                self.graphs_window_id,
                width=event.width,
            )

        # Redibujar con un pequeño retardo para evitar hacerlo muchas veces seguidas.
        if self._resize_graph_job is not None:
            self.root.after_cancel(self._resize_graph_job)

        self._resize_graph_job = self.root.after(300, self._redraw_graphs_after_resize)

    def _redraw_graphs_after_resize(self):
        self._resize_graph_job = None

        if self.current_report:
            self.render_graphs()

    def _on_mousewheel_graphs(self, event):
        if event.delta:
            self.graphs_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            self.graphs_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.graphs_canvas.yview_scroll(1, "units")

    def _on_shift_mousewheel_graphs(self, event):
        if event.delta:
            self.graphs_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        elif getattr(event, "num", None) == 4:
            self.graphs_canvas.xview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.graphs_canvas.xview_scroll(1, "units")

    def _bind_graph_mousewheel(self):
        widgets = [self.graphs_canvas, self.graphs_content, self.pie_frame, self.bar_frame]

        for widget in widgets:
            widget.bind("<MouseWheel>", self._on_mousewheel_graphs)
            widget.bind("<Button-4>", self._on_mousewheel_graphs)
            widget.bind("<Button-5>", self._on_mousewheel_graphs)

            widget.bind("<Shift-MouseWheel>", self._on_shift_mousewheel_graphs)
            widget.bind("<Shift-Button-4>", self._on_shift_mousewheel_graphs)
            widget.bind("<Shift-Button-5>", self._on_shift_mousewheel_graphs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def has_undo_available(self):
        return bool(self.interactive_board.board.move_stack) or bool(self.interactive_board.edit_history)

    def set_busy(self, busy: bool, status_text: str):
        self.status_var.set(status_text)
        self.analyze_image_button.config(state="disabled" if busy else "normal")
        self.undo_button.config(state="normal" if (not busy and self.has_undo_available()) else "disabled")
        self.reset_board_button.config(state="normal" if (not busy and self.detected_fen) else "disabled")
        self.correct_button.config(state="normal" if (not busy and self.current_fen) else "disabled")
        self.reanalyze_button.config(state="normal" if (not busy and self.current_fen) else "disabled")
        self.view_image_button.config(
            state="normal" if (not busy and self.current_image_path and self.current_fen) else "disabled"
        )
        self.root.config(cursor="watch" if busy else "")
        self.root.update_idletasks()

    def refresh_board_buttons(self):
        self.undo_button.config(state="normal" if self.has_undo_available() else "disabled")
        self.reset_board_button.config(state="normal" if self.detected_fen else "disabled")
        self.correct_button.config(state="normal" if self.current_fen else "disabled")
        self.reanalyze_button.config(state="normal" if self.current_fen else "disabled")
        self.view_image_button.config(
            state="normal" if (self.current_image_path and self.current_fen) else "disabled"
        )

    def get_clean_board_fen(self):
        board = self.interactive_board.board.copy(stack=False)
        board.castling_rights = board.clean_castling_rights()
        board.ep_square = None
        return board.fen()

    def is_position_analyzable(self):
        board = self.interactive_board.board
        white_kings = len(board.pieces(chess.KING, chess.WHITE))
        black_kings = len(board.pieces(chess.KING, chess.BLACK))

        if white_kings != 1 or black_kings != 1:
            messagebox.showwarning(
                "Posición no válida",
                "Para analizar la posición debe haber exactamente un rey blanco y un rey negro.\n\n"
                "Corrige la posición antes de pulsar 'Analizar nueva posición'.",
            )
            return False

        return True

    def clear_graphs(self):
        if self.pie_canvas is not None:
            self.pie_canvas.get_tk_widget().destroy()
            self.pie_canvas = None
        if self.bar_canvas is not None:
            self.bar_canvas.get_tk_widget().destroy()
            self.bar_canvas = None

    def clear_tree(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    def get_responsive_graph_size(self):
        canvas_width = self.graphs_canvas.winfo_width()
        canvas_height = self.graphs_canvas.winfo_height()

        canvas_width = max(canvas_width, 500)
        canvas_height = max(canvas_height, 420)

        dpi = 100

        fig_width = max(5.0, min((canvas_width - 40) / dpi, 9.5))
        fig_height = max(3.8, min((canvas_height * 0.55) / dpi, 5.2))

        return fig_width, fig_height, dpi

    def get_source_block(self, report, source_name):
        source_key = "lichess" if source_name == "Lichess" else "masters"
        return report.get(source_key, {})

    def get_moves_for_source(self, report, source_name):
        data = self.get_source_block(report, source_name)
        moves = data.get("frequent_moves")
        if moves is None:
            moves = data.get("top_moves", [])
        stats = data.get("stats", {})
        total_games = stats.get("total_games", 0)
        return moves[:5], total_games

    def winner_text(self, winner):
        if winner is None:
            return "tablas"
        if winner == "white":
            return "blancas"
        if winner == "black":
            return "negras"
        return str(winner)

    def update_current_fen_label(self):
        if self.current_fen:
            self.current_fen_label.config(text=f"FEN actual: {self.current_fen}")
        else:
            self.current_fen_label.config(text="FEN actual: -")

    def show_image_preview(self, image_path):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self.interactive_board.pack_forget()
            self.preview_label.pack(fill="both", expand=True)
            self.preview_label.configure(
                text="Para ver la vista previa instala Pillow:\n\npip install pillow",
                image="",
            )
            self.preview_photo = None
            return

        # Oculta el tablero y muestra la imagen seleccionada.
        self.interactive_board.pack_forget()
        self.preview_label.pack(fill="both", expand=True)

        self.preview_label.configure(text="Cargando imagen...", image="")
        self.root.update_idletasks()

        try:
            img = Image.open(image_path)

            # Usar el tamaño real disponible del contenedor.
            max_w = max(300, self.board_container.winfo_width() - 20)
            max_h = max(300, self.board_container.winfo_height() - 20)

            img.thumbnail((max_w, max_h))

            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_photo, text="")

        except Exception as e:
            self.preview_label.configure(
                text=f"No se pudo cargar la imagen:\n{e}",
                image="",
            )
            self.preview_photo = None

    def show_interactive_board(self):
        self.preview_label.pack_forget()
        self.interactive_board.pack(fill="both", expand=True)

    def update_opening_title(self):
        opening_text = None

        if self.current_report:
            opening = self.current_report.get("opening", {}) or {}
            eco = opening.get("eco")
            name = opening.get("name")

            if eco and name:
                opening_text = f"{eco} — {name}"
            elif name:
                opening_text = name

        if opening_text:
            self.last_detected_opening_text = opening_text
            self.opening_title_var.set(opening_text)
        else:
            if self.last_detected_opening_text:
                self.opening_title_var.set(self.last_detected_opening_text)
            else:
                self.opening_title_var.set("Apertura no identificada")

    # ------------------------------------------------------------------
    # Compare window
    # ------------------------------------------------------------------

    def show_compare_window(self):
        if not self.current_image_path or not self.current_fen:
            messagebox.showwarning(
                "Aviso",
                "Primero debes analizar una imagen para poder compararla con el tablero.",
            )
            return

        if not os.path.exists(self.current_image_path):
            messagebox.showerror("Error", "No se encuentra la imagen original.")
            return

        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showerror(
                "Falta Pillow",
                "Para visualizar imágenes JPG/PNG en esta ventana instala Pillow:\n\npip install pillow",
            )
            return

        win = tk.Toplevel(self.root)
        win.title("Comparación: imagen original y tablero virtual")
        win.geometry("1350x760")
        win.minsize(900, 560)

        main = ttk.Frame(win, padding=10)
        main.pack(fill="both", expand=True)

        paned = ttk.Panedwindow(main, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=5)
        right = ttk.Frame(paned, padding=5)
        paned.add(left, weight=1)
        paned.add(right, weight=1)

        ttk.Label(left, text="Imagen original", font=("Segoe UI", 12, "bold")).pack(anchor="center", pady=(0, 8))
        ttk.Label(right, text="Tablero virtual actual", font=("Segoe UI", 12, "bold")).pack(anchor="center", pady=(0, 8))

        image_container = ttk.Frame(left)
        image_container.pack(fill="both", expand=True)

        image_label = ttk.Label(image_container, anchor="center")
        image_label.pack(fill="both", expand=True)

        def render_image(_event=None):
            try:
                img = Image.open(self.current_image_path)
                max_w = max(300, image_container.winfo_width() - 20)
                max_h = max(300, image_container.winfo_height() - 20)
                img.thumbnail((max_w, max_h))
                photo = ImageTk.PhotoImage(img)
                image_label.configure(image=photo, text="")
                image_label.image = photo
            except Exception as e:
                image_label.configure(text=f"No se pudo cargar la imagen:\n{e}")

        image_container.bind("<Configure>", render_image)
        win.after(100, render_image)

        compare_board = EditableBoard(
            right,
            on_position_changed=None,
            square_size=68,
        )
        compare_board.pack(fill="both", expand=True)
        compare_board.set_fen(self.current_fen)
        compare_board.set_edit_mode(False)

    # ------------------------------------------------------------------
    # Image analysis via main.py
    # ------------------------------------------------------------------

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self.image_path_var.set(path)
            self.current_image_path = path
            self.show_image_preview(path)
            self.status_var.set("Imagen cargada. Pulsa 'Analizar imagen' para comenzar.")

    def run_analysis(self):
        image_path = self.image_path_var.get().strip()

        if not image_path:
            messagebox.showwarning("Aviso", "Selecciona una imagen primero.")
            return

        if not os.path.exists(image_path):
            messagebox.showerror("Error", "La imagen seleccionada no existe.")
            return

        turn_ui = self.turn_var.get()
        rot_ui = self.rot_options[self.rot_display_var.get()]

        if turn_ui == "Blancas":
            cli_turn_mode = "w"
        elif turn_ui == "Negras":
            cli_turn_mode = "b"
        else:
            cli_turn_mode = "auto"

        cmd = [
            sys.executable,
            "main.py",
            "detect",
            "--input", image_path,
            "--turn-mode", cli_turn_mode,
            "--rot", rot_ui,
            "--results-dir", RESULTS_DIR,
        ]

        self.current_report = None
        self.current_fen = None
        self.detected_report = None
        self.detected_fen = None
        self.last_detected_opening_text = None
        self.update_opening_title()
        self.update_current_fen_label()
        self.clear_graphs()
        self.current_image_path = image_path
        self.show_image_preview(image_path)
        self.set_busy(True, "Analizando imagen...")

        thread = threading.Thread(
            target=self._run_analysis_worker,
            args=(cmd, image_path),
            daemon=True,
        )
        thread.start()

    def _run_analysis_worker(self, cmd, image_path):
        try:
            proc = subprocess.run(
                cmd,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda err=err: self._on_analysis_exception(err))
            return

        stem = os.path.splitext(os.path.basename(image_path))[0]
        result_dir = os.path.join(BASE_DIR, RESULTS_DIR, stem)
        self.root.after(0, lambda: self._on_analysis_finished(proc, result_dir))

    def _on_analysis_exception(self, error_text):
        self.set_busy(False, "Listo")
        messagebox.showerror("Error", f"No se pudo ejecutar main.py:\n{error_text}")

    def _on_analysis_finished(self, proc, result_dir):
        self.set_busy(False, "Análisis de imagen completado")

        if proc.returncode != 0:
            error_msg = (
                f"El análisis falló.\n\n"
                f"STDOUT:\n{proc.stdout}\n\n"
                f"STDERR:\n{proc.stderr}"
            )
            messagebox.showerror(
                "Error",
                "El análisis ha fallado. Revisa la consola o la salida del proceso.",
            )
            print(error_msg)
            self.current_report = None
            self.clear_graphs()
            return

        self.load_results(result_dir)

    # ------------------------------------------------------------------
    # Board correction / movement
    # ------------------------------------------------------------------

    def toggle_correction_mode(self):
        enabled = self.interactive_board.toggle_edit_mode()

        if enabled:
            self.correct_button.config(text="Salir de corrección")
            self.board_help_label.config(
                text="Modo corrección: clic izquierdo para mover piezas libremente. Clic derecho para borrar o cambiar una pieza."
            )
            self.status_var.set(
                "Modo corrección activado. Pulsa 'Analizar nueva posición' cuando termines de corregir."
            )
        else:
            self.correct_button.config(text="Corregir posiciones")
            self.board_help_label.config(
                text="Modo análisis: clic en una pieza y luego en la casilla destino."
            )
            self.status_var.set("Modo corrección desactivado.")

    def on_board_position_changed(self, new_fen: str, action: str):
        self.current_fen = new_fen
        self.update_current_fen_label()
        self.refresh_board_buttons()

        if action.startswith("edit"):
            self.status_var.set(
                "Corrección aplicada. Pulsa 'Analizar nueva posición' para actualizar resultados."
            )
        elif action == "legal_move":
            self.status_var.set(
                "Movimiento aplicado. Pulsa 'Analizar nueva posición' para actualizar resultados."
            )
        elif action == "undo":
            self.status_var.set(
                "Cambio deshecho. Pulsa 'Analizar nueva posición' para actualizar resultados."
            )
        elif action == "reset":
            self.status_var.set("Se ha vuelto a la posición inicial detectada.")

    def reanalyze_current_board(self):
        if not self.is_position_analyzable():
            return

        fen = self.get_clean_board_fen()

        if fen:
            self.current_fen = fen
            self.update_current_fen_label()
            self.start_fen_analysis(fen)

    def undo_board_change(self):
        ok = self.interactive_board.undo_last_change()
        if ok:
            self.current_fen = self.interactive_board.get_fen()
            self.update_current_fen_label()
            self.refresh_board_buttons()
            self.status_var.set(
                "Cambio deshecho. Pulsa 'Analizar nueva posición' para actualizar resultados."
            )

    def reset_interactive_board(self):
        if not self.detected_fen:
            return

        self.interactive_board.set_fen(self.detected_fen)
        self.show_interactive_board()
        self.current_fen = self.detected_fen
        self.current_report = self.detected_report

        self.update_current_fen_label()
        self.update_opening_title()
        self.render_tables()
        self.render_graphs()
        self.refresh_board_buttons()

        self.status_var.set("Se ha vuelto a la posición inicial detectada.")

    def start_fen_analysis(self, fen: str):
        self.set_busy(True, "Analizando nueva posición...")
        thread = threading.Thread(
            target=self._run_fen_analysis_worker,
            args=(fen,),
            daemon=True,
        )
        thread.start()

    def _run_fen_analysis_worker(self, fen: str):
        try:
            lichess_api = LichessAPI(token=None)

            report = lichess_api.build_report(
                fen=fen,
                variant="standard",
                moves=12,
                masters_top_games=10,
                lichess_top_games=0,
                recent_games=0,
                include_raw=True,
                auto_detect_turn=False,
            )
            report.pop("turn_selection", None)

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda err=err: self._on_fen_analysis_error(err))
            return

        self.root.after(0, lambda: self._on_fen_analysis_finished(report, fen))

    def _on_fen_analysis_error(self, error_text):
        self.set_busy(False, "Listo")
        self.refresh_board_buttons()

        if "timed out" in error_text or "ConnectTimeout" in error_text:
            messagebox.showwarning(
                "Lichess no responde",
                "La posición corregida se ha actualizado en el tablero, pero Lichess no ha respondido a tiempo.\n\n"
                "Puedes volver a pulsar 'Analizar nueva posición' dentro de unos segundos.\n\n"
                "No es un error de la corrección del tablero."
            )
        else:
            messagebox.showerror(
                "Error",
                f"No se pudo analizar la nueva posición:\n{error_text}"
            )
    def _on_fen_analysis_finished(self, report, fen: str):
        self.set_busy(False, "Nueva posición analizada")
        self.current_report = report
        self.current_fen = report.get("fen", fen)
        self.update_current_fen_label()
        self.update_opening_title()
        self.render_tables()
        self.render_graphs()
        self.refresh_board_buttons()

    # ------------------------------------------------------------------
    # Load initial analysis from saved files
    # ------------------------------------------------------------------

    def load_results(self, result_dir):
        fen_path = os.path.join(result_dir, "fen.txt")
        report_path = os.path.join(result_dir, "lichess_report.json")

        fen = read_text_file(fen_path)

        report = None
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
            except Exception:
                report = None

        if report and report.get("fen"):
            fen = report["fen"]

        self.current_report = report
        self.detected_report = report
        self.current_fen = fen
        self.detected_fen = fen
        self.update_current_fen_label()
        self.update_opening_title()

        if fen:
            self.interactive_board.set_fen(fen)
            self.show_interactive_board()
            self.refresh_board_buttons()

        self.render_tables()
        self.render_graphs()

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def render_tables(self):
        self.clear_tree(self.stats_tree)
        self.clear_tree(self.frequent_tree)
        self.clear_tree(self.best_tree)
        self.clear_tree(self.games_tree)

        report = self.current_report
        if not report:
            return

        for source_name, source_key in [("Lichess", "lichess"), ("Maestros", "masters")]:
            stats = report.get(source_key, {}).get("stats", {})
            self.stats_tree.insert(
                "",
                "end",
                values=(
                    source_name,
                    stats.get("total_games", 0),
                    f"{stats.get('white_wins', 0)} ({stats.get('white_win_pct', 0)}%)",
                    f"{stats.get('draws', 0)} ({stats.get('draw_pct', 0)}%)",
                    f"{stats.get('black_wins', 0)} ({stats.get('black_win_pct', 0)}%)",
                ),
            )

        source_name = self.table_source_var.get()
        source_key = "lichess" if source_name == "Lichess" else "masters"
        data = report.get(source_key, {})

        frequent_moves = data.get("frequent_moves")
        if frequent_moves is None:
            frequent_moves = data.get("top_moves", [])

        for move in frequent_moves[:5]:
            self.frequent_tree.insert(
                "",
                "end",
                values=(
                    move.get("san", ""),
                    move.get("game_count", 0),
                    move.get("white_win_pct", 0),
                    move.get("draw_pct", 0),
                    move.get("black_win_pct", 0),
                ),
            )

        for move in data.get("best_moves", [])[:5]:
            self.best_tree.insert(
                "",
                "end",
                values=(
                    move.get("san", ""),
                    move.get("game_count", 0),
                    f"{move.get('expected_score_pct', 0)}%",
                ),
            )

        for game in report.get("masters", {}).get("top_games", [])[:10]:
            white_name = game.get("white_name", "Desconocido")
            white_rating = game.get("white_rating", "?")
            black_name = game.get("black_name", "Desconocido")
            black_rating = game.get("black_rating", "?")
            month = game.get("month")
            date_str = str(month) if month is not None else "fecha desconocida"

            self.games_tree.insert(
                "",
                "end",
                values=(
                    f"{white_name} ({white_rating})",
                    f"{black_name} ({black_rating})",
                    date_str,
                    self.winner_text(game.get("winner")),
                ),
            )

        self.tables_canvas.update_idletasks()
        self.tables_canvas.configure(scrollregion=self.tables_canvas.bbox("all"))
        self.tables_canvas.yview_moveto(0)

    # ------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------

    def render_graphs(self):
        self.clear_graphs()

        if not self.current_report:
            return

        moves, total_games = self.get_moves_for_source(
            self.current_report,
            self.graph_source_var.get(),
        )

        if not moves:
            return

        fig_width, fig_height, dpi = self.get_responsive_graph_size()
        canvas_width = self.graphs_canvas.winfo_width()

        labels = [m.get("san", "?") for m in moves]
        sizes = [m.get("game_count", 0) for m in moves]

        used_games = sum(sizes)
        other_games = max(total_games - used_games, 0)

        pie_labels = list(labels)
        pie_sizes = list(sizes)

        if other_games > 0:
            pie_labels.append("Otras")
            pie_sizes.append(other_games)

        fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        colors = plt.cm.Set3(range(len(pie_sizes)))

        wedges, texts, autotexts = ax1.pie(
            pie_sizes,
            labels=None,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
            startangle=90,
            pctdistance=0.68,
            wedgeprops=dict(linewidth=1, edgecolor="white"),
        )

        for autotext in autotexts:
            autotext.set_fontsize(9)

        ax1.set_title(
            f"Jugadas más frecuentes ({self.graph_source_var.get()})",
            pad=18,
        )
        ax1.axis("equal")

        legend_labels = pie_labels

        ax1.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=False,
        )

        fig1.subplots_adjust(left=0.06, right=0.72, top=0.86, bottom=0.08)

        self.pie_canvas = FigureCanvasTkAgg(fig1, master=self.pie_frame)
        self.pie_canvas.draw()
        self.pie_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig1)

        bar_labels = labels
        white = [m.get("white_win_pct", 0) for m in moves]
        draws = [m.get("draw_pct", 0) for m in moves]
        black = [m.get("black_win_pct", 0) for m in moves]

        x = list(range(len(bar_labels)))

        fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        ax2.bar(x, white, label="Blancas")
        ax2.bar(x, draws, bottom=white, label="Tablas")
        bottom_black = [white[i] + draws[i] for i in range(len(white))]
        ax2.bar(x, black, bottom=bottom_black, label="Negras")

        ax2.set_xticks(x)
        if canvas_width < 750:
            ax2.set_xticklabels(bar_labels, rotation=25, ha="right", fontsize=8)
        else:
            ax2.set_xticklabels(bar_labels, fontsize=9)
        ax2.set_ylabel("Porcentaje")
        ax2.set_ylim(0, 100)
        ax2.set_title(f"Resultados por jugada ({self.graph_source_var.get()})", pad=14)
        ax2.grid(axis="y", linestyle="--", alpha=0.35)
        ax2.legend()

        for i in range(len(bar_labels)):
            if white[i] >= 7:
                ax2.text(i, white[i] / 2, f"{white[i]:.1f}%", ha="center", va="center", fontsize=8)
            if draws[i] >= 7:
                ax2.text(i, white[i] + draws[i] / 2, f"{draws[i]:.1f}%", ha="center", va="center", fontsize=8)
            if black[i] >= 7:
                ax2.text(
                    i,
                    white[i] + draws[i] + black[i] / 2,
                    f"{black[i]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        if canvas_width < 750:
            fig2.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.24)
        else:
            fig2.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.14)

        self.bar_canvas = FigureCanvasTkAgg(fig2, master=self.bar_frame)
        self.bar_canvas.draw()
        self.bar_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig2)

        self.graphs_canvas.update_idletasks()
        self.graphs_canvas.configure(scrollregion=self.graphs_canvas.bbox("all"))
        self.graphs_canvas.yview_moveto(0)


def main():
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
