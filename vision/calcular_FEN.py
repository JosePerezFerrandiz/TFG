FEN_MAP = {
    "white-pawn": "P",
    "white-knight": "N",
    "white-bishop": "B",
    "white-rook": "R",
    "white-queen": "Q",
    "white-king": "K",
    "black-pawn": "p",
    "black-knight": "n",
    "black-bishop": "b",
    "black-rook": "r",
    "black-queen": "q",
    "black-king": "k",
}

def assigned_to_fen_board(assigned, class_dict):
    rows = []

    for row in range(8):
        fen_row = ""
        empty_count = 0

        for col in range(8):
            cell_id = row * 8 + col + 1

            if cell_id in assigned:
                cls_id, _ = assigned[cell_id]
                piece_name = class_dict[cls_id]
                piece_char = FEN_MAP[piece_name]

                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0

                fen_row += piece_char
            else:
                empty_count += 1

        if empty_count > 0:
            fen_row += str(empty_count)

        rows.append(fen_row)

    return "/".join(rows)


def build_full_fen(assigned, class_dict, turn="w", castling="KQkq", ep="-", halfmove="0", fullmove="1"):
    board_fen = assigned_to_fen_board(assigned, class_dict)
    return f"{board_fen} {turn} {castling} {ep} {halfmove} {fullmove}"