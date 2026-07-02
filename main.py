import gc
import os
import glob
import shutil
import argparse
import json

import vision.utils as utils
print("<<< \x1b[5;32;40m neural-chessboard + tight-grid + padded-detection \x1b[0m >>>")

from config import *
from vision.utils import ImageObject
from vision.slid import pSLID, SLID, slid_tendency
from vision.laps import LAPS
from vision.llr import LLR, llr_pad

from keras import backend as K
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

from vision.calcular_FEN import build_full_fen
from lichessAPI import LichessAPI



load = cv2.imread
save = cv2.imwrite

CLASS_DICT = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn',
    4: 'black-queen', 5: 'black-rook', 6: 'white-bishop', 7: 'white-king',
    8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}

PIECE_MAPPING = {
    'white-pawn': chess.PAWN, 'black-pawn': chess.PAWN,
    'white-knight': chess.KNIGHT, 'black-knight': chess.KNIGHT,
    'white-bishop': chess.BISHOP, 'black-bishop': chess.BISHOP,
    'white-rook': chess.ROOK, 'black-rook': chess.ROOK,
    'white-queen': chess.QUEEN, 'black-queen': chess.QUEEN,
    'white-king': chess.KING, 'black-king': chess.KING,
}

MODEL_PATH = 'chess-model-yolov8m.pt'
CONF_THRES = 0.25
SQUARE_LENGTH = 150  # 150 * 8 = 1200 px el tablero ajustado

# Padding extra alrededor del tablero SOLO para detección.
# Son proporciones respecto al tamaño del tablero ajustado.
DEFAULT_PAD_LEFT = 0.08
DEFAULT_PAD_RIGHT = 0.08
DEFAULT_PAD_TOP = 0.12
DEFAULT_PAD_BOTTOM = 0.10

ASSIGN_TOL_X = 0.020
ASSIGN_TOL_Y = 0.012

NC_LAST_CROP_META = None

################################################################################
# neural-chessboard crop
################################################################################


def _is_valid_quad(points):
    if points is None:
        return False
    try:
        return len(points) == 4
    except Exception:
        return False



def layer():
    global NC_LAYER, NC_IMAGE, NC_LAST_CROP_META

    print(utils.ribb("==", sep="="))
    print(utils.ribb("[%d] LAYER " % NC_LAYER, sep="="))
    print(utils.ribb("==", sep="="), "\n")

    print(utils.ribb(utils.head("SLID"), utils.clock(), "--- 1 step "))
    segments = pSLID(NC_IMAGE['main'])
    raw_lines = SLID(NC_IMAGE['main'], segments)
    lines = slid_tendency(raw_lines)

    print(utils.ribb(utils.head("LAPS"), utils.clock(), "--- 2 step "))
    points = LAPS(NC_IMAGE['main'], lines)

    print(utils.ribb(utils.head(" LLR"), utils.clock(), "--- 3 step "))
    inner_points = None
    four_points = None

    try:
        inner_points = LLR(NC_IMAGE['main'], points, lines)
    except Exception as e:
        utils.warn(f"LLR falló: {e}")

    try:
        if inner_points is not None:
            four_points = llr_pad(inner_points, NC_IMAGE['main'])
    except Exception as e:
        utils.warn(f"llr_pad falló: {e}")

    chosen_points = None
    if _is_valid_quad(four_points):
        chosen_points = four_points
    elif _is_valid_quad(inner_points):
        chosen_points = inner_points
    elif _is_valid_quad(points):
        chosen_points = points
    else:
        raise RuntimeError("No se pudo obtener un cuadrilátero válido del tablero en esta capa.")

    print(utils.ribb(utils.head("   *"), utils.clock(), "--- 4 step "))
    print(chosen_points)

    # Guardamos la imagen fuente y los puntos ANTES del crop.
    NC_LAST_CROP_META = {
        'source_orig': NC_IMAGE['orig'].copy(),
        'scale': NC_IMAGE.scale,
        'points_main': np.array(chosen_points, dtype=np.float32),
    }

    NC_IMAGE.crop(chosen_points)
    print("\n")



def warp_from_quad_with_padding(source_img, points_orig, square_length, pad_left, pad_right, pad_top, pad_bottom):
    """
    Genera un warp en el que el tablero queda exactamente en un rectángulo interior
    y además deja contexto real alrededor usando la misma homografía.
    """
    board_len = int(square_length * 8)

    pad_left_px = int(round(board_len * pad_left))
    pad_right_px = int(round(board_len * pad_right))
    pad_top_px = int(round(board_len * pad_top))
    pad_bottom_px = int(round(board_len * pad_bottom))

    out_w = board_len + pad_left_px + pad_right_px
    out_h = board_len + pad_top_px + pad_bottom_px

    pts_src = np.array(points_orig, dtype=np.float32)

    # Reordenación igual que utils.image_transform: el punto más cercano a (0,0)
    # se considera la esquina superior izquierda del tablero.
    best_idx = 0
    best_val = 10**18
    for idx, val in enumerate(pts_src):
        d = float(np.linalg.norm(val - np.array([0.0, 0.0], dtype=np.float32)))
        if d < best_val:
            best_idx = idx
            best_val = d

    pts_src = np.roll(pts_src, shift=(4 - best_idx), axis=0)

    pts_dst = np.array([
        [pad_left_px, pad_top_px],
        [pad_left_px + board_len, pad_top_px],
        [pad_left_px + board_len, pad_top_px + board_len],
        [pad_left_px, pad_top_px + board_len],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(source_img, M, (out_w, out_h))

    board_rect = (
        int(pad_left_px),
        int(pad_top_px),
        int(pad_left_px + board_len),
        int(pad_top_px + board_len),
    )
    return warped, board_rect



def crop_board_with_neural_dual(image_path: str, square_length: int,
                                pad_left: float, pad_right: float,
                                pad_top: float, pad_bottom: float):
    global NC_LAYER, NC_IMAGE, NC_CONFIG, NC_LAST_CROP_META

    if not os.path.isfile(image_path):
        utils.errn('error: the file "%s" does not exits' % image_path)

    img = load(image_path)
    if img is None:
        utils.errn("OpenCV no pudo leer la imagen. Reguárdala como JPG/PNG e inténtalo otra vez.")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    NC_LAST_CROP_META = None
    NC_IMAGE, NC_LAYER = ImageObject(img), 0

    for _ in range(NC_CONFIG['layers']):
        NC_LAYER += 1
        layer()

    tight_crop = NC_IMAGE['orig'].copy()

    if NC_LAST_CROP_META is None:
        raise RuntimeError("No se pudo recuperar la última homografía del recorte.")

    points_orig = utils.image_scale(NC_LAST_CROP_META['points_main'], NC_LAST_CROP_META['scale'])
    source_orig = NC_LAST_CROP_META['source_orig']

    # Recalculamos ambos warps desde la misma capa final:
    # 1) ajustado exacto del tablero
    tight_warp, tight_board_rect = warp_from_quad_with_padding(
        source_orig,
        points_orig,
        square_length=square_length,
        pad_left=0.0,
        pad_right=0.0,
        pad_top=0.0,
        pad_bottom=0.0,
    )

    # 2) ampliado para detección, pero manteniendo el tablero en un rectángulo interior fijo
    detection_crop, detection_board_rect = warp_from_quad_with_padding(
        source_orig,
        points_orig,
        square_length=square_length,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )

    return {
        'tight_crop': tight_warp if tight_warp is not None else tight_crop,
        'tight_board_rect': tight_board_rect,
        'detection_crop': detection_crop,
        'detection_board_rect': detection_board_rect,
    }


################################################################################
# piece pipeline
################################################################################


def detect_pieces(model: YOLO, image: np.ndarray, conf: float = CONF_THRES):
    results = model.predict(image, conf=conf, verbose=False)
    detections = []

    for result in results:
        if result.boxes is None:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        cls = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()

        for box, c, score in zip(xyxy, cls, scores):
            detections.append((box, c, float(score)))

    return detections



def draw_grid_on_rect(image: np.ndarray, board_rect, with_numbers=True):
    out = image.copy()
    x_min, y_min, x_max, y_max = board_rect

    board_w = x_max - x_min
    board_h = y_max - y_min
    cell_w = board_w / 8.0
    cell_h = board_h / 8.0

    n = 1
    for r in range(8):
        for c in range(8):
            x1 = int(round(x_min + c * cell_w))
            y1 = int(round(y_min + r * cell_h))
            x2 = int(round(x_min + (c + 1) * cell_w))
            y2 = int(round(y_min + (r + 1) * cell_h))

            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            if with_numbers:
                cv2.putText(
                    out, str(n), (x1 + 6, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
                )
            n += 1

    return out



def assign_detections_to_cells(detections, board_rect, image_for_draw):
    assigned = {}
    square_debug = []

    x_min, y_min, x_max, y_max = board_rect
    board_w = x_max - x_min
    board_h = y_max - y_min
    cell_w = board_w / 8.0
    cell_h = board_h / 8.0

    tol_x = ASSIGN_TOL_X * board_w
    tol_y = ASSIGN_TOL_Y * board_h

    for box, cls_id, score in detections:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(image_for_draw, (x1, y1), (x2, y2), (0, 255, 255), 2)

        box_h = max(1, y2 - y1)
        box_w = max(1, x2 - x1)

        # Nos quedamos con la zona baja de la pieza para decidir casilla.
        foot_y1 = y2 - 0.22 * box_h
        foot_y2 = y2
        foot_x1 = x1 + 0.22 * box_w
        foot_x2 = x2 - 0.22 * box_w

        cv2.rectangle(
            image_for_draw,
            (int(foot_x1), int(foot_y1)),
            (int(foot_x2), int(foot_y2)),
            (255, 0, 255),
            1
        )

        foot_cx = (foot_x1 + foot_x2) / 2.0
        foot_cy = (foot_y1 + foot_y2) / 2.0

        if not (x_min - tol_x <= foot_cx < x_max + tol_x and
                y_min - tol_y <= foot_cy < y_max + tol_y):
            continue

        col = int((foot_cx - x_min) / cell_w)
        row = int((foot_cy - y_min) / cell_h)

        col = min(max(col, 0), 7)
        row = min(max(row, 0), 7)
        best_cell = row * 8 + col + 1

        cv2.putText(
            image_for_draw,
            f'{CLASS_DICT.get(cls_id, cls_id)} {score:.2f}',
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        prev = assigned.get(best_cell)
        if prev is None or score > prev[1]:
            assigned[best_cell] = (cls_id, score)

    for cell_id, (cls_id, score) in assigned.items():
        square_debug.append([cell_id, cls_id, score])

    return assigned, square_debug



def choose_rotation_from_piece_colors(assigned):
    white_rc = []
    black_rc = []

    for cell_id, (cls_id, _) in assigned.items():
        piece_name = CLASS_DICT.get(int(cls_id), '')
        row = (cell_id - 1) // 8
        col = (cell_id - 1) % 8

        if piece_name.startswith('white'):
            white_rc.append((row, col))
        elif piece_name.startswith('black'):
            black_rc.append((row, col))

    if len(white_rc) < 2 or len(black_rc) < 2:
        return 'none'

    wr = float(np.mean([r for r, _ in white_rc]))
    wc = float(np.mean([c for _, c in white_rc]))
    br = float(np.mean([r for r, _ in black_rc]))
    bc = float(np.mean([c for _, c in black_rc]))

    dx = wc - bc
    dy = wr - br

    if abs(dx) > abs(dy):
        return 'ccw' if wc < bc else 'cw'
    if wr < br:
        return '180'
    return 'none'



def rotate_cell_id(cell_id: int, rotation: str) -> int:
    row = (cell_id - 1) // 8
    col = (cell_id - 1) % 8

    if rotation == 'none':
        new_row, new_col = row, col
    elif rotation == 'cw':
        new_row, new_col = col, 7 - row
    elif rotation == 'ccw':
        new_row, new_col = 7 - col, row
    elif rotation == '180':
        new_row, new_col = 7 - row, 7 - col
    else:
        raise ValueError(f'Rotación no soportada: {rotation}')

    return new_row * 8 + new_col + 1



def remap_assigned_cells(assigned, rotation: str):
    if rotation == 'none':
        return dict(assigned)

    remapped = {}
    for cell_id, value in assigned.items():
        new_cell = rotate_cell_id(cell_id, rotation)
        prev = remapped.get(new_cell)
        if prev is None or value[1] > prev[1]:
            remapped[new_cell] = value

    return remapped



def remap_square_debug(square_debug, rotation: str):
    if rotation == 'none':
        return square_debug

    out = []
    for cell_id, cls_id, score in square_debug:
        out.append([rotate_cell_id(int(cell_id), rotation), cls_id, score])
    return out



def assigned_to_board(assigned):
    board = chess.Board(None)

    for cell_id in range(1, 65):
        if cell_id not in assigned:
            continue

        cls_id, _ = assigned[cell_id]
        piece_name = CLASS_DICT[cls_id]
        piece_type = PIECE_MAPPING[piece_name]
        color = chess.WHITE if piece_name.startswith('white') else chess.BLACK

        row = (cell_id - 1) // 8
        col = (cell_id - 1) % 8
        square = chess.square(col, 7 - row)
        board.set_piece_at(square, chess.Piece(piece_type, color))

    return board



def infer_castling_rights(assigned, class_dict):
    piece_by_cell = {}

    for cell_id, (cls_id, _) in assigned.items():
        piece_by_cell[cell_id] = class_dict[cls_id]

    rights = ""

    if piece_by_cell.get(61) == "white-king":
        if piece_by_cell.get(64) == "white-rook":
            rights += "K"
        if piece_by_cell.get(57) == "white-rook":
            rights += "Q"

    if piece_by_cell.get(5) == "black-king":
        if piece_by_cell.get(8) == "black-rook":
            rights += "k"
        if piece_by_cell.get(1) == "black-rook":
            rights += "q"

    return rights if rights else "-"



def convert_svg_to_png(svg_file_path, png_file_path):
    drawing = svg2rlg(svg_file_path)
    renderPM.drawToFile(drawing, png_file_path, fmt='PNG')



def save_piece_outputs(tight_crop: np.ndarray,
                       tight_grid_image: np.ndarray,
                       detection_crop: np.ndarray,
                       detection_grid_image: np.ndarray,
                       detected_image: np.ndarray,
                       board: chess.Board,
                       square_debug,
                       rotation_used: str,
                       model_path: str,
                       output_dir: str,
                       pads: tuple,
                       fen_code: str):
    os.makedirs(output_dir, exist_ok=True)

    svg_path = os.path.join(output_dir, '2Dboard.svg')
    png_path = os.path.join(output_dir, '2Dboard.png')

    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(chess.svg.board(board))
    

    convert_svg_to_png(svg_path, png_path)

    pd.DataFrame(square_debug, columns=['cell', 'class_id', 'score']).to_csv(
        os.path.join(output_dir, 'piece_assignments.csv'), index=False
    )

    cv2.imwrite(os.path.join(output_dir, 'tight_crop.png'), tight_crop)
    cv2.imwrite(os.path.join(output_dir, 'tight_grid.png'), tight_grid_image)
    cv2.imwrite(os.path.join(output_dir, 'detection_crop.png'), detection_crop)
    cv2.imwrite(os.path.join(output_dir, 'detection_grid.png'), detection_grid_image)
    cv2.imwrite(os.path.join(output_dir, 'debug_detections.png'), detected_image)

    board_png = cv2.imread(png_path)
    if board_png is not None:
        h = 420

        def resize_keep(img):
            scale = h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1] * scale), h))

        panel = np.hstack([
            resize_keep(tight_grid_image),
            resize_keep(detection_grid_image),
            resize_keep(detected_image),
            resize_keep(board_png),
        ])
        cv2.imwrite(os.path.join(output_dir, 'result.jpeg'), panel)

    with open(os.path.join(output_dir, 'fen.txt'), 'w', encoding='utf-8') as f:
        f.write(fen_code)
        

    pl, pr, pt, pb = pads
    with open(os.path.join(output_dir, 'run_info.txt'), 'w', encoding='utf-8') as f:
        f.write('pipeline=neural-chessboard-tight-grid + padded-piece-detection\n')
        f.write(f'rotation_used={rotation_used}\n')
        f.write(f'model_path={model_path}\n')
        f.write(f'pad_left={pl}\n')
        f.write(f'pad_right={pr}\n')
        f.write(f'pad_top={pt}\n')
        f.write(f'pad_bottom={pb}\n')





def run_piece_pipeline(crops: dict, args, result_dir: str):
    model = YOLO(args.model)

    tight_crop = crops['tight_crop']
    tight_board_rect = crops['tight_board_rect']
    detection_crop = crops['detection_crop']
    detection_board_rect = crops['detection_board_rect']

    tight_grid_image = draw_grid_on_rect(tight_crop, tight_board_rect)
    detection_grid_image = draw_grid_on_rect(detection_crop, detection_board_rect)

    detections = detect_pieces(model, detection_crop, conf=args.conf)

    detected_image = detection_grid_image.copy()
    assigned_raw, square_debug_raw = assign_detections_to_cells(
        detections,
        detection_board_rect,
        detected_image,
    )

    if args.rot == 'auto':
        rotation_used = choose_rotation_from_piece_colors(assigned_raw)
    else:
        rotation_used = args.rot

    assigned = remap_assigned_cells(assigned_raw, rotation_used)
    square_debug = remap_square_debug(square_debug_raw, rotation_used)

    board = assigned_to_board(assigned)
    castling = infer_castling_rights(assigned, CLASS_DICT)
    turn_mode = getattr(args, "turn_mode", "auto")
    auto_detect_turn = (turn_mode == "auto")
    fen_turn = "w" if auto_detect_turn else turn_mode

    fen_code = build_full_fen(
        assigned,
        CLASS_DICT,
        turn=fen_turn,
        castling=castling
    )

    save_piece_outputs(
        tight_crop=tight_crop,
        tight_grid_image=tight_grid_image,
        detection_crop=detection_crop,
        detection_grid_image=detection_grid_image,
        detected_image=detected_image,
        board=board,
        square_debug=square_debug,
        rotation_used=rotation_used,
        model_path=args.model,
        output_dir=result_dir,
        pads=(args.pad_left, args.pad_right, args.pad_top, args.pad_bottom),
        fen_code=fen_code,
    )

    
    report = None

    try:
        LICHESS_TOKEN = "lip_SqnA7wQRFb6fyUmRIqnb"
        lichess_api = LichessAPI(token=LICHESS_TOKEN)

        report = lichess_api.build_report(
            fen=fen_code,
            variant="standard",
            moves=12,
            masters_top_games=10,
            lichess_top_games=0,
            recent_games=0,
            include_raw=True,
            auto_detect_turn=auto_detect_turn
        )
        if not auto_detect_turn:
            report.pop("turn_selection", None)

        lichess_api.print_report(report)
        lichess_api.save_report_json(
            report,
            os.path.join(result_dir, "lichess_report.json")
        )

        # Sobrescribir fen.txt con la FEN final elegida
        with open(os.path.join(result_dir, "fen.txt"), "w", encoding="utf-8") as f:
            f.write(report["fen"])

        with open(os.path.join(result_dir, "lichess_masters.json"), "w", encoding="utf-8") as f:
            json.dump(report["raw"]["masters"], f, ensure_ascii=False, indent=2)

    except Exception as e:
        print("No se pudo consultar Lichess:", e)

    print('Piezas asignadas finales:', len(assigned))
    print('Rotación aplicada al mapeo de casillas:', rotation_used)
    print('FEN:', fen_code)


################################################################################
# modes
################################################################################


def reset_results_dir(results_dir: str):
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    print(f'Results root preparado: {results_dir}')



def detect(args):
    if not args.input:
        utils.errn('detect requiere --input')

    stem = os.path.splitext(os.path.basename(args.input))[0]
    result_dir = os.path.join(args.results_dir, stem)
    os.makedirs(result_dir, exist_ok=True)

    crops = crop_board_with_neural_dual(
        image_path=args.input,
        square_length=args.square_length,
        pad_left=args.pad_left,
        pad_right=args.pad_right,
        pad_top=args.pad_top,
        pad_bottom=args.pad_bottom,
    )

    tight_output = args.output
    if not tight_output:
        tight_output = os.path.join(result_dir, 'tight_crop.jpg')

    out_dir = os.path.dirname(tight_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = save(tight_output, crops['tight_crop'])
    if not ok:
        utils.errn('Could not save output to "%s"' % tight_output)

    run_piece_pipeline(crops, args, result_dir)

    print("DETECT:", args.input)
    print("TIGHT CROP SAVED:", tight_output)
    print("RESULTS DIR:", result_dir)



def dataset(args):
    print("DATASET: use dataset.py")



def train(args):
    print("TRAIN: use train.py")



def test(args):
    patterns = ['test/in/*.jpg', 'test/in/*.jpeg', 'test/in/*.png']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    for iname in files:
        args.input = iname
        args.output = None
        detect(args)

    print("TEST: %d images" % len(files))


################################################################################

if __name__ == "__main__":
    utils.reset()

    p = argparse.ArgumentParser(
        description='Crop ajustado con neural-chessboard, grid exacto y detección con padding real.'
    )

    p.add_argument('mode', nargs=1, type=str, help='detect | dataset | train | test')
    p.add_argument('--input', type=str, help='input image')
    p.add_argument('--output', type=str, default=None,
                   help='ruta opcional del recorte ajustado')
    p.add_argument('--model', type=str, default=MODEL_PATH, help='ruta al modelo YOLO')
    p.add_argument('--conf', type=float, default=CONF_THRES, help='confidence threshold de YOLO')
    p.add_argument('--rot', choices=['auto', 'none', 'cw', 'ccw', '180'], default='auto',
                   help='rotación a aplicar al mapeo de casillas')
    p.add_argument('--turn', choices=['w', 'b'], default='w',
                   help='turno para construir la FEN completa')
    p.add_argument('--square-length', type=int, default=SQUARE_LENGTH,
                   help='tamaño en píxeles de cada casilla del warp')
    p.add_argument('--pad-left', type=float, default=DEFAULT_PAD_LEFT,
                   help='padding extra izquierdo respecto al tamaño del tablero')
    p.add_argument('--pad-right', type=float, default=DEFAULT_PAD_RIGHT,
                   help='padding extra derecho respecto al tamaño del tablero')
    p.add_argument('--pad-top', type=float, default=DEFAULT_PAD_TOP,
                   help='padding extra superior respecto al tamaño del tablero')
    p.add_argument('--pad-bottom', type=float, default=DEFAULT_PAD_BOTTOM,
                   help='padding extra inferior respecto al tamaño del tablero')
    p.add_argument('--results-dir', type=str, default='extracted-data',
                   help='directorio raíz para paneles, CSV, FEN e imágenes')
    p.add_argument('--lichess-token', type=str, default=None,
                   help='token opcional de Lichess; si no se indica, se intenta usar la variable de entorno LICHESS_TOKEN')
    p.add_argument(
    '--turn-mode',
    choices=['auto', 'w', 'b'],
    default='auto',
    help='modo de turno: auto, w o b'
    )

    steps_dir = os.path.join("test", "steps")
    if os.path.exists(steps_dir):
        shutil.rmtree(steps_dir)
    os.makedirs(steps_dir, exist_ok=True)

    args = p.parse_args()
    mode = str(args.mode[0])
    modes = {'detect': detect, 'dataset': dataset, 'train': train, 'test': test}

    if mode not in modes.keys():
        utils.errn("hey, nie mamy takiej procedury!!! (wybrano: %s)" % mode)

    if mode in {'detect', 'test'}:
        reset_results_dir(args.results_dir)

    modes[mode](args)
    print(utils.clock(), "done")
    K.clear_session()
    gc.collect()
