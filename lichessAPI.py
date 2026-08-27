import json
import os
import time
from typing import Any, Dict, List, Optional

import requests


class LichessAPI:
    MIN_MOVE_POOL = 20
    FREQUENT_MOVES_TO_SHOW = 5
    BEST_MOVES_TO_SHOW = 5
    MIN_GAMES_FOR_BEST_MOVE = 20

    def __init__(self, token: Optional[str] = None, timeout: int = 60):
        self.base_url = "https://explorer.lichess.ovh"
        self.timeout = timeout

        self.headers = {
            "Accept": "application/json",
            "User-Agent": "TFG-Jose-Ajedrez/1.0"
        }

        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(
                url,
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error de conexión con Lichess: {e}")

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                raise Exception("Lichess devolvió una respuesta que no es JSON valido.")

        if response.status_code == 429:
            raise Exception(
                "Lichess ha devuelto 429 (demasiadas peticiones). "
                "Espera un poco antes de volver a consultar."
            )

        if response.status_code in (401, 403):
            raise Exception(
                f"Error {response.status_code}: token no válido o acceso denegado."
            )

        raise Exception(f"Error {response.status_code}: {response.text}")

    def get_masters(
        self,
        fen: str,
        variant: str = "standard",
        moves: int = 12,
        top_games: int = 10
    ) -> Dict[str, Any]:
        params = {
            "fen": fen,
            "variant": variant,
            "moves": moves,
            "topGames": top_games
        }
        return self._get("masters", params)

    def get_lichess(
        self,
        fen: str,
        variant: str = "standard",
        moves: int = 12,
        top_games: int = 10,
        recent_games: int = 0
    ) -> Dict[str, Any]:
        params = {
            "fen": fen,
            "variant": variant,
            "moves": moves,
            "topGames": top_games,
            "recentGames": recent_games
        }
        return self._get("lichess", params)

    @staticmethod
    def _safe_opening(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        opening = data.get("opening") or {}
        return {
            "eco": opening.get("eco"),
            "name": opening.get("name")
        }

    @staticmethod
    def _stats_block(data: Dict[str, Any]) -> Dict[str, Any]:
        white = int(data.get("white", 0))
        draws = int(data.get("draws", 0))
        black = int(data.get("black", 0))
        total = white + draws + black

        if total == 0:
            return {
                "total_games": 0,
                "white_wins": 0,
                "draws": 0,
                "black_wins": 0,
                "white_win_pct": 0.0,
                "draw_pct": 0.0,
                "black_win_pct": 0.0
            }

        return {
            "total_games": total,
            "white_wins": white,
            "draws": draws,
            "black_wins": black,
            "white_win_pct": round(100 * white / total, 2),
            "draw_pct": round(100 * draws / total, 2),
            "black_win_pct": round(100 * black / total, 2)
        }

    @staticmethod
    def _parse_moves(data: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        moves = data.get("moves", []) or []
        if limit is not None:
            moves = moves[:limit]

        parsed = []

        for move in moves:
            white = int(move.get("white", 0))
            draws = int(move.get("draws", 0))
            black = int(move.get("black", 0))
            total = white + draws + black

            if total > 0:
                white_pct = round(100 * white / total, 2)
                draws_pct = round(100 * draws / total, 2)
                black_pct = round(100 * black / total, 2)
            else:
                white_pct = draws_pct = black_pct = 0.0

            move_opening = move.get("opening") or {}

            parsed.append({
                "uci": move.get("uci"),
                "san": move.get("san"),
                "average_rating": move.get("averageRating"),
                "game_count": total,
                "white_wins": white,
                "draws": draws,
                "black_wins": black,
                "white_win_pct": white_pct,
                "draw_pct": draws_pct,
                "black_win_pct": black_pct,
                "opening_name": move_opening.get("name"),
                "opening_eco": move_opening.get("eco")
            })

        return parsed

    @staticmethod
    def _parse_top_games(data: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        games = data.get("topGames", []) or []
        parsed = []

        for game in games[:limit]:
            white_player = game.get("white") or {}
            black_player = game.get("black") or {}

            parsed.append({
                "id": game.get("id"),
                "winner": game.get("winner"),
                "year": game.get("year"),
                "month": game.get("month"),
                "white_name": white_player.get("name"),
                "white_rating": white_player.get("rating"),
                "black_name": black_player.get("name"),
                "black_rating": black_player.get("rating")
            })

        return parsed

    @staticmethod
    def _normalize_fen_parts(fen: str) -> Dict[str, Optional[str]]:
        parts = fen.strip().split()
        if not parts:
            raise ValueError("El FEN está vacío.")

        return {
            "board": parts[0],
            "turn": parts[1] if len(parts) > 1 and parts[1] in ("w", "b") else None,
            "castling": parts[2] if len(parts) > 2 else "-",
            "en_passant": parts[3] if len(parts) > 3 else "-",
            "halfmove": parts[4] if len(parts) > 4 else "0",
            "fullmove": parts[5] if len(parts) > 5 else "1",
        }

    @classmethod
    def _compose_fen(
        cls,
        board: str,
        turn: str,
        castling: str = "-",
        en_passant: str = "-",
        halfmove: str = "0",
        fullmove: str = "1",
    ) -> str:
        return f"{board} {turn} {castling} {en_passant} {halfmove} {fullmove}"

    @staticmethod
    def _extract_side_to_move(fen: str) -> str:
        parts = fen.strip().split()
        if len(parts) >= 2 and parts[1] in ("w", "b"):
            return parts[1]
        return "w"

    def _build_turn_candidates(self, fen: str, auto_detect_turn: bool = False) -> List[Dict[str, Any]]:
        info = self._normalize_fen_parts(fen)
        board = info["board"]
        castling = info["castling"] or "-"
        en_passant = info["en_passant"] or "-"
        halfmove = info["halfmove"] or "0"
        fullmove = info["fullmove"] or "1"
        original_turn = info["turn"]

        candidates: List[Dict[str, Any]] = []
        turns_to_try: List[str]

        if auto_detect_turn:
            turns_to_try = ["w", "b"]
        else:
            turns_to_try = [original_turn or "w"]

        seen = set()
        for turn in turns_to_try:
            candidate_fen = self._compose_fen(
                board=board,
                turn=turn,
                castling=castling,
                en_passant=en_passant,
                halfmove=halfmove,
                fullmove=fullmove,
            )
            if candidate_fen in seen:
                continue
            seen.add(candidate_fen)
            candidates.append({
                "fen": candidate_fen,
                "turn": turn,
                "is_original_turn": (original_turn == turn),
            })

        return candidates

    @staticmethod
    def _expected_score(move: Dict[str, Any], side_to_move: str) -> float:
        total = int(move.get("game_count", 0))
        if total <= 0:
            return 0.0

        white = int(move.get("white_wins", 0))
        draws = int(move.get("draws", 0))
        black = int(move.get("black_wins", 0))

        if side_to_move == "b":
            return (black + 0.5 * draws) / total
        return (white + 0.5 * draws) / total

    def _sort_best_moves(
        self,
        moves: List[Dict[str, Any]],
        side_to_move: str,
        min_games: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if min_games is None:
            min_games = self.MIN_GAMES_FOR_BEST_MOVE

        filtered = [m for m in moves if int(m.get("game_count", 0)) >= min_games]
        candidates = filtered if filtered else moves

        enriched: List[Dict[str, Any]] = []
        for move in candidates:
            move_copy = dict(move)
            move_copy["expected_score_pct"] = round(
                100 * self._expected_score(move_copy, side_to_move), 2
            )
            enriched.append(move_copy)

        return sorted(
            enriched,
            key=lambda m: (
                m.get("expected_score_pct", 0.0),
                m.get("game_count", 0),
                m.get("average_rating") or 0
            ),
            reverse=True
        )

    @staticmethod
    def _candidate_score(report: Dict[str, Any]) -> int:
        opening = report.get("opening") or {}
        lichess_stats = report.get("lichess", {}).get("stats", {})
        masters_stats = report.get("masters", {}).get("stats", {})

        lichess_total = int(lichess_stats.get("total_games", 0))
        masters_total = int(masters_stats.get("total_games", 0))
        lichess_move_count = len(report.get("lichess", {}).get("top_moves", []) or [])
        masters_move_count = len(report.get("masters", {}).get("top_moves", []) or [])
        opening_bonus = 500 if opening.get("name") else 0

        return (
            masters_total * 3
            + lichess_total
            + masters_move_count * 50
            + lichess_move_count * 20
            + opening_bonus
        )

    def _build_single_report(
        self,
        fen: str,
        variant: str = "standard",
        moves: int = 12,
        masters_top_games: int = 10,
        lichess_top_games: int = 0,
        recent_games: int = 0,
        include_raw: bool = True
    ) -> Dict[str, Any]:
        move_pool_size = max(int(moves), self.MIN_MOVE_POOL)
        side_to_move = self._extract_side_to_move(fen)

        lichess_data = self.get_lichess(
            fen=fen,
            variant=variant,
            moves=move_pool_size,
            top_games=lichess_top_games,
            recent_games=recent_games
        )

        masters_data = self.get_masters(
            fen=fen,
            variant=variant,
            moves=move_pool_size,
            top_games=masters_top_games
        )

        opening = self._safe_opening(masters_data)
        if not opening["name"]:
            opening = self._safe_opening(lichess_data)

        lichess_all_moves = self._parse_moves(lichess_data, limit=move_pool_size)
        masters_all_moves = self._parse_moves(masters_data, limit=move_pool_size)

        lichess_frequent_moves = lichess_all_moves[:self.FREQUENT_MOVES_TO_SHOW]
        masters_frequent_moves = masters_all_moves[:self.FREQUENT_MOVES_TO_SHOW]

        lichess_best_moves = self._sort_best_moves(
            lichess_all_moves,
            side_to_move=side_to_move
        )[:self.BEST_MOVES_TO_SHOW]
        masters_best_moves = self._sort_best_moves(
            masters_all_moves,
            side_to_move=side_to_move
        )[:self.BEST_MOVES_TO_SHOW]

        report = {
            "fen": fen,
            "side_to_move": side_to_move,
            "opening": opening,
            "ranking_criteria": {
                "frequent_moves": "Ordenadas por número de partidas devuelto por Lichess.",
                "best_moves": (
                    "Reordenadas por score esperado para el bando al que le toca mover "
                    "(victoria = 1, tablas = 0.5, derrota = 0), usando un mínimo de "
                    f"{self.MIN_GAMES_FOR_BEST_MOVE} partidas cuando sea posible."
                ),
                "move_pool_size": move_pool_size
            },
            "lichess": {
                "stats": self._stats_block(lichess_data),
                "top_moves": lichess_frequent_moves,
                "frequent_moves": lichess_frequent_moves,
                "best_moves": lichess_best_moves
            },
            "masters": {
                "stats": self._stats_block(masters_data),
                "top_moves": masters_frequent_moves,
                "frequent_moves": masters_frequent_moves,
                "best_moves": masters_best_moves,
                "top_games": self._parse_top_games(masters_data, limit=masters_top_games)
            }
        }

        if include_raw:
            report["raw"] = {
                "lichess": lichess_data,
                "masters": masters_data
            }

        return report

    def build_report(
        self,
        fen: str,
        variant: str = "standard",
        moves: int = 12,
        masters_top_games: int = 10,
        lichess_top_games: int = 0,
        recent_games: int = 0,
        include_raw: bool = True,
        auto_detect_turn: bool = False,
        pause_between_candidates: float = 0.15,
    ) -> Dict[str, Any]:
        candidates = self._build_turn_candidates(fen, auto_detect_turn=auto_detect_turn)
        evaluated: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for index, candidate in enumerate(candidates):
            try:
                candidate_report = self._build_single_report(
                    fen=candidate["fen"],
                    variant=variant,
                    moves=moves,
                    masters_top_games=masters_top_games,
                    lichess_top_games=lichess_top_games,
                    recent_games=recent_games,
                    include_raw=include_raw,
                )
                candidate_report["selected_turn"] = candidate["turn"]
                candidate_report["score"] = self._candidate_score(candidate_report)
                candidate_report["turn_was_in_input"] = candidate["is_original_turn"]
                evaluated.append(candidate_report)
            except Exception as e:
                errors.append({
                    "fen": candidate["fen"],
                    "turn": candidate["turn"],
                    "error": str(e)
                })

            if auto_detect_turn and index < len(candidates) - 1 and pause_between_candidates > 0:
                time.sleep(pause_between_candidates)

        if not evaluated:
            details = " | ".join(f"{err['turn']}: {err['error']}" for err in errors) or "Sin detalles"
            raise Exception(f"No se pudo consultar Lichess con ninguna hipótesis de turno. {details}")

        best_report = max(
            evaluated,
            key=lambda r: (
                r.get("score", 0),
                r.get("turn_was_in_input", False),
            )
        )

        if auto_detect_turn:
            best_report["turn_selection"] = {
                "mode": "auto",
                "selected_turn": best_report.get("selected_turn"),
                "selected_fen": best_report.get("fen"),
                "alternatives": [
                    {
                        "fen": r.get("fen"),
                        "turn": r.get("selected_turn"),
                        "score": r.get("score"),
                        "opening": r.get("opening"),
                        "masters_total_games": r.get("masters", {}).get("stats", {}).get("total_games", 0),
                        "lichess_total_games": r.get("lichess", {}).get("stats", {}).get("total_games", 0),
                    }
                    for r in sorted(evaluated, key=lambda x: x.get("score", 0), reverse=True)
                ],
                "errors": errors,
            }

        return best_report

    def save_report_json(self, report: Dict[str, Any], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _print_move_block(moves: List[Dict[str, Any]], show_expected_score: bool = False) -> None:
        for i, move in enumerate(moves, start=1):
            base_text = (
                f"{i}. {move['san']} | partidas={move['game_count']} | "
                f"blancas={move['white_win_pct']}% tablas={move['draw_pct']}% negras={move['black_win_pct']}%"
            )
            if show_expected_score:
                base_text += f" | score esperado={move.get('expected_score_pct', 0.0)}%"
            print(base_text)

    def print_report(self, report: Dict[str, Any]) -> None:
        opening = report.get("opening", {})
        opening_name = opening.get("name") or "Apertura no identificada"
        opening_eco = opening.get("eco") or "Sin ECO"

        print("\n===== INFORME LICHESS =====")
        print("FEN:", report["fen"])
        print("Apertura:", f"{opening_eco} - {opening_name}")

        turn_selection = report.get("turn_selection")
        if turn_selection:
            print("Turno seleccionado automáticamente:", turn_selection.get("selected_turn"))
            print("Hipótesis evaluadas:")
            for alt in turn_selection.get("alternatives", []):
                print(
                    f"- turno={alt['turn']} | score={alt['score']} | "
                    f"masters={alt['masters_total_games']} | lichess={alt['lichess_total_games']} | "
                    f"apertura={alt['opening'].get('name') if alt['opening'] else None}"
                )

        ranking_criteria = report.get("ranking_criteria", {})
        move_pool_size = ranking_criteria.get("move_pool_size")
        if move_pool_size:
            print(f"Movimientos analizados para reordenar mejores jugadas: {move_pool_size}")

        lichess_stats = report["lichess"]["stats"]
        print("\n--- Base general de Lichess ---")
        print("Total partidas:", lichess_stats["total_games"])
        print(f"Victorias blancas: {lichess_stats['white_wins']} ({lichess_stats['white_win_pct']}%)")
        print(f"Tablas: {lichess_stats['draws']} ({lichess_stats['draw_pct']}%)")
        print(f"Victorias negras: {lichess_stats['black_wins']} ({lichess_stats['black_win_pct']}%)")

        print("\nJugadas más frecuentes (Lichess):")
        self._print_move_block(report["lichess"].get("frequent_moves", report["lichess"].get("top_moves", [])))

        print("\nMejores jugadas según rendimiento histórico (Lichess):")
        self._print_move_block(report["lichess"].get("best_moves", []), show_expected_score=True)

        masters_stats = report["masters"]["stats"]
        print("\n--- Base de Maestros ---")
        print("Total partidas:", masters_stats["total_games"])
        print(f"Victorias blancas: {masters_stats['white_wins']} ({masters_stats['white_win_pct']}%)")
        print(f"Tablas: {masters_stats['draws']} ({masters_stats['draw_pct']}%)")
        print(f"Victorias negras: {masters_stats['black_wins']} ({masters_stats['black_win_pct']}%)")

        print("\nJugadas más frecuentes (Maestros):")
        self._print_move_block(report["masters"].get("frequent_moves", report["masters"].get("top_moves", [])))

        print("\nMejores jugadas según rendimiento histórico (Maestros):")
        self._print_move_block(report["masters"].get("best_moves", []), show_expected_score=True)

        print("\nPartidas destacadas de maestros:")
        for i, game in enumerate(report["masters"]["top_games"], start=1):
            month = str(game["month"]).zfill(2) if game["month"] is not None else "??"
            year = game["year"] if game["year"] is not None else "????"
            print(
                f"{i}. {game['white_name']} ({game['white_rating']}) vs "
                f"{game['black_name']} ({game['black_rating']}) | "
                f"{year}-{month} | ganador: {game['winner']}"
            )
