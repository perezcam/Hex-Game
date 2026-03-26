from __future__ import annotations

import math
import random
import heapq
import time

from collections import defaultdict, deque
from player import Player
from board import HexBoard

# Geometría even-r

_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1,  0)]
_ODD  = [(-1,  0), (-1, 1), (0, -1), (0, 1), (1,  0), (1,  1)]
_INF  = 10 ** 9

def _other(p: int) -> int:
    return 2 if p == 1 else 1

def _nbrs(r: int, c: int, n: int):
    for dr, dc in (_EVEN if r % 2 == 0 else _ODD):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield nr, nc

# Dijkstra + celdas críticas  (usado una vez por turno, no por simulación)

def _dijkstra(g: list, player: int, n: int, rev: bool = False) -> list:
    """
    dist[r][c] = coste mínimo desde el lado fuente hasta (r,c) inclusive.
    Coste: 0=piedra propia, 1=vacía, INF=rival.
    """
    opp  = _other(player)
    dist = [[_INF] * n for _ in range(n)]
    pq: list = []
    if player == 1:
        src_c = n - 1 if rev else 0
        for r in range(n):
            v = g[r][src_c]
            if v == opp:
                continue
            w = 0 if v == player else 1
            if w < dist[r][src_c]:
                dist[r][src_c] = w; heapq.heappush(pq, (w, r, src_c))
    else:
        src_r = n - 1 if rev else 0
        for c in range(n):
            v = g[src_r][c]
            if v == opp:
                continue
            w = 0 if v == player else 1
            if w < dist[src_r][c]:
                dist[src_r][c] = w; heapq.heappush(pq, (w, src_r, c))
    while pq:
        d, r, c = heapq.heappop(pq)
        if d != dist[r][c]:
            continue
        for nr, nc in _nbrs(r, c, n):
            v = g[nr][nc]
            if v == opp:
                continue
            nd = d + (0 if v == player else 1)
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd; heapq.heappush(pq, (nd, nr, nc))
    return dist

def _opt(dist: list, player: int, n: int) -> int:
    return (min(dist[r][n-1] for r in range(n)) if player == 1
            else min(dist[n-1][c] for c in range(n)))

def _critical_cells(g: list, player: int, n: int) -> frozenset:
    """
    Celdas vacías en algún camino de coste mínimo.
    Condición: fwd[r][c] + bwd[r][c] == opt + 1  (celda vacía cuesta 1).
    """
    fwd = _dijkstra(g, player, n, rev=False)
    bwd = _dijkstra(g, player, n, rev=True)
    opt = _opt(fwd, player, n)
    if opt == _INF:
        return frozenset()
    thr = opt + 1
    return frozenset(
        (r, c) for r in range(n) for c in range(n)
        if g[r][c] == 0
        and fwd[r][c] != _INF and bwd[r][c] != _INF
        and fwd[r][c] + bwd[r][c] == thr
    )

def _check_win(g: list, player: int, n: int) -> bool:
    vis: set = set(); dq: deque = deque()
    if player == 1:
        for r in range(n):
            if g[r][0] == 1: vis.add((r, 0)); dq.append((r, 0))
        while dq:
            r, c = dq.popleft()
            if c == n - 1: return True
            for nr, nc in _nbrs(r, c, n):
                if (nr, nc) not in vis and g[nr][nc] == 1:
                    vis.add((nr, nc)); dq.append((nr, nc))
    else:
        for c in range(n):
            if g[0][c] == 2: vis.add((0, c)); dq.append((0, c))
        while dq:
            r, c = dq.popleft()
            if r == n - 1: return True
            for nr, nc in _nbrs(r, c, n):
                if (nr, nc) not in vis and g[nr][nc] == 2:
                    vis.add((nr, nc)); dq.append((nr, nc))
    return False

# MCTS Node — con inicialización de RAVE priors

class _Node:
    __slots__ = ("parent", "move", "to_move", "pjm",
                 "children", "untried", "v", "w", "av", "aw")

    def __init__(self, parent, move, to_move: int, pjm: int, legal: list):
        self.parent  = parent
        self.move    = move
        self.to_move = to_move
        self.pjm     = pjm
        self.children: dict = {}
        self.untried: list  = list(legal)
        self.v = 0
        self.w = 0.0
        self.av: dict = defaultdict(int)
        self.aw: dict = defaultdict(float)

class SmartPlayer(Player):
    C_UCT      = 0.05    
    RAVE_K     = 300.0   
    HARD_LIMIT = 4.8    

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.rng = random.Random()
        self._nbr_cache: dict = {}

    def play(self, board: HexBoard) -> tuple:
        t0  = time.perf_counter()          
        n   = board.size
        g   = [row[:] for row in board.board]
        pid = self.player_id
        opp = _other(pid)

        legal = [(r, c) for r in range(n) for c in range(n) if g[r][c] == 0]
        if not legal:
            raise RuntimeError("sin movimientos legales")
        if len(legal) == 1:
            return legal[0]

        self._precache(n)

        hard_deadline = t0 + self.HARD_LIMIT   # tiempo máximo

        #  táctica inmediata (se omite si el tablero es tan grande que
        #  el tiempo ya es escaso antes de empezar el MCTS) 
        #  reservamos al menos 0.4 s para MCTS; si no hay margen, saltamos.
        MCTS_MIN = 0.4
        win = blk = None
        if time.perf_counter() < hard_deadline - MCTS_MIN:
            win = self._win_move(g, n, pid, hard_deadline - MCTS_MIN)
        if win:
            return win
        if time.perf_counter() < hard_deadline - MCTS_MIN:
            blk = self._win_move(g, n, opp, hard_deadline - MCTS_MIN)
        if blk:
            return blk

        # dijkstra ONCE  
        joint = frozenset()
        if time.perf_counter() < hard_deadline - MCTS_MIN:
            my_crit  = _critical_cells(g, pid, n)
            opp_crit = _critical_cells(g, opp, n)
            joint    = my_crit | opp_crit   # frozenset pasa a cada nodo

        # presupuesto MCTS: adaptativo pero nunca supera el tiempo restante 
        elapsed  = time.perf_counter() - t0
        budget   = min(self._budget(g, n), self.HARD_LIMIT - elapsed)
        deadline = time.perf_counter() + max(budget, 0.05) 

        root = _Node(None, None, pid, opp, legal)
        crit_first  = [m for m in legal if m in joint]
        non_crit    = [m for m in legal if m not in joint]
        self.rng.shuffle(non_crit)
        root.untried = non_crit + crit_first   

        while time.perf_counter() < deadline:
            sim_g = [row[:] for row in g]
            self._simulate(root, sim_g, n, joint)

        best = max(root.children.values(), key=lambda ch: ch.v)
        return best.move

    #  MCTS 

    def _simulate(self, root: _Node, g: list, n: int, joint: frozenset = frozenset()):
        node = root
        path: list = [(node, 0)]
        rec:  list = []

        # 1-selección
        while not node.untried and node.children:
            mv, node = self._uct_select(node)
            g[mv[0]][mv[1]] = node.pjm
            rec.append((node.pjm, mv))
            path.append((node, len(rec)))

        # 2-expansión 
        if node.untried:
            mv = node.untried.pop()
            g[mv[0]][mv[1]] = node.to_move
            rec.append((node.to_move, mv))

            child_legal = [(r, c) for r in range(n) for c in range(n) if g[r][c] == 0]
            child = _Node(node, mv, _other(node.to_move), node.to_move, child_legal)
            child_crit = [m for m in child_legal if m in joint]
            child_non  = [m for m in child_legal if m not in joint]
            self.rng.shuffle(child_non)
            child.untried = child_non + child_crit
            node.children[mv] = child
            node = child
            path.append((node, len(rec)))

        # 3-rollout — bridge + aleatorio
        winner = self._rollout(g, node.to_move, n, rec)

        # 4-backprop + RAVE
        for nd, s in path:
            nd.v += 1
            if winner == nd.pjm:
                nd.w += 1.0
            seen: set = set()
            for pl, mv in rec[s:]:
                if pl == nd.to_move and mv not in seen:
                    seen.add(mv)
                    nd.av[mv] += 1
                    if winner == nd.to_move:
                        nd.aw[mv] += 1.0

    def _uct_select(self, node: _Node):
        # garantiza exploración positiva incluso con pocos visits
        log_v  = math.log(max(1, node.v) + 1)
        best_s = -1e18
        best_mv = best_ch = None

        for mv, ch in node.children.items():
            q    = ch.w / ch.v if ch.v else 0.5
            av   = node.av[mv]
            rq   = node.aw[mv] / av if av else 0.5
            beta = self.RAVE_K / (self.RAVE_K + ch.v)
            bl   = (1.0 - beta) * q + beta * rq
            ex   = (float("inf") if ch.v == 0
                    else self.C_UCT * math.sqrt(log_v / ch.v))
            s = bl + ex
            if s > best_s:
                best_s = s; best_mv = mv; best_ch = ch

        return best_mv, best_ch

    # rollout rápido  bridge + aleatorio, swap-and-pop O(1) 

    def _rollout(self, g: list, to_move: int, n: int, rec: list) -> int:
      
        avail   = [(r, c) for r in range(n) for c in range(n) if g[r][c] == 0]
        pos     = {mv: i for i, mv in enumerate(avail)}
        avset   = set(avail)
        current = to_move
        last_mv = rec[-1][1] if rec else None

        while avail:
            mv = None

            # Bridge-reply
            if last_mv is not None:
                mv = self._bridge_reply(g, current, last_mv, n, avset)

            # Aleatorio
            if mv is None:
                mv = avail[self.rng.randrange(len(avail))]

            # Swap-and-pop 
            idx       = pos[mv]
            last_item = avail[-1]
            avail[idx] = last_item
            pos[last_item] = idx
            avail.pop()
            del pos[mv]
            avset.discard(mv)

            g[mv[0]][mv[1]] = current
            rec.append((current, mv))
            last_mv = mv
            current = _other(current)

        return 1 if _check_win(g, 1, n) else 2

    def _bridge_reply(self, g: list, player: int, last_mv: tuple,
                      n: int, avset: set):
        lr, lc = last_mv
        own = [(nr, nc) for nr, nc in self._nbrs_of(lr, lc, n)
               if g[nr][nc] == player]
        if len(own) < 2:
            return None
        for i in range(len(own)):
            a = own[i]
            for j in range(i + 1, len(own)):
                b = own[j]
                if b in set(self._nbrs_of(a[0], a[1], n)):
                    continue
                common = (set(self._nbrs_of(a[0], a[1], n))
                          & set(self._nbrs_of(b[0], b[1], n)))
                if last_mv not in common:
                    continue
                empties = [c for c in common if c != last_mv and c in avset]
                if len(empties) == 1:
                    return empties[0]
        return None


    def _win_move(self, g: list, n: int, player: int, deadline: float = 0.0):
        for r in range(n):
            if deadline and time.perf_counter() > deadline:
                return None        
            for c in range(n):
                if g[r][c] == 0:
                    g[r][c] = player
                    won = _check_win(g, player, n)
                    g[r][c] = 0
                    if won:
                        return (r, c)
        return None


    def _budget(self, g: list, n: int) -> float:
        e = sum(1 for row in g for v in row if v == 0)
        r = e / (n * n)
        if r > 0.75:
            return 3.5   
        if r > 0.40:
            return 4.0  
        return 4.5        


    def _precache(self, n: int):
        if n in self._nbr_cache:
            return
        self._nbr_cache[n] = {
            (r, c): list(_nbrs(r, c, n))
            for r in range(n) for c in range(n)
        }

    def _nbrs_of(self, r: int, c: int, n: int) -> list:
        return self._nbr_cache[n][(r, c)]
