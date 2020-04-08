# 3x3x3 Rubik Cube routines by dm (2020/03/18 - ...)
#
#
# notation similar to pypi.org/project/rubik-solver
#
#
# YBRGOW face colors: [Y]ellow ->0  [B]lue ->1  [R]ed ->2  [G]reen ->3  [O]range ->4  [W]hite -> 5
#
# Keep:     4 (Upper center) = YELLOW
#          13 (Left center)  = BLUE
#          22 (Front center) = RED
#          31 (Right center) = GREEN
#          40 (Back center)  = ORANGE
#          49 (Down center)  = WHITE
#
# Note: in Kociemba's code, YBRGOW -> ULFRBD, and facelet data is ordered in URFDLB order (instead of ULFRBD)
#
# State geometry:
#
#               ----------------
#               | 0  | 1  | 2  | 
#               ----------------   UP
#               | 3  | 4  | 5  |
#               ----------------
#   LEFT        | 6  | 7  | 8  |     RIGHT          BACK
#-------------------------------------------------------------
#| 9  | 10 | 11 | 18 | 19 | 20 | 27 | 28 | 29 | 36 | 37 | 38 |
#---------------------FRONT-----------------------------------
#| 12 | 13 | 14 | 21 | 22 | 23 | 30 | 31 | 32 | 39 | 40 | 41 |
#-------------------------------------------------------------
#| 15 | 16 | 17 | 24 | 25 | 26 | 33 | 34 | 35 | 42 | 43 | 44 |
#-------------------------------------------------------------
#               | 45 | 46 | 47 |
#               ----------------
#               | 48 | 49 | 50 |  DOWN
#               ----------------
#               | 51 | 52 | 53 |
#               ----------------
#
#
# Corners:    0-1           Edges:   +0+  
#             | |  UP                3 1
#           0-3-2-                 +3+2+1+-
#           | | |  FRONT           4 5 6 7  
#           4-7-6-                 +b+a+9+-       a = 10, b = 11
#             | |  DOWN              b 9
#             4-5                    +8+
#
#


import sys, numpy as np, itertools


class RubikCube():

   n = 54

   ###
   # face/color coding
   ###

   # face color -> face idx
   facecolor2int_map = { 'Y': 0, 'B': 1, 'R': 2, 'G': 3, 'O': 4, 'W': 5 }   
 
   # face idx -> face color
   int2facecolor_map = {}
   for (k,v) in facecolor2int_map.items():  int2facecolor_map[v] = k

   # facelet name -> facelet idx
   facelet2int_map  = { 'a':  0, 'b':  1, 'c':  2, 'd':  3, 'e':  4,
                        'f':  5, 'g':  6, 'h':  7, 'i':  8, 'j':  9,
                        'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
                        'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
                        'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24,
                        'z': 25, '0': 26, '1': 27, '2': 28, '3': 29,
                        '4': 30, '5': 31, '6': 32, '7': 33, '8': 34,
                        '9': 35, 'A': 36, 'B': 37, 'C': 38, 'D': 39,
                        'E': 40, 'F': 41, 'G': 42, 'H': 43, 'I': 44,
                        'J': 45, 'K': 46, 'L': 47, 'M': 48, 'N': 49,
                        'O': 50, 'P': 51, 'Q': 52, 'R': 53
                      }

   # facelet idx -> facelet name
   int2facelet_map = {}
   for (k,v) in facelet2int_map.items(): int2facelet_map[v] = k

   # facelet idx -> face idx
   def flidx2face(i):
      return (i // 9)

   # facelet idx -> face color
   flidx2face_map = {}
   for i in range(n): flidx2face_map[i] = int2facecolor_map[flidx2face(i)]
   
   # facelet/color name to integer value
   def color2int(c, c2imap = facelet2int_map):
      return c2imap.get(c, 0)

   # facelet idx to facelet/color name
   def int2color(v, i2cmap = int2facelet_map):
      return i2cmap.get(v, "*")

   ###
   # centers, corners and edges
   ###

   center_indices = (  4, 13, 22, 31, 40, 49 );   # U, L, F, R, B, D

   edge_indices = ( ( 1, 37), ( 5, 28), ( 7, 19), ( 3, 10),   #UB, UR, UF, UL
                    (12, 41), (14, 21), (23, 30), (32, 39),   #LB, LF, RF, RB
                    (43, 52), (34, 50), (25, 46), (16, 48)    #DB, DR, DF, DL
                  )

   corner_indices = ( ( 0,  9, 38),  ( 2, 36, 29),      # UBL, UBR
                      ( 8, 27, 20),  ( 6, 18, 11),      # UFR, UFL
                      (51, 44, 15),  (53, 35, 42),      # DBL, DBR
                      (47, 26, 33),  (45, 17, 24)       # DFR, DFL
                    )

   color2center_map = {}
   for i in center_indices:   color2center_map[flidx2face(i)] = i

   color2edge_map = {}
   for (i,j) in edge_indices:
      fi,fj = flidx2face(i), flidx2face(j)
      color2edge_map[(fi,fj)] = (i,j)
      color2edge_map[(fj,fi)] = (j,i)

   color2corner_map = {}
   for v in corner_indices:
      for (i,j,k) in itertools.permutations(v):
         fi,fj,fk = flidx2face(i), flidx2face(j), flidx2face(k)  
         color2corner_map[(fi,fj,fk)] = (i,j,k)

   edge_permutation_map = {}
   for idx, (i,j) in enumerate(edge_indices):
      edge_permutation_map[(i,j)] = (idx, 0)
      edge_permutation_map[(j,i)] = (idx, 1)

   corner_permutation_map = {}
   for idx, (i,j,k) in enumerate(corner_indices): # idx 0: UBL, 1: UBR, 2: UFR, 3: UFL, 4:DBL, 5: DBR, 6: DFR, 7: DFL
      corner_permutation_map[(i,j,k)] = (idx, 0)
      corner_permutation_map[(j,k,i)] = (idx, 1)
      corner_permutation_map[(k,i,j)] = (idx, 2)
      corner_permutation_map[(k,j,i)] = (idx, 3)  # 3-5 are mirrorred, so impossible on normal cube
      corner_permutation_map[(j,i,k)] = (idx, 4)
      corner_permutation_map[(i,k,j)] = (idx, 5)


   ####
   # init
   ####

   def __init__(self, s = ""):
      # allocate state
      self.state = np.zeros((RubikCube.n,), dtype = int)
      if s == "":   # start from solved config for empty string
         self.reset()
      else:         # init from string
         self.setState(RubikCube.string2state(s))

   def reset(self):
      for i in range(self.n):  self.state[i] = i

   # state related

   # whether cube is in solved state - FIXME: is it faster to use isInState() ??
   def isReset(self):
      for i in range(self.n):
         if self.state[i] != i:
            return False
      return True

   # whether state is same as that of another cube
   def isSame(self, cube):
      return self.isInState(cube.state)

   # get state as tuple of ints
   def getState(self):
      return tuple( [ self.state[i] for i in range(self.n) ] )

   # set state directly  -  FIXME: no value or length checks!!
   def setState(self, tpl):
      for i in range(self.n):
         self.state[i] = tpl[i]
        
   def isInState(self, tpl):
      for i in range(self.n):
         if self.state[i] != tpl[i]:
            return False
      return True

   # printing

   FORMAT_DEFAULT = 0
   FORMAT_FACESONLY = 1
   FORMAT_KOCIEMBA = 2
   FORMAT_RUBIKSCUBESOLVER = 3 
   #curl http://localhost:8080/...
   #https://rubiks-cube-solver.com/solution.php?cube=0...

   color_remap_Kociemba = { "Y": "U", "B": "L", "R": "F", "G": "R", "O": "B", "W": "D" }
   color_remap_RCS      = { "Y": "1", "B": "2", "R": "3", "G": "4", "O": "5", "W": "6" }

   color_invmap_Kociemba = {}
   for (k,v) in color_remap_Kociemba.items():  color_invmap_Kociemba[v] = k
   color_invmap_RCS = {}
   for (k,v) in color_remap_RCS.items():  color_invmap_RCS[v] = k

   # print state as string - in various formats
   def toString(self, format = 0):
      if format == RubikCube.FORMAT_DEFAULT:
         return "".join( [ RubikCube.int2color(v, RubikCube.int2facelet_map) for v in self.state ]  )
      elif format == RubikCube.FORMAT_FACESONLY:
         return "".join( [ RubikCube.int2color(v, RubikCube.flidx2face_map) for v in self.state ]  )
      elif format == RubikCube.FORMAT_KOCIEMBA:  # Kociemba's
         ret = self.toString(RubikCube.FORMAT_FACESONLY)
         ret = ret[:9] + ret[27:36] + ret[18:27] + ret[45:] + ret[9:18] + ret[36:45]   # URFDLB face order
         return "".join( [RubikCube.color_remap_Kociemba[c] for c in ret ] )
      else:  # rubiks-cube-solver.com site
         ret = self.toString(RubikCube.FORMAT_FACESONLY)
         return "".join( [RubikCube.color_remap_RCS[c] for c in ret ] )

   def _range2string(self, start, end, facesonly = False):
      i2cmap = RubikCube.int2facelet_map  if not facesonly  else RubikCube.flidx2face_map
      return "".join( [ RubikCube.int2color(v, i2cmap) for v in self.state[start:end] ] )

   def print(self, facesonly = False):
      for row in range(3):   # top face
         start = row * 3
         print("   |" + self._range2string(start, start + 3, facesonly) + "|")
      print("    ---")
      for row in range(3): # left, mid, right, back
         s = ""
         for face in range(1, 5): 
            start = face * 9 + row * 3
            s += self._range2string(start, start + 3, facesonly)
            if face != 4:
               s += "|"
         print(s)
      print("    ---")
      for row in range(3):   # bottom face
         start = 5 * 9 + row * 3
         print("   |" + self._range2string(start, start + 3, facesonly) + "|")

   #
   # basic moves (L R F B U D)
   #

   # cycle positions given by indices
   def _cycleCells(self, shft, indices):
      state = self.state
      tmp = tuple( state[idx] for idx in indices )
      ncycle = len(indices)
      for i in range(ncycle):
         state[indices[(i + shft) % ncycle]] = tmp[i]
            

   # rotate one face clockwise:   012     630
   #                              345 ->  741
   #                              678     852
   def _rotateFace(self, face, dir = 1):
      pos = 9 * face   # top left corner of face
      self._cycleCells(2 * dir, (pos, pos + 1, pos + 2, pos + 5, pos + 8, pos + 7, pos + 6, pos + 3) )
         
    
   # left
   
   def _moveL(self, dir = 1): # clockwise turn of LEFT face (around #13)
      self._rotateFace(1, dir)   # face
      # neighbors: 0, 3, 6|18,21,24|45,48,51|44,41,38| -> rotate by 3 for CW
      self._cycleCells(3 * dir, (0, 3, 6, 18, 21, 24, 45, 48, 51, 44, 41, 38) )

   def _moveLinv(self):   # L'
       self._moveL(-1)

   def _moveL2(self):
       self._moveL(dir = 2)

   # right
 
   def _moveR(self, dir = 1): #clockwise turn of RIGHT face (around #31)
      self._rotateFace(3, dir)   # face
      # neighbors: 53,50,47|26,23,20| 8, 5, 2|36,39,42| -> rotate by 3 for CW
      self._cycleCells(3 * dir, (53, 50, 47, 26, 23, 20, 8, 5, 2, 36, 39, 42) )

   def _moveRinv(self):       # R'
       self._moveR(dir = -1)

   def _moveR2(self):
       self._moveR(dir = 2)

   # front
 
   def _moveF(self, dir = 1): #clockwise turn of FRONT face (around #22)
      self._rotateFace(2, dir)   # face
      # neighbors:  6, 7, 8|27,30,33|47,46,45|17,14,11| -> rotate by 3 for CW
      self._cycleCells(3 * dir, (6, 7, 8, 27, 30, 33, 47, 46, 45, 17, 14, 11) )

   def _moveFinv(self):      # F'
       self._moveF(dir = -1)

   def _moveF2(self):
       self._moveF(dir = 2)

   # back
 
   def _moveB(self, dir = 1): #clockwise turn of BACK face (around #40)
      self._rotateFace(4, dir)   # face
      # neighbors: 35,32,29| 2, 1, 0| 9,12,15|51,52,53| -> rotate by 3 for CW
      self._cycleCells(3 * dir, (35, 32, 29, 2, 1, 0, 9, 12, 15, 51, 52, 53) )

   def _moveBinv(self):       # B'
       self._moveB(dir = -1)

   def _moveB2(self):
       self._moveB(dir = 2)

   # up
 
   def _moveU(self, dir = 1): #clockwise turn of UP face (around #4)
      self._rotateFace(0, dir)   # face
      # neighbors: 11,10, 9|38,37,36|29,28,27|20,19,18| -> rotate by 3 for CW
      self._cycleCells(3 * dir, (11, 10, 9, 38, 37, 36, 29, 28, 27, 20, 19, 18) )

   def _moveUinv(self):        # U'
       self._moveU(dir = -1)

   def _moveU2(self):
       self._moveU(dir = 2)

   # down
 
   def _moveD(self, dir = 1): #clockwise turn of DOWN face (around #49)
      # face
      self._rotateFace(5, dir)
      # neighbors: 15,16,17|24,25,26|33,34,35|42,43,44 -> rotate by 3 for CW
      self._cycleCells(3 * dir, (15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44) )

   def _moveDinv(self):        # D'
       self._moveD(dir = -1)

   def _moveD2(self):
       self._moveD(dir = 2)    # D + D

   #
   # sequence of basic moves (L->1 R->2 F->3 B->4 U->5 D->6)
   #
   # NOTE: we apply moves in left-to-right order
   #       whereas in operator notation, things would go from right to left

   int2move = { 1: _moveL,     2: _moveR,     3: _moveF, 4: _moveB, 5: _moveU, 6: _moveD,    # basic moves
               -1: _moveLinv, -2: _moveRinv, -3: _moveFinv,        # inverses 
               -4: _moveBinv, -5: _moveUinv, -6: _moveDinv,
               11: _moveL2, 12: _moveR2, 13: _moveF2, 14: _moveB2, 15: _moveU2, 16: _moveD2  # double moves
              }

   def move(self, mv, count = 1):   # apply moves (sequence or str)
      sequence = mv   if not isinstance(mv, str)  else   RubikCube.string2moves(mv)
      if count < 0: # interpret negative counts as inverse 
         sequence = RubikCube.invertMoves(sequence)
         count = -count
      for _ in range(count):
         for code in sequence:
            if code == 0:
               continue
            m = self.int2move[code]   # fetch move
            m(self)                   # apply it

   def invertMoves(mv):   # invert moves
      sequence = mv   if not isinstance(mv, str)  else   RubikCube.string2moves(mv)
      return [ c if c > 10  else -c   for c in sequence[::-1] ]

   def invMove(self, mv):  # apply inverse of moves
      self.move(RubikCube.invertMoves(mv))

   # whether two moves (sequence or string) are equivalent
   def areEqualMoves(mv1, mv2):
      cube = RubikCube()
      cube.move(mv1)
      cube.invMove(mv2)
      return cube.isReset()


   # parsing sequences
   
   str2move_map = { "L":   1,  "R":   2,  "F":   3,  "B":   4,  "U":   5,  "D":   6,
                    "L'": -1,  "R'": -2,  "F'": -3,  "B'": -4,  "U'": -5,  "D'": -6,
                    "L2": 11,  "R2": 12,  "F2": 13,  "B2": 14,  "U2": 15,  "D2": 16
                  }
   move2str_map = {}
   for k,v in str2move_map.items():  move2str_map[v] = k

   str2move_map.update(   # add alternative move names to dictionary
                  { "L1":  1,  "R1":  2,  "F1":  3,  "B1":  4,  "U1":  5,  "D1":  6,
                    "L3": -1,  "R3": -2,  "F3": -3,  "B3": -4,  "U3": -5,  "D3": -6 }
                       )

   def string2moves(s):
       return [ RubikCube.str2move_map[w.strip()] for w in s.split(" ") ]
       
   def moves2string(seq):
       return " ".join( [ RubikCube.move2str_map[m] for m in seq ] )

   def invertMoves2String(mv):
      invseq = RubikCube.invertMoves(mv)
      return RubikCube.moves2string(invseq)

   def _areCentersBad(tpl):
      for pos in RubikCube.center_indices:
         if tpl[pos] != pos:
            return True
      return False   

   # convert a 27-character string to a cube state
   def string2state(s): # FIXME: does not check twist/flip parity
      n = RubikCube.n
      if len(s) != n:
         raise ValueError("Incorrect string length " + str(len(s)))
      if s.count("1") == 9:  # map RCS colors to YBRGOW 
         s = "".join(RubikCube.color_invmap_RCS[c]  for c in s) 
      elif s.count("U") == 9:  # map Kociemba colors/indices to YBRGOW
         s = s[:9] + s[36:45] + s[18:27] + s[9:18] + s[45:] + s[27:36]
         s = "".join(RubikCube.color_invmap_Kociemba[c]  for c in s) 
      # if only faces YBRGOW 
      if s.count("Y") == 9:
         lst = [RubikCube.facecolor2int_map[v] for v in s]
         # centers
         for i in RubikCube.center_indices:
            lst[i] = RubikCube.color2center_map[lst[i]]
         # edges
         for (i, j) in RubikCube.edge_indices:
            (vi, vj) = RubikCube.color2edge_map[(lst[i], lst[j])]
            lst[i], lst[j] = vi, vj
         # corners
         for (i, j, k) in RubikCube.corner_indices:
            (vi, vj, vk) = RubikCube.color2corner_map[(lst[i], lst[j], lst[k])]
            lst[i], lst[j], lst[k] = vi, vj, vk
      # otherwise, assume 27 cells a-z,0-9,A-R
      else:  
         lst = [-1] * n
         for i in range(n):
            if lst[i] != -1:
               raise ValueError("Duplicate symbol " + str(s[i]))
            lst[i] = RubikCube.facelet2int_map[s[i]]
      if RubikCube._areCentersBad(lst):
         raise ValueError("Centers wrong")
      return tuple(lst) 
  
   # permutations

   # apply permutation  - tpl[i] = j gives which state[j] moves to state[i]
   def permute(self, tpl):     
      state2 = self.getState()    # copy current state
      for i in range(self.n):
         self.state[i] = state2[tpl[i]]

   # apply inverse permutation  - tpl[i] = j gives which state[i] moves to state[j]
   def invPermute(self, tpl):
      state2 = self.getState()
      for i in range(self.n):
         self.state[tpl[i]] = state2[i]

   # give permutation that reaches the current state from the solved cube
   def state2permutation(self):
      return tuple( self.getState() )

   # convert moves (sequence/string) to a permutation
   def moves2permutation(mv):
      cube = RubikCube()
      cube.move(mv)
      return cube.state2permutation()

   # convert moves (sequence/string) to a state - starting from the solved cube
   def moves2state(mv):
      cube = RubikCube()
      cube.move(mv)
      return cube.getState()

   # read off edge permutations
   def getEdgePermutation(self):
      state = self.state
      res = []
      for (i,j) in RubikCube.edge_indices:
         pattern = (state[i], state[j])
         res.append(RubikCube.edge_permutation_map[pattern])
      return res

   # read off corner permutations
   def getCornerPermutation(self):
      state = self.state
      res = []
      for (i,j,k) in RubikCube.corner_indices:
         pattern = (state[i], state[j], state[k])
         res.append(RubikCube.corner_permutation_map[pattern])
      return res

   #
   # group stuff
   #

   # find order of moves (sequence/string) - apply it until we get back original cube
   def findOrder(mv):
      cube = RubikCube()
      # convert move to permutation
      cube.reset()                
      cube.move(mv)
      perm = cube.state2permutation()
      # apply perm repeatedly   
      order = 1
      while True:
         if cube.isReset():
            return order
         cube.permute(perm)
         order += 1


   # rigid rotations
   #
   # 24 configurations (e.g., 8 positions for a given corner * 3 rotations about that corner
   #                    or, 6 positions for a given face * 4 rotations about the normal to that face)
   #
   # -> choose 6x4, so index = (position of originally "L" face) * 4 + (CW rotation / 90 degrees)

   # LFURBD relabelings for rotated configurations
   # - the LFU normals form a righthanded system
   # - normals for R,B,D are always opposite to L,F,U
   rigid_rotation_mapped_moves = (
                  "LFURBD", "LUBRDF", "LBDRFU", "LDFRUB",   #L in place, FUBD rotate
                  "BLUFRD", "BURFDL", "BRDFLU", "BDLFUR",   #B->L, then LURD rotate
                  "RBULFD", "RUFLDB", "RFDLBU", "RDBLUF",   #R->L, then BUFD rotate
                  "FRUBLD", "FULBDR", "FLDBRU", "FDRBUL",   #F->L, then RULD rotate
                  "UFRDBL", "URBDLF", "UBLDFR", "ULFDRB",   #U->L, then FRBL rotate
                  "DFLUBR", "DLBURF", "DBRUFL", "DRFULB"    #D->L, then FLBR rotate
                )

   rigid_rotation_move_str_map = [{} for _ in range(24)]
   for rot in range(24):
      remap = rigid_rotation_mapped_moves
      for i in range(6):
         rigid_rotation_move_str_map[rot][remap[0][i]] = remap[rot][i]
   rigid_rotation_move_str_map = tuple(rigid_rotation_move_str_map)

   def rotateMoveString(str, rot_idx):
      if rot_idx == 0:
         return str
      remap = RubikCube.rigid_rotation_move_str_map[rot_idx];
      return "".join( [ remap.get(c)   if remap.get(c) != None  else c   for c in str] )


   # elementary edge and corner moves
   #
   # based in part on "Group Theory and the Rubik's Cube" by Janet Chen
   #   http://people.math.harvard.edu/~jjchen/docs/Group%20Theory%20and%20the%20Rubik's%20Cube.pdf
   #
   # Parity of (unoriented) edge permutations and (unoriented) corner permutations must match.
   # (e.g., if precisely two edges are swapped (->odd), then at least two corners must be swapped as well.)
   #
   # Even edge permutations and even corner permutations form commuting subgroups (orientations may change).
   # Edge orientations and corner orientations also form commuting subgroups.
   #
   # => one can reach a given state by adjusting edge permutations, then corner permutations, then edge orientations, 
   #    and finally corner orientations
   # 
   
   # cycle edges 0,1,2->2,0,1 (edges on same face)
   #             0,1,4->4,0,1 and 0,1,6->6,0,1 (chain of edges on two neighboring faces - two parities)
   #             0,1,5->5,0,1 (two neighbors, third starting from corner on same face)
   #             0,1,7->7,0,1 (edges from same corner)
   #             0,1,8->8,0,1 and 0,1,9->9,0,1 (two neighbors, third across one on same face - two parities)
   #             0,1,10->10,0,1 and 0,1,11->11,0,1 (two neighbors + mirror of one across cube center - two parities)
   #             0,2,8->8,0,2 (all three parallel)
   #             0,2,9->9,0,2 (two across on same face, third orthogonal on opposite face)
   #             0,5,9->9,0,5 and 0,6,11->11,0,6 (none are on same face - two parities)
   edge_cycle_012_str = "R2 U' B2 R2 B2 U B' F' U2 B F U2 R2"  
   edge_cycle_014_str = "U2 F D2 U2 B' U' B D2 U2 F' U'"
   edge_cycle_016_str = "U2 L2 U' B2 F2 U F2 U' B2 U2 L' U2 F2 U2 L' U"
   edge_cycle_015_str = "B R2 D2 B F L2 D2 R2 U' R2 F2 R2 U' F"
   edge_cycle_017_str = "U2 F' D2 U2 B U' B' D2 U2 F U'"
   edge_cycle_018_str = "F2 L' R U' B' U L R' F' U F'"
   edge_cycle_019_str = "L2 R2 F2 L2 U R2 U' L2 F2 L' R' F' U2 F L' R'"
   edge_cycle_01a_str = "B U2 B F2 L2 R2 B2 F2 L' F' D' F2 D L' R2 U"
   edge_cycle_01b_str = "L2 R2 U R2 B2 R2 B2 R2 U' L R' F U2 F' L R'"
   edge_cycle_028_str = "U2 F2 L2 F2 L2 U' L2 F2 L' R' F' D2 F' L' R U'"
   edge_cycle_029_str = "R2 U' R2 U2 B2 R2 B2 U2 R2 U L R' F' U2 F L' R'"
   edge_cycle_059_str = "R2 F R2 D2 R2 F2 L2 U2 R2 B R' B L2 R2 F' R'"
   edge_cycle_06b_str = "R2 D R2 U R2 D' U F2 U' B' D L D2 L' B F2 R' U'"

   # storage for all 12*11*10 = 1320 three-edge cycles - initialized after class code
   edge_cycles = [ [ [None]*12 for __ in range(12) ] for _ in range(12) ]

   # flip orientations of edges 0,1 (neighbors), 0,2 (across same face), 
   #                            0,5 & 0,6 (across neighbor), and 0,10 (across cube)
   edge_flip_01_str = "R U F R2 F' R' U F2 R2 U2 F2 U2 R2 F U2 F"
   edge_flip_02_str = "U2 R2 D L2 F2 R2 U L2 B2 R' U' F U F2 R F U' F'"
   edge_flip_05_str = "D2 F2 R2 U2 R2 F2 D' L2 D' B' L' D' B2 D B L'"
   edge_flip_06_str = "U' B2 U' F2 D' L2 D F2 U2 B R' D' R2 D B R'"
   edge_flip_0a_str = "U B2 L2 D' F2 L2 B2 U2 R2 U' L R' F R2 F L' R' U'"

   # storage for all 12 * 11 = 66*2 two-edge flips - initialized after class code
   # - [i][j] gives a move that flips the orientation of the i-th and j-th edges
   edge_flips = [ [None]*12  for _ in range(12) ]

   # basic 3-corner cycles that generate even corner permutations
   #
   # cycle corners 0,1,2->2,1,0 (corners on same face) -> 8*3*6 = 144
   #               0,1,6->6,0,1 (two on same edge, two across same face) ->8*3*6 = 144
   #               0,2,5->5,0,2 (two across same face, another two across same face) -> 8*(3/3)*6 = 48
   corner_cycle_012_str = "L' U R' D2 R U' R' D2 L R"
   corner_cycle_015_str = "F R2 B L2 B' R2 B L2 F' L B' R B L' B' R'"
   corner_cycle_025_str = "F2 L2 F' R2 U2 B D2 R2 B' F D' F' U2 F D' F' R2 F'"

   # storage for all 8*7*6 = 336 three-corner cycles - initialized after class code
   # - [i][j][k] gives a move that cycles ijk -> kij
   corner_cycles = [ [ [None]*8 for __ in range(8) ] for _ in range(8) ]

   # basic 2-corner twists that generate corner orientations
   #
   # twist corners 0,1 (on same edge), 0,2 (across same face), 0,5 (across cube)
   corner_twist_01_str = "D R D L' D' R' D R' F2 R B2 R' F2 R B2 L D2"
   corner_twist_02_str = "U2 F D B2 D' F' D2 B2 D' R2 D' R2 U F2 U F2" 
   corner_twist_06_str = "U B U' F2 U B' U' F2 D B2 D' F2 D B2 D' F2"

   # storage for all 8 * 7 = 28*2 two-corner twists - initialized after class code
   # - [i][j] gives a move with twist=1 for the i-th corner, twist=2 for the j-th one
   corner_twists = [ [None]*8  for _ in range(8) ]

   ##
   # static methods - converted only as last step so that we could use them inside class def above
   ##
   invertMoves        = staticmethod(invertMoves)
   areEqualMoves      = staticmethod(areEqualMoves)
   string2moves       = staticmethod(string2moves)
   moves2string       = staticmethod(moves2string)
   invertMoves2String = staticmethod(invertMoves2String)
   string2state       = staticmethod(string2state)
   moves2permutation  = staticmethod(moves2permutation)
   moves2state        = staticmethod(moves2state)
   findOrder          = staticmethod(findOrder)

	
####
# initialization that did not fit inside class definition
####


# build all basic edge flips: neighbors       -> 12*4 = 48
#                             across face     -> 12*2 = 24
#                             across neighbor -> 12*4 = 48
#                             across cube     -> 12*1 = 12
for rot in range(24):
   moveset = (RubikCube.edge_flip_01_str, 
              RubikCube.edge_flip_02_str,
              RubikCube.edge_flip_05_str, RubikCube.edge_flip_06_str, 
              RubikCube.edge_flip_0a_str)
   edge_flips = RubikCube.edge_flips
   for s in moveset:
      s = RubikCube.rotateMoveString(s, rot)
      seq = RubikCube.string2moves(s)
      cube = RubikCube()
      cube.move(seq)
      p = cube.getEdgePermutation()
      lst = [ i   for i in range(12)   if p[i][1] != 0 ]
      i,j = lst[0],lst[1]
      if edge_flips[i][j] == None:
         edge_flips[i][j] = seq
         edge_flips[j][i] = seq


# build all basic edge cycles: all 3 on same face -> 12 * 2 * 6 = 144 
#                              chain of 3 on neighboring faces -> 2 * (12 * 6) = 144
#                              all 3 from same corner -> 8 * 6 = 48
#                              2 neighbors, 3rd opposite to both from corner on same face -> 8 * 3 * 6 = 144
#                              2 neighbors, 3rd across one on same face -> 2 * (8 * 3 * 6) = 288
#                              2 neighbors + mirror of one across cube center -> 2 * (8 * 3 * 6) = 288
#                              3 parallel ones -> 12 * 6 = 72 
#                              2 across on same face, 3rd orthogonal on opposite face -> 12 * 2 * 6 = 144
#                              none are on same face -> 2 * (4 * 6) = 48
for rot in range(24):
   moveset = (RubikCube.edge_cycle_012_str,
              RubikCube.edge_cycle_014_str, RubikCube.edge_cycle_016_str,
              RubikCube.edge_cycle_015_str,
              RubikCube.edge_cycle_017_str, 
              RubikCube.edge_cycle_018_str, RubikCube.edge_cycle_019_str,
              RubikCube.edge_cycle_01a_str, RubikCube.edge_cycle_01b_str,
              RubikCube.edge_cycle_028_str,
              RubikCube.edge_cycle_029_str,
              RubikCube.edge_cycle_059_str, RubikCube.edge_cycle_06b_str 
             )
   edge_cycles = RubikCube.edge_cycles
   for s in moveset:
      s = RubikCube.rotateMoveString(s, rot)
      seq = RubikCube.string2moves(s)
      seqinv = RubikCube.invertMoves(seq) #inverse cycle
      cube = RubikCube()
      cube.move(seq)
      p = cube.getEdgePermutation()
      lst = [ i for i in range(12)   if p[i][0] != i ]  # reconstruct cycle ijk->jki
      lst[1] = p[lst[2]][0]  
      lst[0] = p[lst[1]][0]
      #print(rot, p, lst)
      i,j,k = lst[0],lst[1],lst[2]        
      if edge_cycles[i][j][k] == None:   # store cycle and inverse
         edge_cycles[i][j][k] = seq
         edge_cycles[j][k][i] = seq
         edge_cycles[k][i][j] = seq
         edge_cycles[k][j][i] = seqinv
         edge_cycles[j][i][k] = seqinv
         edge_cycles[i][k][j] = seqinv


# build all basic corner twists: neighbor twists -> 8*3 = 24
#                                same-face diag twists -> 8*3 = 24
#                                opposite corner twists -> 8*1 = 8
for rot in range(24):
   moveset = (RubikCube.corner_twist_01_str, RubikCube.corner_twist_02_str, RubikCube.corner_twist_06_str)
   corner_twists = RubikCube.corner_twists
   for s in moveset:
      s = RubikCube.rotateMoveString(s, rot)
      seq = RubikCube.string2moves(s)
      seqinv = RubikCube.invertMoves(seq) #inverse cycle
      cube = RubikCube()
      cube.move(seq)
      p = cube.getCornerPermutation()
      lst = [ i   for i in range(8)   if p[i][1] != 0 ]
      i,j = lst[0],lst[1]
      if p[i][1] != 1:
         seq, seqinv = seqinv, seq
      if corner_twists[i][j] == None:
         corner_twists[i][j] = seq
         corner_twists[j][i] = seqinv


# build all basic corner cycles: all 3 on same face -> 8*3*6 = 144
#                                two on same edge, two across same face -> 8*3*6 = 144
#                                two across same face, another two across another face -> 8*(3/3)*6 = 48
for rot in range(24):
   moveset = (RubikCube.corner_cycle_012_str, RubikCube.corner_cycle_015_str, RubikCube.corner_cycle_025_str)
   corner_cycles = RubikCube.corner_cycles
   for s in moveset:
      s = RubikCube.rotateMoveString(s, rot)
      seq = RubikCube.string2moves(s)
      seqinv = RubikCube.invertMoves(seq) #inverse cycle
      cube = RubikCube()
      cube.move(seq)
      p = cube.getCornerPermutation()
      lst = [ i for i in range(8)   if p[i][0] != i ]  # reconstruct cycle ijk->jki
      lst[1] = p[lst[2]][0]  
      lst[0] = p[lst[1]][0]
      #print(rot, p, lst)
      i,j,k = lst[0],lst[1],lst[2]        
      if corner_cycles[i][j][k] == None:   # store cycle and inverse
         corner_cycles[i][j][k] = seq
         corner_cycles[j][k][i] = seq
         corner_cycles[k][i][j] = seq
         corner_cycles[k][j][i] = seqinv
         corner_cycles[j][i][k] = seqinv
         corner_cycles[i][k][j] = seqinv




###
# TESTS
###


def TESTsuite(format = RubikCube.FORMAT_DEFAULT):

   facesonly =  (format != RubikCube.FORMAT_DEFAULT)

   cube = RubikCube()
   print(cube.toString(format))
     
   # init
   print("#initial:")
   cube.print(facesonly)

   # basic moves
   lst = [("L", cube._moveL), ("R", cube._moveR), ("F", cube._moveF), 
          ("B", cube._moveB), ("U", cube._moveU), ("D", cube._moveD)]
   for (move, move_fn) in lst:
      # number of applications
      for i in range(1, 5):
         print("#%s%d:" % (move, i) )
         move_fn()
         cube.print(facesonly)
         print(cube.isReset())
      # inverse move + move
      print("#%sinv:" % move)
      move_fn(-1)
      cube.print(facesonly)
      print("#%sinv + %s:" % (move, move))
      move_fn()
      cube.print(facesonly)

   # sequence of moves
   print("moves:")
   cube.move([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
   cube.print(facesonly)
   print(cube.isReset())

   # states
   print("states:")
   state = cube.getState()
   cube.reset()
   print(state, cube.isInState(state))
   cube.move([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
   print(cube.isInState(state))

   # permutations
   print("permutations:")
   state = cube.getState()
   perm = cube.state2permutation()
   cube.reset()
   cube.permute(perm)
   print(cube.isInState(state))
   cube.move([1, 2, 1, 2, 1, 2])
   print("edge permutation:", cube.getEdgePermutation())
   print("corner permutation:", cube.getCornerPermutation())

   # state to/from string
   print("to/from string:")
   cube.reset()
   for i in range(3):  cube.move([ 1, 2, 3])
   for i in range(3):  cube.move([ 4, 5, 6])
   s = cube.toString(RubikCube.FORMAT_DEFAULT)
   print(s)
   state = RubikCube.string2state(s)
   print(cube.isInState(state))
   s = cube.toString(RubikCube.FORMAT_FACESONLY)
   print(s)
   state = RubikCube.string2state(s)
   print(cube.isInState(state))
   s = cube.toString(RubikCube.FORMAT_KOCIEMBA)
   print(s)
   state = RubikCube.string2state(s)
   print(cube.isInState(state))
   s = cube.toString(RubikCube.FORMAT_RUBIKSCUBESOLVER)
   print(s)
   state = RubikCube.string2state(s)
   print(cube.isInState(state))

   # rigid rotations
   print("rigid rotations:")
   for rot in range(24):
      cube.reset()
      cube.move(RubikCube.rotateMoveString(RubikCube.edge_cycle_012_str, rot))
      print(rot, cube.getEdgePermutation())
   for rot in range(24):
      cube.reset()
      cube.move(RubikCube.rotateMoveString(RubikCube.corner_cycle_012_str, rot))
      print(rot, cube.getCornerPermutation())

   # (group) order of moves
   print("#order of basic moves:")
   print( [ RubikCube.findOrder( [m] )  for m in range(1, 7) ] )

   print("L F:", RubikCube.findOrder( [1, 3] ))
   seq1_str = "R U2 D' B D'"
   seq2_str = "D' B D' U2 R"
   seq1 = RubikCube.string2moves(seq1_str)
   seq2 = RubikCube.string2moves(seq2_str)
   print(seq1_str + ":", RubikCube.findOrder(seq1))
   print(seq2_str + ":", RubikCube.findOrder(seq2))
  

#TESTsuite(False)
#TESTsuite(True)


######
# END
######

