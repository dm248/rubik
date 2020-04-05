# 3x3x3 Rubik Cube routines by dm (2020/03/18 - ...)
#
#
# normally would use pypi.org/project/rubik-solver but it seems to be buggy
# (e.g., it gives nonsensical solve if you start from the solved state)
#
# so do it ourselves, in similar notation as in there.
#
#
# YBRGOW face colors: [Y]ellow ->1  [B]lue ->2  [R]ed ->3  [G]reen ->4  [O]range ->5  [W]hite -> 6
#
# Keep:     4 (Upper center) = YELLOW
#          13 (Left center)  = BLUE
#          22 (Front center) = RED
#          31 (Right center) = GREEN
#          40 (Back center)  = ORANGE
#          49 (Down center)  = WHITE
#
#
# state geometry:
#
#               ----------------
#               | 0  | 1  | 2  | 
#               ----------------   UP
#               | 3  | 4  | 5  |
#               ----------------
#               | 6  | 7  | 8  |
#   LEFT        ----------------    RIGHT          BACK
#-------------------------------------------------------------
#| 9  | 10 | 11 | 18 | 19 | 20 | 27 | 28 | 29 | 36 | 37 | 38 |
#-------------------------------------------------------------
#| 12 | 13 | 14 | 21 | 22 | 23 | 30 | 31 | 32 | 39 | 40 | 41 |
#-------------------------------------------------------------
#| 15 | 16 | 17 | 24 | 25 | 26 | 33 | 34 | 35 | 42 | 43 | 44 |
#-------------------------------------------------------------
#               ----------------
#               | 45 | 46 | 47 |
#               ----------------
#               | 48 | 49 | 50 |  DOWN
#               ----------------
#               | 51 | 52 | 53 |
#               ----------------
#
#
# corners:    0-1           edges:  +0+  
#             | |  UP               3 1
#           0-3-2-                +3+2+1+-
#           | | |  FRONT          4 5 6 7  
#           4-7-6-                +b+a+9+-       a = 10, b = 11
#             | |  DOWN             b 9
#             4-5                   +8+
#
#

import sys, numpy as np


class RubikCube():

   n = 54

   ###
   # face/color coding
   ###

   color2int_map_FACES = { 'Y': 1, 'B': 2, 'R': 3, 'G': 4, 'O': 5, 'W': 6 }

   color2int_map  = { 'a':  1, 'b':  2, 'c':  3, 'd':  4, 'e':  5,
                      'f':  6, 'g':  7, 'h':  8, 'i':  9, 'j': 10,
                      'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15,
                      'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20,
                      'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25,
                      'z': 26, '0': 27, '1': 28, '2': 29, '3': 30,
                      '4': 31, '5': 32, '6': 33, '7': 34, '8': 35,
                      '9': 36, 'A': 37, 'B': 38, 'C': 39, 'D': 40,
                      'E': 41, 'F': 42, 'G': 43, 'H': 44, 'I': 45,
                      'J': 46, 'K': 47, 'L': 48, 'M': 49, 'N': 50,
                      'O': 51, 'P': 52, 'Q': 53, 'R': 54
                    }

   int2color_map_FACES = {  1: 'Y',  2: 'Y',  3: 'Y',  4: 'Y',  5: 'Y',
                            6: 'Y',  7: 'Y',  8: 'Y',  9: 'Y', 10: 'B',
                           11: 'B', 12: 'B', 13: 'B', 14: 'B', 15: 'B',
                           16: 'B', 17: 'B', 18: 'B', 19: 'R', 20: 'R',
                           21: 'R', 22: 'R', 23: 'R', 24: 'R', 25: 'R',
                           26: 'R', 27: 'R', 28: 'G', 29: 'G', 30: 'G',
                           31: 'G', 32: 'G', 33: 'G', 34: 'G', 35: 'G',
                           36: 'G', 37: 'O', 38: 'O', 39: 'O', 40: 'O',
                           41: 'O', 42: 'O', 43: 'O', 44: 'O', 45: 'O',
                           46: 'W', 47: 'W', 48: 'W', 49: 'W', 50: 'W',
                           51: 'W', 52: 'W', 53: 'W', 54: 'W'
                        }
   
   int2color_map = {  1: 'a',  2: 'b',  3: 'c',  4: 'd',  5: 'e',
                      6: 'f',  7: 'g',  8: 'h',  9: 'i', 10: 'j',
                     11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o',
                     16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
                     21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y',
                     26: 'z', 27: '0', 28: '1', 29: '2', 30: '3',
                     31: '4', 32: '5', 33: '6', 34: '7', 35: '8',
                     36: '9', 37: 'A', 38: 'B', 39: 'C', 40: 'D',
                     41: 'E', 42: 'F', 43: 'G', 44: 'H', 45: 'I',
                     46: 'J', 47: 'K', 48: 'L', 49: 'M', 50: 'N',
                     51: 'O', 52: 'P', 53: 'Q', 54: 'R'
                  }

   # color name to integer value
   def color2int(c, c2imap = color2int_map):
      return c2imap.get(c, 0)

   # color value to name
   def int2color(v, i2cmap = int2color_map):
      return i2cmap.get(v, "*")

   ###
   # centers, corners and edges
   ###

   def idx2face(i):
      return (i // 9) + 1

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
   for i in center_indices:   color2center_map[idx2face(i)] = i

   color2edge_map = {}
   for (i,j) in edge_indices:
      fi,fj = idx2face(i), idx2face(j)
      color2edge_map[(fi,fj)] = (i,j)
      color2edge_map[(fj,fi)] = (j,i)

   # FIXME: can we avoid spelling out each permutation?
   color2corner_map = {}
   for (i,j,k) in corner_indices:
      fi,fj,fk = idx2face(i), idx2face(j), idx2face(k)
      color2corner_map[(fi,fj,fk)] = (i,j,k)
      color2corner_map[(fi,fk,fj)] = (i,k,j)
      color2corner_map[(fj,fi,fk)] = (j,i,k)
      color2corner_map[(fj,fk,fi)] = (j,k,i)
      color2corner_map[(fk,fi,fj)] = (k,i,j)
      color2corner_map[(fk,fj,fi)] = (k,j,i)

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
      for i in range(self.n):
         self.state[i] = i + 1

   # state related

   # whether cube is in solved state - FIXME: is it faster to use isInState() ??
   def isReset(self):
      for i in range(self.n):
         if self.state[i] != i + 1:
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

   def toString(self, facesonly = False):
      i2cmap = RubikCube.int2color_map  if not facesonly  else RubikCube.int2color_map_FACES
      return "".join( [ RubikCube.int2color(v, i2cmap) for v in self.state ]  )

   def __range2string(self, start, end, facesonly = False):
      i2cmap = RubikCube.int2color_map  if not facesonly  else RubikCube.int2color_map_FACES
      return "".join( [ RubikCube.int2color(v, i2cmap) for v in self.state[start:end] ] )

   def print(self, facesonly = False):
      for row in range(3):   # top face
         start = row * 3
         print("   |" + self.__range2string(start, start + 3, facesonly) + "|")
      print("    ---")
      for row in range(3): # left, mid, right, back
         s = ""
         for face in range(1, 5): 
            start = face * 9 + row * 3
            s += self.__range2string(start, start + 3, facesonly)
            if face != 4:
               s += "|"
         print(s)
      print("    ---")
      for row in range(3):   # bottom face
         start = 5 * 9 + row * 3
         print("   |" + self.__range2string(start, start + 3, facesonly) + "|")

   #
   # basic moves (L R F B U D)
   #

   # cycle positions given by indices
   def __cycleCells(self, shft, indices):
      state = self.state
      tmp = tuple( state[idx] for idx in indices )
      ncycle = len(indices)
      for i in range(ncycle):
         state[indices[(i + shft) % ncycle]] = tmp[i]
            

   # rotate one face clockwise:   012     630
   #                              345 ->  741
   #                              678     852
   def __rotateFace(self, face, dir = 1):
      pos = 9 * face   # top left corner of face
      self.__cycleCells(2 * dir, (pos, pos + 1, pos + 2, pos + 5, pos + 8, pos + 7, pos + 6, pos + 3) )
         
    
   # left
   
   def moveL(self, dir = 1): # clockwise turn of LEFT face (around #13)
      self.__rotateFace(1, dir)   # face
      # neighbors: 0, 3, 6|18,21,24|45,48,51|44,41,38| -> rotate by 3 for CW
      self.__cycleCells(3 * dir, (0, 3, 6, 18, 21, 24, 45, 48, 51, 44, 41, 38) )

   def moveLinv(self):   # L'
       self.moveL(-1)

   def moveL2(self):
       self.moveL(dir = 2)

   # right
 
   def moveR(self, dir = 1): #clockwise turn of RIGHT face (around #31)
      self.__rotateFace(3, dir)   # face
      # neighbors: 53,50,47|26,23,20| 8, 5, 2|36,39,42| -> rotate by 3 for CW
      self.__cycleCells(3 * dir, (53, 50, 47, 26, 23, 20, 8, 5, 2, 36, 39, 42) )

   def moveRinv(self):       # R'
       self.moveR(dir = -1)

   def moveR2(self):
       self.moveR(dir = 2)

   # front
 
   def moveF(self, dir = 1): #clockwise turn of FRONT face (around #22)
      self.__rotateFace(2, dir)   # face
      # neighbors:  6, 7, 8|27,30,33|47,46,45|17,14,11| -> rotate by 3 for CW
      self.__cycleCells(3 * dir, (6, 7, 8, 27, 30, 33, 47, 46, 45, 17, 14, 11) )

   def moveFinv(self):      # F'
       self.moveF(dir = -1)

   def moveF2(self):
       self.moveF(dir = 2)

   # back
 
   def moveB(self, dir = 1): #clockwise turn of BACK face (around #40)
      self.__rotateFace(4, dir)   # face
      # neighbors: 35,32,29| 2, 1, 0| 9,12,15|51,52,53| -> rotate by 3 for CW
      self.__cycleCells(3 * dir, (35, 32, 29, 2, 1, 0, 9, 12, 15, 51, 52, 53) )

   def moveBinv(self):       # B'
       self.moveB(dir = -1)

   def moveB2(self):
       self.moveB(dir = 2)

   # up
 
   def moveU(self, dir = 1): #clockwise turn of UP face (around #4)
      self.__rotateFace(0, dir)   # face
      # neighbors: 11,10, 9|38,37,36|29,28,27|20,19,18| -> rotate by 3 for CW
      self.__cycleCells(3 * dir, (11, 10, 9, 38, 37, 36, 29, 28, 27, 20, 19, 18) )

   def moveUinv(self):        # U'
       self.moveU(dir = -1)

   def moveU2(self):
       self.moveU(dir = 2)

   # down
 
   def moveD(self, dir = 1): #clockwise turn of DOWN face (around #49)
      # face
      self.__rotateFace(5, dir)
      # neighbors: 15,16,17|24,25,26|33,34,35|42,43,44 -> rotate by 3 for CW
      self.__cycleCells(3 * dir, (15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44) )

   def moveDinv(self):        # D'
       self.moveD(dir = -1)

   def moveD2(self):
       self.moveD(dir = 2)    # D + D

   #
   # sequence of basic moves (L->1 R->2 F->3 B->4 U->5 D->6)
   #
   # NOTE: we apply moves in left-to-right order
   #       whereas in operator notation, things go from right to left

   int2move = { 1: moveL, 2: moveR, 3: moveF, 4: moveB, 5: moveU, 6: moveD,    # basic moves
               -1: moveLinv, -2: moveRinv, -3: moveFinv,        # inverses 
               -4: moveBinv, -5: moveUinv, -6: moveDinv,
               11: moveL2, 12: moveR2, 13: moveF2, 14: moveB2, 15: moveU2, 16: moveD2  # double moves
              }

   def move(self, sequence, count = 1):   # apply sequence of moves
      if count < 0: # interpret negative counts as inverse 
         sequence = RubikCube.invertMoves(sequence)
         count = -count
      for _ in range(count):
         for code in sequence:
            if code == 0:
               continue
            m = self.int2move[code]   # fetch move
            m(self)                   # apply it

   def stringMove(self, moves: str, count = 1):
      sequence = RubikCube.string2moves(moves)
      self.move(sequence, count)
 
   def invertMoves(sequence):   # invert move sequence
      return [ c if c > 10  else -c   for c in sequence[::-1] ]

   def invMove(self, sequence):  # apply inverse of moves
      self.move(RubikCube.invertMoves(sequence))

   # whether two move sequences are equivalent
   def areEqualMoves(seq1, seq2):
      cube = RubikCube()
      cube.move(seq1)
      cube.invMove(seq2)
      return cube.isReset()

   # parsing sequences
   
   str2move = { "L":   1,  "R":   2,  "F":   3,  "B":   4,  "U":   5,  "D":   6,
                "L'": -1,  "R'": -2,  "F'": -3,  "B'": -4,  "U'": -5,  "D'": -6,
                "L2": 11,  "R2": 12,  "F2": 13,  "B2": 14,  "U2": 15,  "D2": 16
              }

   def string2moves(s):
       return [ RubikCube.str2move[w.strip()] for w in s.split(" ") ]
       
   def __areCentersBad(tpl):
      return tpl[4] != 5 or tpl[13] != 14 or tpl[22] != 23 or tpl[31] != 32 or tpl[40] != 41 or tpl[49] != 50 

   def string2state(s): # FIXME: does not check rotation parity
      n = RubikCube.n
      if len(s) != n:
         raise("Incorrect string length " + str(len(s)))
      # only faces YBRGOW
      if s.count("Y") == 9:
         lst = [RubikCube.color2int_map_FACES[v] for v in s]
         # centers
         for i in RubikCube.center_indices:
            lst[i] = RubikCube.color2center_map[lst[i]] + 1
         # edges
         for (i, j) in RubikCube.edge_indices:
            (vi, vj) = RubikCube.color2edge_map[(lst[i], lst[j])]
            lst[i], lst[j] = vi + 1, vj + 1
         # corners
         for (i, j, k) in RubikCube.corner_indices:
            (vi, vj, vk) = RubikCube.color2corner_map[(lst[i], lst[j], lst[k])]
            lst[i], lst[j], lst[k] = vi + 1, vj + 1, vk + 1
      # 27 cells a-z,0-9,A-R
      else:  
         lst = [-1] * n
         for i in range(n):
            if lst[i] != -1:
               raise("Duplicate symbol " + str(len(s)))
            lst[i] = RubikCube.color2int_map[s[i]]
      if RubikCube.__areCentersBad(lst):
         raise("Centers wrong")
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

   # give permutation that reaches current state from solved cube
   def state2permutation(self):
      return tuple( [ (j - 1) for j in self.getState() ] )

   # convert a sequence of moves to a permutation
   def moves2permutation(sequence):
      cube = RubikCube()
      cube.move(sequence)
      return cube.state2permutation()

   # convert a sequence of moves to a state (starting from the solved cube)
   def moves2state(sequence):
      cube = RubikCube()
      cube.move(sequence)
      return cube.getState()

   # read off edge permutations
   def getEdgePermutation(self):
      state = self.state
      res = []
      for (i,j) in RubikCube.edge_indices:
         pattern = (state[i] - 1, state[j] - 1)
         res.append(RubikCube.edge_permutation_map[pattern])
      return res

   # read off corner permutations
   def getCornerPermutation(self):
      state = self.state
      res = []
      for (i,j,k) in RubikCube.corner_indices:
         pattern = (state[i] - 1, state[j] - 1, state[k] - 1)
         res.append(RubikCube.corner_permutation_map[pattern])
      return res

   #
   # group stuff
   #

   # find order of a sequence - apply it until we get back original cube
   def findOrder(sequence):
      cube = RubikCube()
      # convert move to permutation
      cube.reset()                
      cube.move(sequence)
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

   rigid_rotation_move_str_map = [None]*24
   for rot in range(24):
      rigid_rotation_move_str_map[rot] = {}
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
   
   # cycle edges 0,1,2->1,2,0 (edges on same face)
   edge_cycle_012_str = "R2 U' B2 R2 B2 U B' F' U2 B F U2 R2"
   # flip orientations of edges 0 and 1 (neighbors on same face)
   edge_flip_01_str = "R U F R2 F' R' U F2 R2 U2 F2 U2 R2 F U2 F"
   # cycle corners 0,1,2->1,2,0 (corners on same face)
   corner_cycle_012_str = "R' L' D2 R U R' D2 R U' L"
   # twist corners 0,1 (on same edge), 0,2 (across same face), 0,6 (across cube)
   corner_twist_01_str = "D R D L' D' R' D R' F2 R B2 R' F2 R B2 L D2"
   corner_twist_02_str = "U2 F D B2 D' F' D2 B2 D' R2 D' R2 U F2 U F2" 



#
# TESTS
# 


def TESTsuite(facesonly = False):
   cube = RubikCube()
   print(cube.toString(facesonly))
     
   # init
   print("#initial:")
   cube.print(facesonly)

   # basic moves
   lst = [("L", cube.moveL), ("R", cube.moveR), ("F", cube.moveF), 
          ("B", cube.moveB), ("U", cube.moveU), ("D", cube.moveD)]
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
   s = cube.toString(True)
   print(s)
   state = RubikCube.string2state(s)
   print(cube.isInState(state))
   s = cube.toString()
   print(s)
   state = RubikCube.string2state(s)
   print(cube.isInState(state))

   # rigid rotations
   print("rigid rotations:")
   for rot in range(24):
      cube.reset()
      cube.stringMove(RubikCube.rotateMoveString(RubikCube.edge_cycle_012_str, rot))
      print(rot, cube.getEdgePermutation())
   for rot in range(24):
      cube.reset()
      cube.stringMove(RubikCube.rotateMoveString(RubikCube.corner_cycle_012_str, rot))
      print(rot, cube.getCornerPermutation())
  
#TESTsuite(False)
#TESTsuite(True)



print("#order of basic moves:")
print( [ RubikCube.findOrder( [m] )  for m in range(1, 7) ] )

print("L F:", RubikCube.findOrder( [1, 3] ))
seq1_str = "R U2 D' B D'"
seq2_str = "D' B D' U2 R"
seq1 = RubikCube.string2moves(seq1_str)
seq2 = RubikCube.string2moves(seq2_str)
print(RubikCube.findOrder(seq1), seq1_str)
print(RubikCube.findOrder(seq2), seq2_str)

# colormap at https://rubiks-state-solver.com/
#
# RubikCube.int2color_map = { 1: "W", 2: "O", 3: "G", 4: "R", 5: "B", 6: "Y" }


def checkMoves(s):
   moves = RubikCube.string2moves(s)
   print(s)
   print(moves)
   print(RubikCube.findOrder(moves))


wStr = "U R2 F2 R2 D L2 B2 D' F2 D U2 L' R2 U B U F' L2 F R'"
waStr = "B2 D2 U' R2 D U2 B2 L R2 D' U2 L D L2 B L' U L R'"
wbStr = "L2 U L2 R2 U F2 D2 F2 R2 U R' U' L' B L2 R D B2 F' R'"

#waStr = " ".join( ["R U L B", wStr, "B' L' U' R'"] )

checkMoves(wStr)
checkMoves(waStr)
checkMoves(wbStr)


# brute force a

# moves, ordered such that it is easy to distinguish moves generated by same basic move
# - indices (1,2,3), (4,5,6), ... (16,17,18) are from same basic move
# 
move_set = (0, 1, -1, 11, 2, -2, 12, 3, -3, 13, 4, -4, 14, 5, -5, 15, 6, -6, 16)
imove_set = tuple(enumerate(move_set))


def search5v1(seq_wa, seq_w):
   cube = RubikCube()
   for m1 in move_set:
      m1inv = -m1   if m1 < 10  else m1
      for m2 in move_set:
         sys.stdout.write('.')
         sys.stdout.flush()
         m2inv = -m2   if m2 < 10  else m2
         for m3 in move_set:
            m3inv = -m3   if m3 < 10  else m3
            for m4 in move_set:
               m4inv = -m4   if m4 < 10  else m4
               for m5 in move_set:
                  m5inv = -m5   if m5 < 10  else m5
                  seq = (m1,m2,m3,m4,m5) + seq_w + (m5inv,m4inv,m3inv,m2inv,m1inv)
                  if RubikCube.areEqualMoves(seq, seq_wa):
                     print(m1, m2, m3, m4, m5)



def search5v2(state_wa, seq_w):
   state_wa = RubikCube.moves2state(seq_wa)
   cube = RubikCube()
   for m1 in move_set:
      m1inv = -m1   if m1 < 10  else m1
      for m2 in move_set:
         sys.stdout.write('.')
         sys.stdout.flush()
         m2inv = -m2   if m2 < 10  else m2
         for m3 in move_set:
            m3inv = -m3   if m3 < 10  else m3
            for m4 in move_set:
               m4inv = -m4   if m4 < 10  else m4
               for m5 in move_set:
                  m5inv = -m5   if m5 < 10  else m5
                  seq = (m1,m2,m3,m4,m5) + seq_w + (m5inv,m4inv,m3inv,m2inv,m1inv)
                  cube.reset()
                  cube.move(seq)
                  if cube.isInState(state_wa):
                     print(m1, m2, m3, m4, m5)
              

def search5v3(seq_wa, seq_w):
   state_wa = RubikCube.moves2state(seq_wa)
   perm_w = RubikCube.moves2permutation(seq_w)
   cube = RubikCube()
   for m1 in move_set:
      m1inv = -m1   if m1 < 10  else m1
      for m2 in move_set:
         sys.stdout.write('.')
         sys.stdout.flush()
         m2inv = -m2   if m2 < 10  else m2
         for m3 in move_set:
            m3inv = -m3   if m3 < 10  else m3
            for m4 in move_set:
               m4inv = -m4   if m4 < 10  else m4
               for m5 in move_set:
                  m5inv = -m5   if m5 < 10  else m5
                  cube.reset()
                  cube.move( (m1,m2,m3,m4,m5) )
                  cube.permute(perm_w)
                  cube.move( (m5inv,m4inv,m3inv,m2inv,m1inv) )
                  if cube.isInState(state_wa):
                     print(m1, m2, m3, m4, m5) 


def skipCheck(i1, i2):
   return i1 != 0 and ((i1 - 1) // 3 == (i2 - 1) // 3 or i2 == 0)


def search5v3b(seq_wa, seq_w):
   state_wa = RubikCube.moves2state(seq_wa)
   perm_w = RubikCube.moves2permutation(seq_w)
   cube = RubikCube()
   #cnt = 0
   for (i1,m1) in imove_set:
      m1inv = -m1   if m1 < 10  else m1
      for (i2,m2) in imove_set:
         if skipCheck(i1, i2):  continue
         sys.stdout.write('.')
         sys.stdout.flush()
         m2inv = -m2   if m2 < 10  else m2
         for (i3,m3) in imove_set:
            if skipCheck(i2, i3):  continue
            m3inv = -m3   if m3 < 10  else m3
            for (i4,m4) in imove_set:
               if skipCheck(i3, i4):  continue
               m4inv = -m4   if m4 < 10  else m4
               for (i5,m5) in imove_set:
                  if skipCheck(i4, i5):  continue
                  m5inv = -m5   if m5 < 10  else m5
                  #cnt += 1
                  cube.reset()
                  cube.move( (m1,m2,m3,m4,m5) )
                  cube.permute(perm_w)
                  cube.move( (m5inv,m4inv,m3inv,m2inv,m1inv) )
                  if cube.isInState(state_wa):
                     print(m1, m2, m3, m4, m5) 
   #print(cnt, 18 * (pow(15, 4) + pow(15, 3) + pow(15, 2) + pow(15, 1) + 1) + 1)


def search5v4(seq_wa, seq_w):
   state_wa = RubikCube.moves2state(seq_wa)
   perm_w = RubikCube.moves2permutation(seq_w)
   cube = RubikCube()
   #cnt = 0
   for (i1,m1) in imove_set:
      for (i2,m2) in imove_set:
         if skipCheck(i1, i2):  continue
         sys.stdout.write('.')
         sys.stdout.flush()
         for (i3,m3) in imove_set:
            if skipCheck(i2, i3):  continue
            for (i4,m4) in imove_set:
               if skipCheck(i3, i4):  continue
               cube.reset()
               cube.move( (m1,m2,m3,m4) )
               state4 = cube.getState() 
               for (i5,m5) in imove_set:
                  if skipCheck(i4, i5):  continue
                  #cnt += 1
                  cube.setState(state4)
                  cube.move( (m5,) )
                  perm = cube.state2permutation()
                  cube.permute(perm_w)
                  cube.invPermute(perm)
                  if cube.isInState(state_wa):
                     print(m1, m2, m3, m4, m5) 
   #print(cnt, 18 * (pow(15, 4) + pow(15, 3) + pow(15, 2) + pow(15, 1) + 1) + 1)


def search5v4b(seq_wa, seq_w):
   state_wa = RubikCube.moves2state(seq_wa)
   perm_w = RubikCube.moves2permutation(seq_w)
   cube = RubikCube()
   state0 = cube.getState()
   cnt = 0
   for (i1,m1) in imove_set:
      cube.setState(state0)
      cube.move( (m1,) )
      state1 = cube.getState()
      for (i2,m2) in imove_set:
         if skipCheck(i1, i2):  continue
         sys.stdout.write('.')
         sys.stdout.flush()
         cube.setState(state1)
         cube.move( (m2,) )
         state2 = cube.getState()
         for (i3,m3) in imove_set:
            if skipCheck(i2, i3):  continue
            cube.setState(state2)
            cube.move( (m3,) )
            state3 = cube.getState()
            for (i4,m4) in imove_set:
               if skipCheck(i3, i4):  continue
               cube.setState(state3)
               cube.move( (m4,) )
               state4 = cube.getState() 
               for (i5,m5) in imove_set:
                  if skipCheck(i4, i5):  continue
                  cnt += 1
                  cube.setState(state4)
                  cube.move( (m5,) )
                  perm = cube.state2permutation()
                  cube.permute(perm_w)
                  cube.invPermute(perm)
                  if cube.isInState(state_wa):
                     print(m1, m2, m3, m4, m5) 
   print(cnt, 18 * (pow(15, 4) + pow(15, 3) + pow(15, 2) + pow(15, 1) + 1) + 1)


def search5v5(seq_wa, seq_w, Nmax, progIndLvl):
   state_wa = RubikCube.moves2state(seq_wa)
   perm_w = RubikCube.moves2permutation(seq_w)
   #
   cube = RubikCube()
   state0 = cube.getState()
   states = [state0] * Nmax
   indices = [0] * (Nmax + 1)
   idxMax = len(move_set)
   cnt = 0
   while True:
      # increment, track state too
      i = 0
      while i < Nmax:
         if i == progIndLvl:
            sys.stdout.write('.')
            sys.stdout.flush()
         idx = indices[i] + 1
         if idx == idxMax:     # standard increment
            indices[i] = 0      # set 0 for now, will be updated at end
            i += 1
            continue
         indices[i] = idx
         if skipCheck(indices[i + 1], idx):  # skip consecutive repeated moves
            continue
         break   # exit loop with i = last updated counter
      if i == Nmax:  # termination check
         break
      cube.setState(states[i])             # load saved state at lvl i
      cube.move( (move_set[indices[i]],) ) # make i-th level move
      for j in range(i - 1, -1, -1):  # take first valid moves at levels j < i, update states
         while skipCheck(indices[j + 1], indices[j]):
            indices[j] += 1
         states[j] = cube.getState()    # store starting state
         cube.move( (move_set[indices[j]],) )  # make level j move
      cnt += 1
      #print(cnt, indices, cube.getState(), move_set)
      perm = cube.state2permutation()
      cube.permute(perm_w)
      cube.invPermute(perm)
      if cube.isInState(state_wa):
         print( tuple( [move_set[idx] for idx in indices] ) ) 
   print(cnt, 18 * (pow(15, Nmax) - 1) // (15 - 1) + 1 )




seq_wa = tuple(RubikCube.string2moves(waStr))
seq_wb = tuple(RubikCube.string2moves(wbStr))
seq_w = tuple(RubikCube.string2moves(wStr))

cube = RubikCube()
print("#w_a:")
cube.move(seq_wa)
state_wa = cube.getState()
cube.print()
print("#w:")
cube.reset()
cube.move(seq_w)
state_w = cube.getState()
cube.print()
print("#w_b:")
cube.reset()
cube.move(seq_wb)
state_wb = cube.getState()
cube.print()


def edgeAndCornerPerms(state):
   cube = RubikCube()
   cube.setState(state)
   edges = cube.getEdgePermutation()
   corners = cube.getCornerPermutation()
   print(edges)
   print(corners)
   return edges, corners  

# compare edge and corner permutations
print("#w_a - edge and corner perms:")
wa_edges, wa_corners = edgeAndCornerPerms(state_wa)
print("#w - edge and corner perms:")
w_edges, w_corners = edgeAndCornerPerms(state_w)
print("#w_b - edge and corner perms:")
wb_edges, wb_corners = edgeAndCornerPerms(state_wb)

# get corner perms of in a from wa, w
corn_wa = [p[0]   for p in wa_corners]
corn_w  = [p[0]   for p in w_corners]
print(corn_wa, corn_w)

import itertools

# consider all corner permutations for 'a'
# and check whether w_a = a^(-1) w a, i.e., a * w_a = w * a
print("#viable corner perms for a:")
for p in itertools.permutations(range(8)):
   # p[i] is the new corner at 'i'  (so it moves from p[i] -> i)
   # construct w * a
   st_w_a = [corn_w[p[i]] for i in range(8)]
   # construct a^-1 on 0...7
   st_ainv = [p[i] for i in range(8)]
   st_ainv_wa = [st_ainv[corn_wa[i]] for i in range(8)]
   if st_ainv_wa == st_w_a:
      # count swaps
      p2 = list(p)
      cnt = 0
      for i in range(8):
         if p2[i] != i:
            pos = p2.index(i)
            p2[i], p2[pos] = p2[pos], p2[i]
            cnt += 1
      if cnt & 1 == 0: # only keep even perms
         print(p, cnt)

# [A,B] = A B A' B' 
#
# from "Group Theory and the Rubik's Cube" by Janet Chen
# http://people.math.harvard.edu/~jjchen/docs/Group%20Theory%20and%20the%20Rubik's%20Cube.pdfhttp://people.math.harvard.edu/~jjchen/docs/Group%20Theory%20and%20the%20Rubik's%20Cube.pdf 
#
# "L R' U2 L' R B2"  edge cycle 0,2,8->8,0,2
#                    (1 edge + 2 parallel edges on its two faces - not useful as building block)
#
# "L R' F L R' D L R' B L R' U L R' F' L R' D' L R' B' L R' U'"  flip orientations of edges 1,3 
#                                      (opposite on same face - not useful as building block)
#
# ("D R D' R' F")^3  corner swap + some edge transformations  (not useful as building block)
#
# Parity of (unoriented) edge permutations and (unoriented) corner permutations must match.
# => if precisely two edges are swapped (->odd), then at least two corners must be swapped as well.
# Lemma 11.14 -> any two edge cubies can be flipped (orientations changed), with no other change to the cube.
#

def analyze(str, count = 1):
   print(str, "*", count)
   cube.reset()
   cube.stringMove(str, count)
   print(cube.getEdgePermutation())
   print(cube.getCornerPermutation())

def analyze2(str, count = 1):
   #print(str, "*", count)
   cube.reset()
   cube.stringMove(str, count)
   #print(cube.getEdgePermutation())
   return cube.getCornerPermutation()

# construct all possible 8 * 7 / 2 = 28 two-corner twists
# neighbor twists -> 8*3/2 = 12
# same-face diag twists -> 8*3/2 = 12
# opposite corner twists -> 8*1/2 = 4
corner_twists = [None]*8
for i in range(8):
   corner_twists[i] = [None]*8
for rot in range(24):
   moveset = (RubikCube.corner_twist_01_str, RubikCube.corner_twist_02_str)
   for s in moveset:
      cube = RubikCube()
      s = RubikCube.rotateMoveString(s, rot)	
      cube.stringMove(s)
      p = cube.getCornerPermutation()
      lst = [ i   for i in range(8)   if p[i][1] != 0 ]
      i,j = lst[0],lst[1]
      if corner_twists[i][j] == None:
         corner_twists[i][j] = s
         corner_twists[j][i] = s

for i in range(8):
   for j in range(8):
      s = corner_twists[i][j]
      p = None
      if s != None:
         cube = RubikCube()
         cube.stringMove(s)
         p = cube.getCornerPermutation()
      print(i, j, p)


s1 = corner_twists[0][2]
s2 = corner_twists[2][6]

cube = RubikCube()
cube.stringMove(s1)
cube.stringMove(s2, -1)
print(cube.getCornerPermutation())
print(s1 + " " + s2)
s = "U2 F D B2 D' F' D2 B2 D' R2 D' R2 U F2 U F2"
cube.reset()
cube.stringMove(s)
print(cube.getCornerPermutation())


exit(0)

#search5v1(seq_wa, seq_w)
#search5v2(seq_wa, seq_w)
#search5v3(seq_wa, seq_w)
#search5v3b(seq_wa, seq_w)
#search5v4(seq_wa, seq_w)
#search5v4b(seq_wa, seq_w)
#search5v5(seq_wa, seq_w, 5, 4)    # 5-move bruteforce for a
search5v5(seq_wa, seq_w, 7, 5)     # 7-move bruteforce for a

