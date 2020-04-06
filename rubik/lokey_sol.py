# solve Loo Key problem on angstromCTF 2020
#
# given w, w_a = a^(-1) w a, w_b = b^(-1) w b,  where a and b commute
# must construct key = a^(-1) b^(-1) w b a
#
# => strategy, construct 'a' by bruteforce search given w_a and W


from rubik import *


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



seq_wa = tuple(RubikCube.string2moves(waStr))
seq_wb = tuple(RubikCube.string2moves(wbStr))
seq_w = tuple(RubikCube.string2moves(wStr))

cube = RubikCube()
print("#w_a:")
cube.move(seq_wa)
state_wa = cube.getState()
perm_wa = cube.state2permutation()
cube.print()
print("#w:")
cube.reset()
cube.move(seq_w)
state_w = cube.getState()
perm_w = cube.state2permutation()
cube.print()
print("#w_b:")
cube.reset()
cube.move(seq_wb)
state_wb = cube.getState()
perm_wb = cube.state2permutation()
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
#
# first move corners into position, then twist them the right way
#

# find corner permutation
#
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
      if cnt & 1 == 0:  # only even perms possible on real cube
         print(p, cnt)


#cnt = 0
#for i in range(8):
#   for j in range(8):
#      if j == i: continue
#      for k in range(8):
#         if k == i or k == j: continue
#         s = corner_cycles[i][j][k]
#         p = None
#         if s != None:
#            cube = RubikCube()
#            cube.stringMove(s)
#            p = cube.getCornerPermutation()
#            cnt += 1
#         print(i, j, k, p)
#print(cnt)


# build corner rotations for a = (0, 3, 1, 6, 4, 2, 5, 7)  - pick cycles by hand
print("build a (corner cycles):")
cube.reset()
s1 = RubikCube.invertStringMoves(corner_cycles[2][5][6])
s2 = RubikCube.invertStringMoves(corner_cycles[1][2][3])
acycle_str = s1 + " " + s2
print(acycle_str)
acycle_perm = RubikCube.stringMoves2permutation(acycle_str)
cube.permute(acycle_perm)
print(cube.getCornerPermutation())
# check a^-1 w a
print(wa_corners)
cube.reset()
cube.invPermute(acycle_perm)
cube.move(seq_w)
cube.permute(acycle_perm)
print(cube.getCornerPermutation())
# it works

# simplify a_str, remove corner twists
print("simplfy a_str:")
a_str = RubikCube.invertStringMoves("B2 R' D L2 D' R B2 U2 R2 B2 D' F2 D B2 R2 U2")
print(a_str)
cube.reset()
a_perm = RubikCube.stringMoves2permutation(a_str)
cube.permute(a_perm)
print("a_str corners:", cube.getCornerPermutation())
cube.reset()
cube.invPermute(a_perm)
cube.move(seq_w)
cube.permute(a_perm)
print(cube.getCornerPermutation())
print(wa_corners)



# consider all corner twists for 'a'
# and check whether w_a = a^(-1) w a, i.e., a * w_a = w * a
print("#viable corner position perms for a:")

def find_aflips(wa_corners):
   wa_flipgoal = [j for (i,j) in wa_corners]
   print("goal:", wa_flipgoal)
   for p in itertools.product(range(3), repeat = 8):
      # check parity
      if sum(p) % 3 != 0:
         continue
      #print(p)
      # build moves that create given parity
      cube = RubikCube()
      p0 = [0]*8
      moves = ""
      for i in range(7):
         if p[i] == p0[i]:  continue
         m = RubikCube.corner_twists[i][i+1]
         cube2 = RubikCube()
         cube2.stringMove(m)
         corners = cube2.getCornerPermutation()
         #print(i, corners, p0)
         if p[i] == (corners[i][1] + p0[i]) % 3:   # if the move works
            p0[i + 1] = (p0[i + 1] + corners[i + 1][1]) % 3
            cube.stringMove(m)
            moves += " " + m
         else:                     # otherwise use inverse
            p0[i + 1] = (p0[i + 1] - corners[i + 1][1]) % 3
            cube.stringMove(m, -1)
            moves += " " + RubikCube.invertStringMoves(m)
         p0[i] = p[i]
      #print(p, p0)
      aflip_perm = cube.state2permutation()
      # construct a^(-1) * w * a
      cube.reset()
      cube.invPermute(aflip_perm)
      cube.invPermute(acycle_perm)
      cube.permute(perm_w)
      cube.permute(acycle_perm)
      cube.permute(aflip_perm)
      # check result
      if [j for (i,j) in cube.getCornerPermutation()] == wa_flipgoal:
         print(p, moves)
         return p, moves

_, aflip_str = find_aflips(wa_corners)
print(aflip_str)

a_str = acycle_str + aflip_str
print("a=", a_str)

# verify result
cube.reset()
cube.stringMove(a_str, -1)
cube.permute(perm_w)
cube.stringMove(a_str)
cube.invPermute(perm_wa)
print(cube.isReset())

# compactify result for a_str
a1_str = ("B' R F' L2 F R' F' L2 B F B' U F' D2 F U' F' D2 B F D2 B' R2 F' L2 F R2 F' L2 F D' F D B D'",
          "F' D' B2 L' U2 R' D2 R U2 R' D2 R B' R B L B' R' B' L2 B R F2 R' B' R2 F2 R' D2 R' D2 L B2",
          "L B2 F R F L' F' R' F R' U2 R D2 R' U2 R D2 L F2 U R U L' U' R' U R' B2 R F2 R' B2 R F2 L U2"
         )

print(" ".join(a1_str) == a_str)

a2_str = (RubikCube.invertStringMoves("U D' F2 U' D F L2 B2 R2 B2 D' R2 D L2 D' R2"),
          RubikCube.invertStringMoves("R2 L' D2 F2 B' U' R2 L2 F' B2 U2 R2 L2 F2 U' R2 L2"),
          RubikCube.invertStringMoves("R D R' D B2 D' R2 D' L' B2 D R2 L2 U' F2 U L2 D")
         )

a3_str = " ".join(a2_str)
print(a3_str, RubikCube.areEqualStringMoves(a_str, a3_str) )

a4_str = ("R2 D L2 D' R2 D B2 R2 B2 L2 F' D' U F2 D U' L2 R2 U F2 L2 R2 U2 B2 F L2 R2 U B F2 D2 L R2",
          "D' L2 U' F2 U L2 R2 D' B2 L D R2 D B2 D' R D' R'")

a5_str = (RubikCube.invertStringMoves("R2 B2 U2 F2 U' R U D2 R' U2 F2 B2 D' F2 D L2 F2 D R2"),
          "D' L2 U' F2 U L2 R2 D' B2 L D R2 D B2 D' R D' R'")

a6_str = " ".join(a5_str)
print(a6_str, RubikCube.areEqualStringMoves(a_str, a6_str) )

a7_str = ("R2 D' F2 L2 D' F2 D B2 F2 U2 R D2 U' R' U F2 U2 B2 R2 D' L2 U' F2 U L2 R2 D' B2 L D R2",
          "D B2 D' R D' R'")

a8_str = (RubikCube.invertStringMoves("U' F2 U L2 D F2 L' F2 L' D' L2 B2 U2 D R2 F2 D2 F2 U2"),
          "D B2 D' R D' R'")

a9_str = " ".join(a8_str)
print(a9_str, RubikCube.areEqualStringMoves(a_str, a9_str) )

a10_str = RubikCube.invertStringMoves("F' D2 B' D F' B' R2 D' B U' L2 F2 R2 D' B2 U R2 U2 L2")
print(a10_str, RubikCube.areEqualStringMoves(a_str, a10_str) )

a11_str = "L2 U2 R2 U' B2 D R2 F2 L2 U B' D R2 B F D' B D2 F"

# compute secret key a^(-1) w_b a

key_str = RubikCube.invertStringMoves(a11_str) + " " + wbStr + " " + a11_str
print(key_str)

key2_str = ("F' D2 B' D F' B' R2 D' B U' L2 F2 R2 D' B2 U R2 U2 L2 L2 U L2 R2 U F2 D2 F2 R2 U R'",
            "U' L' B L2 R D B2 F' R' L2 U2 R2 U' B2 D R2 F2 L2 U B' D R2 B F D' B D2 F")

print(RubikCube.areEqualStringMoves(key_str, " ".join(key2_str)) )

key3_str = (RubikCube.invertStringMoves("F R' L F R L' F R' F2 B2 R2 D R2 D2 B2 R2 D'"),
            RubikCube.invertStringMoves("R' F' L D2 F' U' B2 R2 L U' F U B2 D L2 U2 D' F2 U2 R2")
           )

key4_str = " ".join(key3_str)
print(key4_str, RubikCube.areEqualStringMoves(key_str, key4_str) )
             
key5_str = RubikCube.invertStringMoves("R B2 D2 F R2 D L' B' R B L2 U' R2 U' B2 L2 U' F2 D F2")

print(key5_str, RubikCube.areEqualStringMoves(key_str, key5_str) )

print(key5_str.replace(" ",""))

exit(0)


## blind search up to some depth using 6 basic moves
## -> ineffective

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


#search5v1(seq_wa, seq_w)
#search5v2(seq_wa, seq_w)
#search5v3(seq_wa, seq_w)
#search5v3b(seq_wa, seq_w)
#search5v4(seq_wa, seq_w)
#search5v4b(seq_wa, seq_w)
#search5v5(seq_wa, seq_w, 5, 4)    # 5-move bruteforce for a
search5v5(seq_wa, seq_w, 7, 5)     # 7-move bruteforce for a

