# solve Loo Key problem on angstromCTF 2020
#
# given w, w_a = a^(-1) w a, w_b = b^(-1) w b,  where a and b commute
# must construct key = a^(-1) b^(-1) w b a
#
# => strategy, construct 'a' by bruteforce search given w_a and w


from rubik import *

#
# cursory look at the moves/states involved
#

def checkMoves(s):
   moves = RubikCube.string2moves(s)
   print(s, "->",  moves)
   print("order:", RubikCube.findOrder(moves))


wStr = "U R2 F2 R2 D L2 B2 D' F2 D U2 L' R2 U B U F' L2 F R'"
waStr = "B2 D2 U' R2 D U2 B2 L R2 D' U2 L D L2 B L' U L R'"
wbStr = "L2 U L2 R2 U F2 D2 F2 R2 U R' U' L' B L2 R D B2 F' R'"

checkMoves(wStr)
checkMoves(waStr)
checkMoves(wbStr)


# move sequences (ints)
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

# => w_a and w have the same edge positions and orientations, so 'a' very likely only affects corners



# get corner perms of in a from wa, w
corn_wa = [p[0]   for p in wa_corners]
corn_w  = [p[0]   for p in w_corners]
print(corn_wa, corn_w)


# consider all corner permutations for 'a'
# and check whether w_a = a^(-1) w a, i.e., a * w_a = w * a
#
# first move corners into position, then twist them the right way
#

# find corner permutation
#
print("#viable corner perms for a:")

import itertools


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
      if cnt & 1 == 0:  # only even perms possible on real cube (if edges are unchanged)
         print(p, cnt)


# build corner rotations for a = (0, 3, 1, 6, 4, 2, 5, 7) from 3-cycles
print("build a (corner cycles):")

pgoal = (0, 3, 1, 6, 4, 2, 5, 7)
acycle_str = []
p0 = [0, 1, 2, 3, 4, 5, 6, 7]
for i in range(8):
   if p0[i] == pgoal[i]:
      continue
   idx1, idx2 = p0.index(pgoal[i]), i     # construct a cycle that fills i-th position using tail of p0
   idx3 = i + 1   if idx2 != i + 1   else i + 2
   print(p0, idx1, idx2, idx3)
   acycle_str.append(RubikCube.corner_cycles[idx1][idx2][idx3])
   p0copy = tuple(p0)     # apply to p0
   p0[idx1] = p0copy[idx3]
   p0[idx2] = p0copy[idx1]
   p0[idx3] = p0copy[idx2]

acycle_str = " ".join(acycle_str)
print(acycle_str)     # 256+123 would have been shorter but this works too
cube.reset()
cube.stringMove(acycle_str)
print( cube.toString(RubikCube.FORMAT_KOCIEMBA) )  #UUUUUUDUUBRLRRRBRUFFRFFFDFRLDDDDDDDFLLRLLLLLFFBBBBBRBB
# reduce via Kociemba alg 
acycle_str = RubikCube.invertStringMoves("D2 F1 U1 B2 U3 F1 D1 R2 B2 R2 B2 U3 R2 D1 F2 U1")
print("acycle=", acycle_str)

# verify that corner positions work
cube.reset()
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


# consider all corner twists for 'a'
# and check whether w_a = a^(-1) w a, i.e., a * w_a = w * a
print("#viable corner twists for a:")

def find_atwists(wa_corners):
   wa_twistgoal = [j for (i,j) in wa_corners]
   print("goal:", wa_twistgoal)
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
      atwist_perm = cube.state2permutation()
      # construct a^(-1) * w * a
      cube.reset()
      cube.invPermute(atwist_perm)
      cube.invPermute(acycle_perm)  # apply corner cycle -> may change corner orientations too
      cube.permute(perm_w)
      cube.permute(acycle_perm)
      cube.permute(atwist_perm)
      # check result
      if [j for (i,j) in cube.getCornerPermutation()] == wa_twistgoal:
         print(p, moves)
         return p, moves

_, atwist_str = find_atwists(wa_corners)
print("atwist=", atwist_str)

a_str = acycle_str + atwist_str
print("a=", a_str)

# verify result
cube.reset()
cube.stringMove(a_str, -1)
cube.permute(perm_w)
cube.stringMove(a_str)
cube.invPermute(perm_wa)
print(cube.isReset())

# compactify result for a_str

print(a_str)
cube.reset()
cube.stringMove(a_str)
print("a_state:", cube.toString(RubikCube.FORMAT_KOCIEMBA))
    #UUFUUUDUBRRURRRBRFFFUFFFLFRFDDDDDBDRLLRLLLDLDLBBBBBUBL
    #-> B2 U1 F2 U2 F2 D3 L2 U1 L2 U2 F3 L2 F1 D1 U3 B3 U3 B3
a1_str = RubikCube.invertStringMoves("B2 U1 F2 U2 F2 D3 L2 U1 L2 U2 F3 L2 F1 D1 U3 B3 U3 B3")
print("simplified a=", a1_str, RubikCube.areEqualStringMoves(a_str, a1_str))

a2_str = "B U B U D' F' L2 F U2 L2 U' L2 D F2 U2 F2 U' B2"

# compute secret key a^(-1) w_b a

key_str = RubikCube.invertStringMoves(a2_str) + " " + wbStr + " " + a2_str
print(key_str)

# simplify it
cube.reset()
cube.stringMove(key_str)
print("key_state:", cube.toString(RubikCube.FORMAT_KOCIEMBA))
   #RRDBUUURFRBFLRFLFUBFDBFURRBBBDRDLULBFLRFLDLUDLDUDBULDF
   #-> R1 B2 D2 F1 R2 D1 L3 B3 R1 B1 L2 U3 R2 U3 B2 L2 U3 F2 D1 F2

key1_str = RubikCube.invertStringMoves("R1 B2 D2 F1 R2 D1 L3 B3 R1 B1 L2 U3 R2 U3 B2 L2 U3 F2 D1 F2")
print(key1_str)
print(key1_str.replace(" ",""))

#actf{F2D'F2UL2B2UR2UL2B'R'BLD'R2F'D2B2R'}


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

