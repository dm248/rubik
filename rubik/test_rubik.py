# test RubikCube class


from rubik import *


  
TESTsuite()
TESTsuite(RubikCube.FORMAT_FACESONLY)


# test flips/twists
def test_tables2(tbl, edge = True):
   cnt = 0
   n = len(tbl)
   cube = RubikCube()
   for i in range(n):
      for j in range(n):
         s, p, bad = tbl[i][j], None, False
         bad = (s != None)  if i == j   else  (s == None) 
         if s != None:
            cnt += 1
            cube.reset()
            cube.move(s)
            p = cube.getEdgePermutation()   if edge   else  cube.getCornerPermutation()
            bad = ( p[i][1] != 1 or p[j][1] == 0)
         if bad:
            print((i,j), p, s)
   print("count(2):", cnt)


# test cycles
def test_tables3(tbl, edge = True):
   cnt = 0
   n = len(tbl)
   for i in range(n):
      for j in range(n):
         for k in range(n):
            s, p = tbl[i][j][k], None
            bad = (s != None)  if j == i or k == j or k == i   else  (s == None) 
            if s != None:
               cube = RubikCube()
               cube.move(s)
               p = cube.getEdgePermutation()   if edge   else  cube.getCornerPermutation()
               cnt += 1
               bad = (p[i][0] != k or p[j][0] != i or p[k][0] != j)
            if bad:
               print((i,j,k), p, s)
   print("count(3):", cnt)


def test_tables(tbl, edge = True):
  # determine number of columns
  d, t = 0, tbl
  while isinstance(t, list):
     d, t = d + 1, t[0]
  if d == 2:   # flip/twist
     test_tables2(tbl, edge)
  elif d == 3: # cycle
     test_tables3(tbl, edge)
  

test_tables(RubikCube.edge_cycles, True)
test_tables(RubikCube.edge_flips, True)
test_tables(RubikCube.corner_cycles, False)
test_tables(RubikCube.corner_twists, False)
#END
