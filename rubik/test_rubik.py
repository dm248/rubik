# test RubikCube class


from rubik import *


  
#TESTsuite()
#TESTsuite(RubikCube.FORMAT_FACESONLY)

# 0,10
cube = RubikCube()
cube.stringMove(RubikCube.edge_flips[0][2])
cube.stringMove(RubikCube.edge_flips[2][10])
#cube.stringMove("")
print(RubikCube.invertStringMoves("U1 R1 L1 F3 R2 F3 R1 L3 U1 R2 U2 B2 L2 F2 D1 L2 B2 U3"))
print(cube.toString(RubikCube.FORMAT_KOCIEMBA))
print(cube.getEdgePermutation())


cnt = 0
for i in range(12):
   for j in range(12):
         s = RubikCube.edge_flips[i][j]
         p = None
         if s != None:
            cube = RubikCube()
            cube.stringMove(s)
            p = cube.getEdgePermutation()
            cnt += 1                                                                                                                                                                                            #         print(i, j, k, p)                                                                                                                                                                                      #print(cnt)#cnt = 0
         print((i,j), p, s)

print(cnt)

#END
