# test RubikCube class


from rubik import *


  
TESTsuite()
TESTsuite(RubikCube.FORMAT_FACESONLY)

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
#            cnt += 1                                                                                                                                                                                            #         print(i, j, k, p)                                                                                                                                                                                      #print(cnt)#cnt = 0


#END
