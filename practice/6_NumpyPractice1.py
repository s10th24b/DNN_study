import numpy as np

b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
#array([ 1, 2,  3,  4],
#        5, 6,  7,  8],
#        9, 10, 11, 12] ] )

print(b[:1])
# all row, index 1 column
print(b[0])
# all row, index 1 column
print(b[-1])
# the last row
print(b[-1,:])
# same with above
print(b[-1, ...]) # :(colon) instances can be replaced with ...(dots)
print(b[-1:, ...]) # :(colon) instances can be replaced with ...(dots)
# same with above
print(b[0:2,:])
# 0~1 row


