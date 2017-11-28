"""
	N coins of values v1, v2, .. vn and n is even. Players take turns to pick coins from the end.
	You start the game. How do you maximize the value of the coins?
	
	ex: 8, 15, 3, 7 --> I should choose, 7 (opponent 8), 15, (opponent 3)
	Thus, greedy approach does not work

	RECURRENCE RELATION:	
	F(i, j) --> represents the max value the user can collect from i'th to j'th coin
	
	F(i, j) = max( vi + min(F(i+2, j), F(i+1, j-1) ), #Opp chose v[i+1] or v[j]
				   vj + min(F(i+1, j-1), F(i, j-2) )  #Opp chose v[i] or v[j-1]
				 )
	 Base cases:
	 F(i, j) = v[i] if i == j
	 F(i, j) = max(v[i], v[j]) if j == i+1
"""

def dp_game_strategy(v):
	# We fill the dp matrix in diagonal manner, as the base case is on equality
	n = cols = rows = len(v)
	table = [[None]*cols for i in range(rows)] # NOTE, see how i am constructing 2d matrix 
	
	for gap in range(0, n):
		i = 0
		for j in range(gap, n):
			'''
				Here, 
				x = F(i+2, j)
				y = F(i+1, j-1)
				z = F(i, j-2)
			'''
			x = table[i+2][j] if i+2 <= j else 0
			y = table[i+1][j-1] if i+1 <= j-1 else 0
			z = table[i][j-2] if i <= j-2 else 0
			
			table[i][j] = max( v[i] + min(x, y), v[j] + min(y, z) )
			i += 1
	print '#### table:'
	for i in table:		
		print i
	return table[0][n-1]

def get_max_game_strategy(v, i, j):
	if i == j:
		return v[i]
	
	if j == i+1:
		return max(v[i], v[j])
	
	return max(
				v[i] + min( get_max_game_strategy(v, i+2, j), get_max_game_strategy(v, i+1, j-1) ),
				v[j] + min( get_max_game_strategy(v, i+1, j-1), get_max_game_strategy(v, i, j-2) )
			  )


if __name__ == '__main__':
	v1 = [8, 15, 3, 7]
	v2 = [5, 3, 7, 10]
	v3 = [20, 30, 2, 2, 2, 10]
	
	assert get_max_game_strategy(v1, 0, len(v1)-1) == 22
	assert get_max_game_strategy(v2, 0, len(v2)-1) == 15
	assert get_max_game_strategy(v3, 0, len(v3)-1) == 42
	
	print dp_game_strategy(v1)
	
	print 'Done!'

