function score = giniscorenorm(a, p)
	score = giniscore(a, p) / giniscore(a, a);
end