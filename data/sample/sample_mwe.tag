# Sample TAGbank format with MWE (Multi-Word Expression) annotations
# Sentence: Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.

IDX LEX POS HD ELEM RHS LHS
1 pierre nnp 2 beta (S (S (NP-SBJ (NP-SBJ (NP (NP _
2 vinken nnp 9 alpha _ )NP
3 , punct 2 beta _ )NP
4 61 cd 5-6 beta (ADJP (NP _
5-6 years_old _ _ βyears_old _ _
5 years nns 2 beta _ )NP
6 old jj 5 _ _ )ADJP )NP-SBJ
7 , punct 6 beta _ )NP-SBJ
8 will md 9 beta (VP (MD _
9 join vb 0 alpha (VP (VP _
10 the dt 11 alpha (NP (DT _
11 board nn 9 alpha _ )NP
12 as in 9 alpha (PP-CLR _
13 a dt 15 alpha (NP _
14 nonexecutive jj 15 beta (NP _
15 director nn 12 alpha _ )NP )NP )PP-CLR )VP
16-17 nov._29 _ _ βnov_29 _ _
16 nov. nnp 9 beta (NP-TMP _
17 29 cd 16 beta _ )NP-TMP )VP )VP )S
18 . punct 9 beta _ )S

# Sentence: New York City is a global center for finance and culture.

IDX LEX POS HD ELEM RHS LHS
1-3 new_york_city _ _ αnew_york_city (S (S (NP-SBJ _
1 new jj 3 _ _ _
2 york nnp 3 _ _ _
3 city nnp 4 alpha _ )NP-SBJ
4 is vbz 0 alpha (VP _
5 a dt 8 alpha (NP (DT _
6 global jj 8 beta _ _
7-8 center_for _ _ βcenter_for _ _
7 center nn 4 alpha _ _
8 finance nn 4 alpha _ _
9 and cc 8 alpha _ _
10 culture nn 8 alpha _ )NP )VP )S
11 . punct 4 beta _ )S

# Sentence: Artificial intelligence has shown tremendous progress in recent years.

IDX LEX POS HD ELEM RHS LHS
1-2 artificial_intelligence _ _ αartificial_intelligence (S (S (NP-SBJ _
1 artificial jj 2 _ _ _
2 intelligence nn 3 alpha _ )NP-SBJ
3 has vbz 0 alpha (VP _
4 shown vbn 3 alpha (VP _
5 tremendous jj 6 beta (NP (JJ _
6 progress nn 4 alpha _ )NP
7 in in 6 alpha (PP _
8 recent jj 9 beta (NP (JJ _
9 years nns 7 alpha _ )NP )PP )VP )VP )S
10 . punct 3 beta _ )S
