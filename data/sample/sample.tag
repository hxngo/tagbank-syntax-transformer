# Sample TAGbank format
# Sentence: Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29.

IDX LEX POS HD ELEM RHS LHS
1 pierre nnp 2 beta (S (S (NP-SBJ (NP-SBJ (NP (NP _
2 vinken nnp 9 alpha _ )NP
3 , punct 2 beta _ )NP
4 61 cd 5 beta (ADJP (NP _
5 years nns 2 beta _ )NP
6 old jj 5 beta _ )ADJP )NP-SBJ
7 , punct 6 beta _ )NP-SBJ
8 will md 9 beta (VP (MD _
9 join vb 0 alpha (VP (VP _
10 the dt 11 alpha (NP (DT _
11 board nn 9 alpha _ )NP
12 as in 9 alpha (PP-CLR _
13 a dt 15 alpha (NP _
14 nonexecutive jj 15 beta (NP _
15 director nn 12 alpha _ )NP )NP )PP-CLR )VP
16 nov. nnp 9 beta (NP-TMP _
17 29 cd 16 beta _ )NP-TMP )VP )VP )S
18 . punct 9 beta _ )S

# Sentence: The researchers published their findings in the journal Science.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 researchers nns 3 alpha _ )NP )NP-SBJ
3 published vbd 0 alpha (VP _
4 their prp$ 5 alpha (NP (NP (PRP$ _
5 findings nns 3 alpha _ )NP
6 in in 3 alpha (PP _
7 the dt 9 alpha (NP (DT _
8 journal nn 9 beta _ )NP
9 science nnp 6 alpha _ )NP )PP )VP )S
10 . punct 3 beta _ )S

# Sentence: Climate change is causing more frequent and severe weather events worldwide.

IDX LEX POS HD ELEM RHS LHS
1 climate nn 2 beta (S (S (NP-SBJ (NP (NN _
2 change nn 3 alpha _ )NP )NP-SBJ
3 is vbz 0 alpha (VP _
4 causing vbg 3 alpha (VP _
5 more jjr 8 beta (NP (ADJP (JJR _
6 frequent jj 8 beta _ )ADJP
7 and cc 6 beta (ADJP _
8 severe jj 9 beta _ )ADJP
9 weather nn 10 beta _ )NP
10 events nns 4 alpha (NP _
11 worldwide rb 4 beta (ADVP _ )ADVP )VP )VP )S
12 . punct 3 beta _ )S
