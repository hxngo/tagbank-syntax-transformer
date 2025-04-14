# Sample TAGbank format - Extended Dataset
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
15 director nn 12 alpha _ )NP )NP )PP-CLR )VP )VP )S
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

# Sentence: John Smith, the CEO of Tech Corp, announced a new product line yesterday.

IDX LEX POS HD ELEM RHS LHS
1 john nnp 2 beta (S (S (NP-SBJ (NP-SBJ (NP (NP _
2 smith nnp 8 alpha _ )NP
3 , punct 2 beta _ )NP
4 the dt 5 alpha (NP (DT _
5 ceo nn 2 beta _ )NP
6 of in 5 beta (PP _
7 tech nnp 8 beta (NP _
8 corp nnp 6 alpha _ )NP )PP )NP-SBJ
9 , punct 8 beta _ )NP-SBJ
10 announced vbd 0 alpha (VP _
11 a dt 13 alpha (NP (DT _
12 new jj 13 beta _ )NP
13 product nn 14 beta _ )NP
14 line nn 10 alpha (NP _
15 yesterday nn 10 beta (NP-TMP _ )NP-TMP )VP )S
16 . punct 10 beta _ )S

# Sentence: Students who complete their assignments on time tend to receive better grades.

IDX LEX POS HD ELEM RHS LHS
1 students nns 10 beta (S (S (NP-SBJ (NP _
2 who wp 3 beta (SBAR (WHNP _
3 complete vbp 1 alpha (S (VP _
4 their prp$ 5 alpha (NP (PRP$ _
5 assignments nns 3 alpha _ )NP
6 on in 3 beta (PP _
7 time nn 6 alpha (NP _ )NP )PP )VP )S )SBAR )NP )NP-SBJ
8 tend vbp 0 alpha (VP _
9 to to 10 beta _ )VP
10 receive vb 8 alpha (VP _
11 better jjr 12 beta (NP _
12 grades nns 10 alpha _ )NP )VP )S
13 . punct 8 beta _ )S

# Sentence: The company's revenue grew by 15 percent in the last fiscal quarter.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 company's nnp 3 beta _ )NP
3 revenue nn 4 alpha _ )NP )NP-SBJ
4 grew vbd 0 alpha (VP _
5 by in 4 alpha (PP _
6 15 cd 7 beta (NP _
7 percent nn 5 alpha _ )NP )PP
8 in in 4 beta (PP _
9 the dt 12 alpha (NP (DT _
10 last jj 12 beta _ )NP
11 fiscal jj 12 beta _ )NP
12 quarter nn 8 alpha _ )NP )PP )VP )S
13 . punct 4 beta _ )S

# Sentence: The museum will display artifacts from ancient civilizations throughout history.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 museum nn 4 alpha _ )NP )NP-SBJ
3 will md 4 beta (VP (MD _
4 display vb 0 alpha (VP _
5 artifacts nns 4 alpha (NP _
6 from in 5 beta (PP _
7 ancient jj 8 beta (NP _
8 civilizations nns 6 alpha _ )NP )PP
9 throughout in 5 beta (PP _
10 history nn 9 alpha (NP _ )NP )PP )NP )VP )VP )S
11 . punct 4 beta _ )S

# Sentence: Many experts believe that artificial intelligence will transform the job market.

IDX LEX POS HD ELEM RHS LHS
1 many jj 2 beta (S (S (NP-SBJ (NP _
2 experts nns 3 alpha _ )NP )NP-SBJ
3 believe vbp 0 alpha (VP _
4 that in 3 alpha (SBAR _
5 artificial jj 6 beta (S (NP-SBJ (NP _
6 intelligence nn 8 alpha _ )NP )NP-SBJ
7 will md 8 beta (VP (MD _
8 transform vb 4 alpha _ )VP
9 the dt 11 alpha (NP (DT _
10 job nn 11 beta _ )NP
11 market nn 8 alpha _ )NP )VP )S )SBAR )VP )S
12 . punct 3 beta _ )S

# Sentence: Despite the rain, thousands of people attended the outdoor concert last night.

IDX LEX POS HD ELEM RHS LHS
1 despite in 7 beta (S (S (PP (IN _
2 the dt 3 alpha (NP (DT _
3 rain nn 1 alpha _ )NP )PP
4 , punct 1 beta _ )PP
5 thousands nns 7 beta (NP-SBJ (NP _
6 of in 5 beta (PP _
7 people nns 8 alpha _ )PP )NP )NP-SBJ
8 attended vbd 0 alpha (VP _
9 the dt 11 alpha (NP (DT _
10 outdoor jj 11 beta _ )NP
11 concert nn 8 alpha _ )NP
12 last jj 13 beta (NP-TMP _
13 night nn 8 beta _ )NP-TMP )VP )S
14 . punct 8 beta _ )S

# Sentence: The government announced new policies to address environmental concerns.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 government nn 3 alpha _ )NP )NP-SBJ
3 announced vbd 0 alpha (VP _
4 new jj 5 beta (NP _
5 policies nns 3 alpha _ )NP
6 to to 7 beta (S (VP _
7 address vb 5 alpha _ )VP
8 environmental jj 9 beta (NP _
9 concerns nns 7 alpha _ )NP )S )VP )S
10 . punct 3 beta _ )S

# Sentence: Scientists discovered a new species of marine life in the deepest part of the ocean.

IDX LEX POS HD ELEM RHS LHS
1 scientists nns 2 beta (S (S (NP-SBJ (NP _
2 discovered vbd 0 alpha (VP _
3 a dt 5 alpha (NP (DT _
4 new jj 5 beta _ )NP
5 species nn 2 alpha _ )NP
6 of in 5 beta (PP _
7 marine jj 8 beta (NP _
8 life nn 6 alpha _ )NP )PP
9 in in 2 beta (PP _
10 the dt 12 alpha (NP (DT _
11 deepest jjs 12 beta _ )NP
12 part nn 9 alpha _ )NP
13 of in 12 beta (PP _
14 the dt 15 alpha (NP (DT _
15 ocean nn 13 alpha _ )NP )PP )PP )VP )S
16 . punct 2 beta _ )S

# Sentence: The patient showed significant improvement after undergoing the experimental treatment.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 patient nn 3 alpha _ )NP )NP-SBJ
3 showed vbd 0 alpha (VP _
4 significant jj 5 beta (NP _
5 improvement nn 3 alpha _ )NP
6 after in 3 beta (PP _
7 undergoing vbg 6 alpha (S (VP _
8 the dt 10 alpha (NP (DT _
9 experimental jj 10 beta _ )NP
10 treatment nn 7 alpha _ )NP )VP )S )PP )VP )S
11 . punct 3 beta _ )S

# Sentence: Local farmers have adopted sustainable practices to conserve water resources.

IDX LEX POS HD ELEM RHS LHS
1 local jj 2 beta (S (S (NP-SBJ (NP _
2 farmers nns 4 alpha _ )NP )NP-SBJ
3 have vbp 4 beta (VP (VBP _
4 adopted vbn 0 alpha (VP _
5 sustainable jj 6 beta (NP _
6 practices nns 4 alpha _ )NP
7 to to 8 beta (S (VP _
8 conserve vb 6 alpha _ )VP
9 water nn 10 beta (NP _
10 resources nns 8 alpha _ )NP )S )VP )VP )S
11 . punct 4 beta _ )S

# Sentence: The architect designed a building that maximizes natural light and energy efficiency.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 architect nn 3 alpha _ )NP )NP-SBJ
3 designed vbd 0 alpha (VP _
4 a dt 5 alpha (NP (DT _
5 building nn 3 alpha _ )NP
6 that wdt 7 beta (SBAR (WHNP _
7 maximizes vbz 5 alpha (S (VP _
8 natural jj 9 beta (NP _
9 light nn 7 alpha _ )NP
10 and cc 9 beta _ )NP
11 energy nn 12 beta (NP _
12 efficiency nn 7 beta _ )NP )VP )S )SBAR )VP )S
13 . punct 3 beta _ )S

# Sentence: Several countries have signed an agreement to reduce carbon emissions by 2030.

IDX LEX POS HD ELEM RHS LHS
1 several jj 2 beta (S (S (NP-SBJ (NP _
2 countries nns 4 alpha _ )NP )NP-SBJ
3 have vbp 4 beta (VP (VBP _
4 signed vbn 0 alpha (VP _
5 an dt 6 alpha (NP (DT _
6 agreement nn 4 alpha _ )NP
7 to to 8 beta (S (VP _
8 reduce vb 6 alpha _ )VP
9 carbon nn 10 beta (NP _
10 emissions nns 8 alpha _ )NP
11 by in 8 beta (PP _
12 2030 cd 11 alpha (NP _ )NP )PP )S )VP )VP )S
13 . punct 4 beta _ )S

# Sentence: The study examined the relationship between sleep patterns and cognitive performance.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 study nn 3 alpha _ )NP )NP-SBJ
3 examined vbd 0 alpha (VP _
4 the dt 5 alpha (NP (DT _
5 relationship nn 3 alpha _ )NP
6 between in 5 beta (PP _
7 sleep nn 8 beta (NP _
8 patterns nns 6 alpha _ )NP
9 and cc 8 beta _ )NP
10 cognitive jj 11 beta (NP _
11 performance nn 6 beta _ )NP )PP )VP )S
12 . punct 3 beta _ )S

# Sentence: The company announced plans to expand its operations into Asian markets next year.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 company nn 3 alpha _ )NP )NP-SBJ
3 announced vbd 0 alpha (VP _
4 plans nns 3 alpha (NP _
5 to to 6 beta (S (VP _
6 expand vb 4 alpha _ )VP
7 its prp$ 8 alpha (NP (PRP$ _
8 operations nns 6 alpha _ )NP
9 into in 6 beta (PP _
10 asian jj 11 beta (NP _
11 markets nns 9 alpha _ )NP )PP
12 next jj 13 beta (NP-TMP _
13 year nn 6 beta _ )NP-TMP )S )NP )VP )S
14 . punct 3 beta _ )S

# Sentence: The committee reviewed all proposals and selected three finalists for the competition.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 committee nn 3 alpha _ )NP )NP-SBJ
3 reviewed vbd 0 alpha (VP _
4 all dt 5 alpha (NP (DT _
5 proposals nns 3 alpha _ )NP
6 and cc 3 beta _ )NP
7 selected vbd 3 beta (VP _
8 three cd 9 beta (NP _
9 finalists nns 7 alpha _ )NP
10 for in 7 beta (PP _
11 the dt 12 alpha (NP (DT _
12 competition nn 10 alpha _ )NP )PP )VP )VP )S
13 . punct 3 beta _ )S

# Sentence: The professor explained complex theories using simple examples that students could understand.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 professor nn 3 alpha _ )NP )NP-SBJ
3 explained vbd 0 alpha (VP _
4 complex jj 5 beta (NP _
5 theories nns 3 alpha _ )NP
6 using vbg 3 beta (PP _
7 simple jj 8 beta (NP _
8 examples nns 6 alpha _ )NP
9 that wdt 12 beta (SBAR (WHNP _
10 students nns 12 beta (S (NP-SBJ _
11 could md 12 beta (VP _
12 understand vb 8 alpha _ )VP )S )SBAR )PP )VP )S
13 . punct 3 beta _ )S

# Sentence: The historical document reveals important information about ancient trading practices.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 historical jj 3 beta _ )NP
3 document nn 4 alpha _ )NP )NP-SBJ
4 reveals vbz 0 alpha (VP _
5 important jj 6 beta (NP _
6 information nn 4 alpha _ )NP
7 about in 6 beta (PP _
8 ancient jj 10 beta (NP _
9 trading vbg 10 beta _ )NP
10 practices nns 7 alpha _ )NP )PP )VP )S
11 . punct 4 beta _ )S

# Sentence: The company invested heavily in research and development to stay competitive in the market.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 company nn 3 alpha _ )NP )NP-SBJ
3 invested vbd 0 alpha (VP _
4 heavily rb 3 beta (ADVP _ )ADVP
5 in in 3 beta (PP _
6 research nn 8 beta (NP _
7 and cc 6 beta _ )NP
8 development nn 5 alpha _ )NP )PP
9 to to 10 beta (S (VP _
10 stay vb 3 beta _ )VP
11 competitive jj 10 alpha (ADJP _
12 in in 11 beta (PP _
13 the dt 14 alpha (NP (DT _
14 market nn 12 alpha _ )NP )PP )ADJP )S )VP )S
15 . punct 3 beta _ )S

# Sentence: The rescue team located all survivors after searching through the debris for hours.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 rescue nn 3 beta _ )NP
3 team nn 4 alpha _ )NP )NP-SBJ
4 located vbd 0 alpha (VP _
5 all dt 6 alpha (NP (DT _
6 survivors nns 4 alpha _ )NP
7 after in 4 beta (PP _
8 searching vbg 7 alpha (S (VP _
9 through in 8 beta (PP _
10 the dt 11 alpha (NP (DT _
11 debris nn 9 alpha _ )NP )PP
12 for in 8 beta (PP _
13 hours nns 12 alpha (NP _ )NP )PP )VP )S )PP )VP )S
14 . punct 4 beta _ )S

# Sentence: Many observers predict that renewable energy will become more affordable in the coming decade.

IDX LEX POS HD ELEM RHS LHS
1 many jj 2 beta (S (S (NP-SBJ (NP _
2 observers nns 3 alpha _ )NP )NP-SBJ
3 predict vbp 0 alpha (VP _
4 that in 3 alpha (SBAR _
5 renewable jj 6 beta (S (NP-SBJ (NP _
6 energy nn 8 alpha _ )NP )NP-SBJ
7 will md 8 beta (VP (MD _
8 become vb 4 alpha _ )VP
9 more jjr 10 beta (ADJP _
10 affordable jj 8 alpha _ )ADJP
11 in in 8 beta (PP _
12 the dt 14 alpha (NP (DT _
13 coming vbg 14 beta _ )NP
14 decade nn 11 alpha _ )NP )PP )VP )S )SBAR )VP )S
15 . punct 3 beta _ )S

# Sentence: The organization provides education and healthcare services to underserved communities.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 organization nn 3 alpha _ )NP )NP-SBJ
3 provides vbz 0 alpha (VP _
4 education nn 6 beta (NP _
5 and cc 4 beta _ )NP
6 healthcare nn 7 beta _ )NP
7 services nns 3 alpha (NP _
8 to to 7 beta (PP _
9 underserved jj 10 beta (NP _
10 communities nns 8 alpha _ )NP )PP )NP )VP )S
11 . punct 3 beta _ )S

# Sentence: The museum's new exhibition features artwork from renowned international artists.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 museum's nn 3 beta _ )NP
3 new jj 4 beta _ )NP
4 exhibition nn 5 alpha _ )NP )NP-SBJ
5 features vbz 0 alpha (VP _
6 artwork nn 5 alpha (NP _
7 from in 6 beta (PP _
8 renowned jj 10 beta (NP _
9 international jj 10 beta _ )NP
10 artists nns 7 alpha _ )NP )PP )NP )VP )S
11 . punct 5 beta _ )S

# Sentence: The financial report indicates strong growth in retail sales during the holiday season.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 financial jj 3 beta _ )NP
3 report nn 4 alpha _ )NP )NP-SBJ
4 indicates vbz 0 alpha (VP _
5 strong jj 6 beta (NP _
6 growth nn 4 alpha _ )NP
7 in in 6 beta (PP _
8 retail nn 9 beta (NP _
9 sales nns 7 alpha _ )NP )PP
10 during in 6 beta (PP _
11 the dt 13 alpha (NP (DT _
12 holiday nn 13 beta _ )NP
13 season nn 10 alpha _ )NP )PP )VP )S
14 . punct 4 beta _ )S

# Sentence: The film received critical acclaim for its innovative storytelling and visual effects.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 film nn 3 alpha _ )NP )NP-SBJ
3 received vbd 0 alpha (VP _
4 critical jj 5 beta (NP _
5 acclaim nn 3 alpha _ )NP
6 for in 3 beta (PP _
7 its prp$ 9 alpha (NP (NP (PRP$ _
8 innovative jj 9 beta _ )NP
9 storytelling nn 6 alpha _ )NP
10 and cc 9 beta _ )NP
11 visual jj 12 beta (NP _
12 effects nns 6 beta _ )NP )PP )VP )S
13 . punct 3 beta _ )S

# Sentence: The conference brought together experts from various fields to discuss interdisciplinary approaches.

IDX LEX POS HD ELEM RHS LHS
1 the dt 2 alpha (S (S (NP-SBJ (NP (DT _
2 conference nn 3 alpha _ )NP )NP-SBJ
3 brought vbd 0 alpha (VP _
4 together rb 3 beta (ADVP _ )ADVP
5 experts nns 3 alpha (NP _
6 from in 5 beta (PP _
7 various jj 8 beta (NP _
8 fields nns 6 alpha _ )NP )PP )NP
9 to to 10 beta (S (VP _
10 discuss vb 3 beta _ )VP
11 interdisciplinary jj 12 beta (NP _
12 approaches nns 10 alpha _ )NP )S )VP )S
13 . punct 3 beta _ )S

# Sentence: The survey results show a significant increase in consumer confidence over the past quarter.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 survey nn 3 beta _ )NP
3 results nns 4 alpha _ )NP )NP-SBJ
4 show vbp 0 alpha (VP _
5 a dt 7 alpha (NP (DT _
6 significant jj 7 beta _ )NP
7 increase nn 4 alpha _ )NP
8 in in 7 beta (PP _
9 consumer nn 10 beta (NP _
10 confidence nn 8 alpha _ )NP )PP
11 over in 7 beta (PP _
12 the dt 14 alpha (NP (DT _
13 past jj 14 beta _ )NP
14 quarter nn 11 alpha _ )NP )PP )VP )S
15 . punct 4 beta _ )S

# Sentence: The research team developed a breakthrough technology for renewable energy storage.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 research nn 3 beta _ )NP
3 team nn 4 alpha _ )NP )NP-SBJ
4 developed vbd 0 alpha (VP _
5 a dt 7 alpha (NP (DT _
6 breakthrough nn 7 beta _ )NP
7 technology nn 4 alpha _ )NP
8 for in 7 beta (PP _
9 renewable jj 10 beta (NP _
10 energy nn 11 beta _ )NP
11 storage nn 8 alpha _ )NP )PP )VP )S
12 . punct 4 beta _ )S

# Sentence: Global temperatures have risen significantly due to increased greenhouse gas emissions.

IDX LEX POS HD ELEM RHS LHS
1 global jj 2 beta (S (S (NP-SBJ (NP _
2 temperatures nns 4 alpha _ )NP )NP-SBJ
3 have vbp 4 beta (VP (VBP _
4 risen vbn 0 alpha (VP _
5 significantly rb 4 beta (ADVP _ )ADVP
6 due jj 4 beta (PP _
7 to to 6 beta _ )PP
8 increased vbn 10 beta (NP _
9 greenhouse nn 10 beta _ )NP
10 gas nn 11 beta _ )NP
11 emissions nns 7 alpha _ )NP )VP )VP )S
12 . punct 4 beta _ )S

# Sentence: The artificial intelligence system demonstrated remarkable problem-solving capabilities.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 artificial jj 3 beta _ )NP
3 intelligence nn 4 beta _ )NP
4 system nn 5 alpha _ )NP )NP-SBJ
5 demonstrated vbd 0 alpha (VP _
6 remarkable jj 8 beta (NP _
7 problem-solving jj 8 beta _ )NP
8 capabilities nns 5 alpha _ )NP )VP )S
9 . punct 5 beta _ )S

# Sentence: Scientists discovered evidence of ancient microbial life on Mars.

IDX LEX POS HD ELEM RHS LHS
1 scientists nns 2 beta (S (S (NP-SBJ (NP _
2 discovered vbd 0 alpha (VP _
3 evidence nn 2 alpha (NP _
4 of in 3 beta (PP _
5 ancient jj 7 beta (NP _
6 microbial jj 7 beta _ )NP
7 life nn 4 alpha _ )NP )PP
8 on in 3 beta (PP _
9 mars nnp 8 alpha (NP _ )NP )PP )NP )VP )S
10 . punct 2 beta _ )S

# Sentence: The quantum computer successfully completed complex calculations in milliseconds.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 quantum nn 3 beta _ )NP
3 computer nn 5 alpha _ )NP )NP-SBJ
4 successfully rb 5 beta (ADVP _ )ADVP
5 completed vbd 0 alpha (VP _
6 complex jj 7 beta (NP _
7 calculations nns 5 alpha _ )NP
8 in in 5 beta (PP _
9 milliseconds nns 8 alpha (NP _ )NP )PP )VP )S
10 . punct 5 beta _ )S

# Sentence: The autonomous vehicle navigated through dense urban traffic without human intervention.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 autonomous jj 3 beta _ )NP
3 vehicle nn 4 alpha _ )NP )NP-SBJ
4 navigated vbd 0 alpha (VP _
5 through in 4 beta (PP _
6 dense jj 8 beta (NP _
7 urban jj 8 beta _ )NP
8 traffic nn 5 alpha _ )NP )PP
9 without in 4 beta (PP _
10 human jj 11 beta (NP _
11 intervention nn 9 alpha _ )NP )PP )VP )S
12 . punct 4 beta _ )S

# Sentence: Researchers developed a new vaccine using advanced genetic engineering techniques.

IDX LEX POS HD ELEM RHS LHS
1 researchers nns 2 beta (S (S (NP-SBJ (NP _
2 developed vbd 0 alpha (VP _
3 a dt 5 alpha (NP (DT _
4 new jj 5 beta _ )NP
5 vaccine nn 2 alpha _ )NP
6 using vbg 2 beta (PP _
7 advanced jj 9 beta (NP _
8 genetic jj 9 beta _ )NP
9 engineering nn 10 beta _ )NP
10 techniques nns 6 alpha _ )NP )PP )VP )S
11 . punct 2 beta _ )S

# Sentence: The space telescope captured stunning images of distant galaxies forming.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 space nn 3 beta _ )NP
3 telescope nn 4 alpha _ )NP )NP-SBJ
4 captured vbd 0 alpha (VP _
5 stunning jj 6 beta (NP _
6 images nns 4 alpha _ )NP
7 of in 6 beta (PP _
8 distant jj 9 beta (NP _
9 galaxies nns 7 alpha _ )NP
10 forming vbg 9 beta (VP _ )VP )PP )VP )S
11 . punct 4 beta _ )S

# Sentence: The neural network learned to recognize complex patterns in large datasets.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 neural jj 3 beta _ )NP
3 network nn 4 alpha _ )NP )NP-SBJ
4 learned vbd 0 alpha (VP _
5 to to 6 beta (VP _
6 recognize vb 4 alpha _ )VP
7 complex jj 8 beta (NP _
8 patterns nns 6 alpha _ )NP
9 in in 6 beta (PP _
10 large jj 11 beta (NP _
11 datasets nns 9 alpha _ )NP )PP )VP )S
12 . punct 4 beta _ )S

# Sentence: The renewable energy plant generated enough power for the entire city.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 renewable jj 3 beta _ )NP
3 energy nn 4 beta _ )NP
4 plant nn 5 alpha _ )NP )NP-SBJ
5 generated vbd 0 alpha (VP _
6 enough jj 7 beta (NP _
7 power nn 5 alpha _ )NP
8 for in 5 beta (PP _
9 the dt 11 alpha (NP (DT _
10 entire jj 11 beta _ )NP
11 city nn 8 alpha _ )NP )PP )VP )S
12 . punct 5 beta _ )S

# Sentence: Artificial intelligence has revolutionized modern society.
IDX LEX POS HD ELEM RHS LHS
1 artificial jj 2 beta (S (NP-SBJ (NP _
2 intelligence nn 4 alpha _ )NP
3 has vbz 4 beta (VP _
4 revolutionized vbn 0 alpha (VP (VP _
5 modern jj 6 beta (NP _
6 society nn 4 alpha _ )NP )VP )S
7 . punct 4 beta _ )S

# Sentence: The climate crisis demands immediate global action.
IDX LEX POS HD ELEM RHS LHS
1 the dt 2 beta (S (NP-SBJ (NP (DT _
2 climate nn 3 alpha _ )NP
3 crisis nn 4 alpha _ )NP-SBJ
4 demands vbz 0 alpha (VP _
5 immediate jj 7 beta (NP (JJ _
6 global jj 7 beta _ )NP
7 action nn 4 alpha _ )VP )S
8 . punct 4 beta _ )S

# Sentence: Economic indicators show robust growth this quarter.
IDX LEX POS HD ELEM RHS LHS
1 economic jj 2 beta (S (NP-SBJ (NP (JJ _
2 indicators nns 3 alpha _ )NP-SBJ
3 show vbp 0 alpha (VP _
4 robust jj 5 beta (NP (JJ _
5 growth nn 3 alpha _ )NP
6 this dt 7 beta (NP (DT _
7 quarter nn 3 alpha _ )NP )VP )S
8 . punct 3 beta _ )S

# Sentence: Recent studies reveal significant improvements in vaccine efficacy.
IDX LEX POS HD ELEM RHS LHS
1 recent jj 2 beta (S (NP-SBJ (NP (JJ _
2 studies nns 3 alpha _ )NP-SBJ
3 reveal vbp 0 alpha (VP _
4 significant jj 5 beta (NP (JJ _
5 improvements nns 3 alpha _ )NP
6 in in 5 beta (PP (IN _
7 vaccine nn 8 beta (NP _
8 efficacy nn 5 alpha _ )NP )PP )VP )S
9 . punct 3 beta _ )S

# Sentence: Online learning platforms have transformed educational access worldwide.
IDX LEX POS HD ELEM RHS LHS
1 online jj 3 beta (S (NP-SBJ (NP (JJ _
2 learning nn 3 beta _ )NP
3 platforms nns 4 alpha _ )NP-SBJ
4 have vbp 4 beta (VP (VBP _
5 transformed vbn 0 alpha (VP _
6 educational jj 7 beta (NP (JJ _
7 access nn 4 alpha _ )NP
8 worldwide rb 7 beta (ADVP _ )ADVP )VP )VP )S
9 . punct 4 beta _ )S

# Sentence: The spacecraft successfully entered orbit around Mars.
IDX LEX POS HD ELEM RHS LHS
1 the dt 2 beta (S (NP-SBJ (NP (DT _
2 spacecraft nn 4 alpha _ )NP-SBJ
3 successfully rb 4 beta (ADVP _ )ADVP
4 entered vbd 0 alpha (VP _
5 orbit nn 4 alpha (NP _
6 around in 5 beta (PP (IN _
7 mars nnp 6 alpha _ )NP )PP )VP )S
8 . punct 4 beta _ )S

# Sentence: New legislation aims to strengthen data privacy rights.
IDX LEX POS HD ELEM RHS LHS
1 new jj 2 beta (S (NP-SBJ (NP (JJ _
2 legislation nn 3 alpha _ )NP-SBJ
3 aims vbz 0 alpha (VP _
4 to to 5 beta (VP _
5 strengthen vb 3 alpha _ )VP
6 data nn 7 beta (NP (NN _
7 privacy nn 8 beta _ )NP
8 rights nns 3 alpha _ )VP )S
9 . punct 3 beta _ )S

# Sentence: The exhibition showcases a diverse range of contemporary art.
IDX LEX POS HD ELEM RHS LHS
1 the dt 2 beta (S (NP-SBJ (NP (DT _
2 exhibition nn 3 alpha _ )NP-SBJ
3 showcases vbz 0 alpha (VP _
4 a dt 6 alpha (NP (DT _
5 diverse jj 6 beta _ )NP
6 range nn 3 alpha _ )NP
7 of in 6 beta (PP (IN _
8 contemporary jj 9 beta (NP (JJ _
9 art nn 7 alpha _ )NP )PP )VP )S
10 . punct 3 beta _ )S

# Sentence: Historians uncover new insights into ancient civilizations.
IDX LEX POS HD ELEM RHS LHS
1 historians nns 2 beta (S (NP-SBJ (NP-SBJ (NNS _
2 uncover vbp 0 alpha (VP _
3 new jj 4 beta (NP (JJ _
4 insights nns 2 alpha _ )NP
5 into in 4 beta (PP (IN _
6 ancient jj 7 beta (NP (JJ _
7 civilizations nns 5 alpha _ )NP )PP )VP )S
8 . punct 2 beta _ )S

# Sentence: The championship match ended in a thrilling overtime victory.
IDX LEX POS HD ELEM RHS LHS
1 the dt 2 beta (S (NP-SBJ (NP (DT _
2 championship nn 3 alpha _ )NP-SBJ
3 match nn 4 alpha _ )NP
4 ended vbd 0 alpha (VP _
5 in in 4 beta (PP (IN _
6 a dt 8 alpha (NP (DT _
7 thrilling jj 8 beta _ )NP
8 overtime nn 4 alpha _ )NP )PP )VP )S
9 victory nn 4 beta _ )S
10 . punct 4 beta _ )S

# Sentence: The technology startup secured significant funding for their innovative project.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 technology nn 3 beta _ )NP
3 startup nn 4 alpha _ )NP )NP-SBJ
4 secured vbd 0 alpha (VP _
5 significant jj 6 beta (NP _
6 funding nn 4 alpha _ )NP
7 for in 4 beta (PP _
8 their prp$ 10 alpha (NP (NP (PRP$ _
9 innovative jj 10 beta _ )NP
10 project nn 7 alpha _ )NP )PP )VP )S
11 . punct 4 beta _ )S

# Sentence: Global markets responded positively to the new economic policies.

IDX LEX POS HD ELEM RHS LHS
1 global jj 2 beta (S (S (NP-SBJ (NP _
2 markets nns 3 alpha _ )NP )NP-SBJ
3 responded vbd 0 alpha (VP _
4 positively rb 3 beta (ADVP _ )ADVP
5 to in 3 beta (PP _
6 the dt 9 alpha (NP (DT _
7 new jj 9 beta _ )NP
8 economic jj 9 beta _ )NP
9 policies nns 5 alpha _ )NP )PP )VP )S
10 . punct 3 beta _ )S

# Sentence: The research team published groundbreaking findings in quantum computing.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 research nn 3 beta _ )NP
3 team nn 4 alpha _ )NP )NP-SBJ
4 published vbd 0 alpha (VP _
5 groundbreaking jj 6 beta (NP _
6 findings nns 4 alpha _ )NP
7 in in 4 beta (PP _
8 quantum jj 9 beta (NP _
9 computing nn 7 alpha _ )NP )PP )VP )S
10 . punct 4 beta _ )S

# Sentence: Environmental activists organized protests against industrial pollution.

IDX LEX POS HD ELEM RHS LHS
1 environmental jj 2 beta (S (S (NP-SBJ (NP _
2 activists nns 3 alpha _ )NP )NP-SBJ
3 organized vbd 0 alpha (VP _
4 protests nns 3 alpha (NP _
5 against in 3 beta (PP _
6 industrial jj 7 beta (NP _
7 pollution nn 5 alpha _ )NP )PP )VP )S
8 . punct 3 beta _ )S

# Sentence: The autonomous vehicle successfully completed extensive road tests.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 autonomous jj 3 beta _ )NP
3 vehicle nn 4 alpha _ )NP )NP-SBJ
4 successfully rb 5 beta (VP (ADVP _
5 completed vbd 0 alpha _ )ADVP
6 extensive jj 8 beta (NP _
7 road nn 8 beta _ )NP
8 tests nns 5 alpha _ )NP )VP )S
9 . punct 5 beta _ )S

# Sentence: Medical researchers developed promising treatments for rare diseases.

IDX LEX POS HD ELEM RHS LHS
1 medical jj 2 beta (S (S (NP-SBJ (NP _
2 researchers nns 3 alpha _ )NP )NP-SBJ
3 developed vbd 0 alpha (VP _
4 promising jj 5 beta (NP _
5 treatments nns 3 alpha _ )NP
6 for in 3 beta (PP _
7 rare jj 8 beta (NP _
8 diseases nns 6 alpha _ )NP )PP )VP )S
9 . punct 3 beta _ )S

# Sentence: The international conference attracted experts from various fields of study.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 international jj 3 beta _ )NP
3 conference nn 4 alpha _ )NP )NP-SBJ
4 attracted vbd 0 alpha (VP _
5 experts nns 4 alpha (NP _
6 from in 5 beta (PP _
7 various jj 8 beta (NP _
8 fields nns 6 alpha _ )NP
9 of in 8 beta (PP _
10 study nn 9 alpha (NP _ )NP )PP )PP )VP )S
11 . punct 4 beta _ )S

# Sentence: Renewable energy sources are becoming increasingly important worldwide.

IDX LEX POS HD ELEM RHS LHS
1 renewable jj 2 beta (S (S (NP-SBJ (NP _
2 energy nn 3 beta _ )NP
3 sources nns 4 alpha _ )NP )NP-SBJ
4 are vbp 0 alpha (VP _
5 becoming vbg 4 alpha (VP _
6 increasingly rb 7 beta (ADJP _
7 important jj 5 alpha _ )ADJP
8 worldwide rb 5 beta (ADVP _ )ADVP )VP )VP )S
9 . punct 4 beta _ )S

# Sentence: The archaeological team discovered ancient artifacts buried beneath the city.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 archaeological jj 3 beta _ )NP
3 team nn 4 alpha _ )NP )NP-SBJ
4 discovered vbd 0 alpha (VP _
5 ancient jj 6 beta (NP _
6 artifacts nns 4 alpha _ )NP
7 buried vbn 6 beta (VP _
8 beneath in 7 beta (PP _
9 the dt 10 alpha (NP (DT _
10 city nn 8 alpha _ )NP )PP )VP )VP )S
11 . punct 4 beta _ )S

# Sentence: The digital transformation initiative improved operational efficiency across departments.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 digital jj 3 beta _ )NP
3 transformation nn 4 beta _ )NP
4 initiative nn 5 alpha _ )NP-SBJ
5 improved vbd 0 alpha (VP _
6 operational jj 7 beta (NP _
7 efficiency nn 5 alpha _ )NP
8 across in 5 beta (PP _
9 departments nns 8 alpha (NP _ )NP )PP )VP )S
10 . punct 5 beta _ )S

# Sentence: Advanced robotics systems are revolutionizing manufacturing processes worldwide.

IDX LEX POS HD ELEM RHS LHS
1 advanced jj 3 beta (S (S (NP-SBJ (NP _
2 robotics nn 3 beta _ )NP
3 systems nns 4 alpha _ )NP )NP-SBJ
4 are vbp 0 alpha (VP _
5 revolutionizing vbg 4 alpha (VP _
6 manufacturing nn 7 beta (NP _
7 processes nns 5 alpha _ )NP
8 worldwide rb 5 beta (ADVP _ )ADVP )VP )VP )S
9 . punct 4 beta _ )S

# Sentence: The cybersecurity team detected and prevented multiple sophisticated attacks.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 cybersecurity nn 3 beta _ )NP
3 team nn 4 alpha _ )NP )NP-SBJ
4 detected vbd 0 alpha (VP _
5 and cc 4 beta _ )VP
6 prevented vbd 4 beta (VP _
7 multiple jj 9 beta (NP _
8 sophisticated jj 9 beta _ )NP
9 attacks nns 6 alpha _ )NP )VP )VP )S
10 . punct 4 beta _ )S

# Sentence: Sustainable urban development projects focus on reducing environmental impact.

IDX LEX POS HD ELEM RHS LHS
1 sustainable jj 3 beta (S (S (NP-SBJ (NP _
2 urban jj 3 beta _ )NP
3 development nn 4 beta _ )NP
4 projects nns 5 alpha _ )NP-SBJ
5 focus vbp 0 alpha (VP _
6 on in 5 beta (PP _
7 reducing vbg 6 alpha (S (VP _
8 environmental jj 9 beta (NP _
9 impact nn 7 alpha _ )NP )VP )S )PP )VP )S
10 . punct 5 beta _ )S

# Sentence: The biotechnology company announced breakthrough results in gene therapy trials.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 biotechnology nn 3 beta _ )NP
3 company nn 4 alpha _ )NP )NP-SBJ
4 announced vbd 0 alpha (VP _
5 breakthrough nn 6 beta (NP _
6 results nns 4 alpha _ )NP
7 in in 6 beta (PP _
8 gene nn 9 beta (NP _
9 therapy nn 10 beta _ )NP
10 trials nns 7 alpha _ )NP )PP )VP )S
11 . punct 4 beta _ )S

# Sentence: Machine learning algorithms identified patterns in complex financial data.

IDX LEX POS HD ELEM RHS LHS
1 machine nn 2 beta (S (S (NP-SBJ (NP _
2 learning nn 3 beta _ )NP
3 algorithms nns 4 alpha _ )NP )NP-SBJ
4 identified vbd 0 alpha (VP _
5 patterns nns 4 alpha (NP _
6 in in 5 beta (PP _
7 complex jj 9 beta (NP _
8 financial jj 9 beta _ )NP
9 data nn 6 alpha _ )NP )PP )VP )S
10 . punct 4 beta _ )S

# Sentence: The virtual reality experience transported users to ancient historical sites.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 virtual jj 3 beta _ )NP
3 reality nn 4 beta _ )NP
4 experience nn 5 alpha _ )NP )NP-SBJ
5 transported vbd 0 alpha (VP _
6 users nns 5 alpha (NP _
7 to to 5 beta (PP _
8 ancient jj 10 beta (NP _
9 historical jj 10 beta _ )NP
10 sites nns 7 alpha _ )NP )PP )VP )S
11 . punct 5 beta _ )S

# Sentence: Innovative startups are disrupting traditional business models worldwide.

IDX LEX POS HD ELEM RHS LHS
1 innovative jj 2 beta (S (S (NP-SBJ (NP _
2 startups nns 3 alpha _ )NP )NP-SBJ
3 are vbp 0 alpha (VP _
4 disrupting vbg 3 alpha (VP _
5 traditional jj 7 beta (NP _
6 business nn 7 beta _ )NP
7 models nns 4 alpha _ )NP
8 worldwide rb 4 beta (ADVP _ )ADVP )VP )VP )S
9 . punct 3 beta _ )S

# Sentence: The quantum encryption system provided unbreakable security for sensitive data.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 quantum jj 3 beta _ )NP
3 encryption nn 4 beta _ )NP
4 system nn 5 alpha _ )NP )NP-SBJ
5 provided vbd 0 alpha (VP _
6 unbreakable jj 7 beta (NP _
7 security nn 5 alpha _ )NP
8 for in 7 beta (PP _
9 sensitive jj 10 beta (NP _
10 data nn 8 alpha _ )NP )PP )VP )S
11 . punct 5 beta _ )S

# Sentence: Environmental scientists monitored changes in global biodiversity patterns.

IDX LEX POS HD ELEM RHS LHS
1 environmental jj 2 beta (S (S (NP-SBJ (NP _
2 scientists nns 3 alpha _ )NP )NP-SBJ
3 monitored vbd 0 alpha (VP _
4 changes nns 3 alpha (NP _
5 in in 4 beta (PP _
6 global jj 8 beta (NP _
7 biodiversity nn 8 beta _ )NP
8 patterns nns 5 alpha _ )NP )PP )VP )S
9 . punct 3 beta _ )S

# Sentence: The blockchain technology revolutionized financial transactions and digital security.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 blockchain nn 3 beta _ )NP
3 technology nn 4 alpha _ )NP )NP-SBJ
4 revolutionized vbd 0 alpha (VP _
5 financial jj 6 beta (NP _
6 transactions nns 4 alpha _ )NP
7 and cc 6 beta _ )NP
8 digital jj 9 beta (NP _
9 security nn 4 beta _ )NP )VP )S
10 . punct 4 beta _ )S

# Sentence: Advanced materials scientists developed self-healing polymers for sustainable manufacturing.

IDX LEX POS HD ELEM RHS LHS
1 advanced jj 3 beta (S (S (NP-SBJ (NP _
2 materials nn 3 beta _ )NP
3 scientists nns 4 alpha _ )NP )NP-SBJ
4 developed vbd 0 alpha (VP _
5 self-healing jj 6 beta (NP _
6 polymers nns 4 alpha _ )NP
7 for in 4 beta (PP _
8 sustainable jj 9 beta (NP _
9 manufacturing nn 7 alpha _ )NP )PP )VP )S
10 . punct 4 beta _ )S

# Sentence: The quantum computing breakthrough enabled unprecedented computational capabilities.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 quantum jj 3 beta _ )NP
3 computing nn 4 beta _ )NP
4 breakthrough nn 5 alpha _ )NP )NP-SBJ
5 enabled vbd 0 alpha (VP _
6 unprecedented jj 8 beta (NP _
7 computational jj 8 beta _ )NP
8 capabilities nns 5 alpha _ )NP )VP )S
9 . punct 5 beta _ )S

# Sentence: Neuroscientists mapped complex neural networks controlling cognitive functions.

IDX LEX POS HD ELEM RHS LHS
1 neuroscientists nns 2 beta (S (S (NP-SBJ (NP _
2 mapped vbd 0 alpha (VP _
3 complex jj 5 beta (NP _
4 neural jj 5 beta _ )NP
5 networks nns 2 alpha _ )NP
6 controlling vbg 5 beta (VP _
7 cognitive jj 8 beta (NP _
8 functions nns 6 alpha _ )NP )VP )VP )S
9 . punct 2 beta _ )S

# Sentence: The space exploration mission discovered potential signs of extraterrestrial life.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 space nn 3 beta _ )NP
3 exploration nn 4 beta _ )NP
4 mission nn 5 alpha _ )NP )NP-SBJ
5 discovered vbd 0 alpha (VP _
6 potential jj 7 beta (NP _
7 signs nns 5 alpha _ )NP
8 of in 7 beta (PP _
9 extraterrestrial jj 10 beta (NP _
10 life nn 8 alpha _ )NP )PP )VP )S
11 . punct 5 beta _ )S

# Sentence: The renewable energy initiative transformed urban power distribution systems.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 renewable jj 3 beta _ )NP
3 energy nn 4 beta _ )NP
4 initiative nn 5 alpha _ )NP )NP-SBJ
5 transformed vbd 0 alpha (VP _
6 urban jj 8 beta (NP _
7 power nn 8 beta _ )NP
8 distribution nn 9 beta _ )NP
9 systems nns 5 alpha _ )NP )VP )S
10 . punct 5 beta _ )S

# Sentence: Biotechnology researchers developed gene-editing techniques for disease prevention.

IDX LEX POS HD ELEM RHS LHS
1 biotechnology nn 2 beta (S (S (NP-SBJ (NP _
2 researchers nns 3 alpha _ )NP )NP-SBJ
3 developed vbd 0 alpha (VP _
4 gene-editing jj 5 beta (NP _
5 techniques nns 3 alpha _ )NP
6 for in 3 beta (PP _
7 disease nn 8 beta (NP _
8 prevention nn 6 alpha _ )NP )PP )VP )S
9 . punct 3 beta _ )S

# Sentence: The artificial intelligence system demonstrated advanced learning capabilities.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 artificial jj 3 beta _ )NP
3 intelligence nn 4 beta _ )NP
4 system nn 5 alpha _ )NP )NP-SBJ
5 demonstrated vbd 0 alpha (VP _
6 advanced jj 8 beta (NP _
7 learning nn 8 beta _ )NP
8 capabilities nns 5 alpha _ )NP )VP )S
9 . punct 5 beta _ )S

# Sentence: Environmental monitoring systems detected significant changes in ocean temperatures.

IDX LEX POS HD ELEM RHS LHS
1 environmental jj 3 beta (S (S (NP-SBJ (NP _
2 monitoring nn 3 beta _ )NP
3 systems nns 4 alpha _ )NP )NP-SBJ
4 detect vbp 0 alpha (VP _
5 climate nn 6 beta (NP _
6 change nn 7 beta _ )NP
7 patterns nns 4 alpha _ )NP
8 accurately rb 4 beta (ADVP _ )ADVP )VP )S
9 . punct 4 beta _ )S

# Sentence: The quantum cryptography protocol ensured unbreakable communication security.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 quantum jj 3 beta _ )NP
3 cryptography nn 4 beta _ )NP
4 protocol nn 5 alpha _ )NP )NP-SBJ
5 ensured vbd 0 alpha (VP _
6 unbreakable jj 8 beta (NP _
7 communication nn 8 beta _ )NP
8 security nn 5 alpha _ )NP )VP )S
9 . punct 5 beta _ )S

# Sentence: The nanotechnology research revealed promising applications in medicine.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 nanotechnology nn 3 beta _ )NP
3 research nn 4 alpha _ )NP )NP-SBJ
4 revealed vbd 0 alpha (VP _
5 promising jj 6 beta (NP _
6 applications nns 4 alpha _ )NP
7 in in 6 beta (PP _
8 medicine nn 7 alpha (NP _ )NP )PP )VP )S
9 . punct 4 beta _ )S

# Sentence: Cloud computing platforms enable scalable business solutions globally.

IDX LEX POS HD ELEM RHS LHS
1 cloud nn 2 beta (S (S (NP-SBJ (NP _
2 computing nn 3 beta _ )NP
3 platforms nns 4 alpha _ )NP )NP-SBJ
4 enable vbp 0 alpha (VP _
5 scalable jj 7 beta (NP _
6 business nn 7 beta _ )NP
7 solutions nns 4 alpha _ )NP
8 globally rb 4 beta (ADVP _ )ADVP )VP )S
9 . punct 4 beta _ )S

# Sentence: Smart city initiatives integrate IoT sensors for urban management.

IDX LEX POS HD ELEM RHS LHS
1 smart jj 2 beta (S (S (NP-SBJ (NP _
2 city nn 3 beta _ )NP
3 initiatives nns 4 alpha _ )NP )NP-SBJ
4 integrate vbp 0 alpha (VP _
5 iot nn 6 beta (NP _
6 sensors nns 4 alpha _ )NP
7 for in 4 beta (PP _
8 urban jj 9 beta (NP _
9 management nn 7 alpha _ )NP )PP )VP )S
10 . punct 4 beta _ )S

# Sentence: Renewable energy investments show significant growth potential.

IDX LEX POS HD ELEM RHS LHS
1 renewable jj 2 beta (S (S (NP-SBJ (NP _
2 energy nn 3 beta _ )NP
3 investments nns 4 alpha _ )NP )NP-SBJ
4 show vbp 0 alpha (VP _
5 significant jj 7 beta (NP _
6 growth nn 7 beta _ )NP
7 potential nn 4 alpha _ )NP )VP )S
8 . punct 4 beta _ )S

# Sentence: The autonomous drone system monitors agricultural conditions effectively.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 autonomous jj 3 beta _ )NP
3 drone nn 4 beta _ )NP
4 system nn 5 alpha _ )NP )NP-SBJ
5 monitors vbz 0 alpha (VP _
6 agricultural jj 7 beta (NP _
7 conditions nns 5 alpha _ )NP
8 effectively rb 5 beta (ADVP _ )ADVP )VP )S
9 . punct 5 beta _ )S

# Sentence: Digital transformation accelerates business process optimization.

IDX LEX POS HD ELEM RHS LHS
1 digital jj 2 beta (S (S (NP-SBJ (NP _
2 transformation nn 3 alpha _ )NP )NP-SBJ
3 accelerates vbz 0 alpha (VP _
4 business nn 5 beta (NP _
5 process nn 6 beta _ )NP
6 optimization nn 3 alpha _ )NP )VP )S
7 . punct 3 beta _ )S

# Sentence: The quantum sensor detects microscopic magnetic field variations.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 quantum jj 3 beta _ )NP
3 sensor nn 4 alpha _ )NP )NP-SBJ
4 detects vbz 0 alpha (VP _
5 microscopic jj 8 beta (NP _
6 magnetic jj 7 beta _ )NP
7 field nn 8 beta _ )NP
8 variations nns 4 alpha _ )NP )VP )S
9 . punct 4 beta _ )S

# Sentence: Advanced materials enable efficient energy storage solutions.

IDX LEX POS HD ELEM RHS LHS
1 advanced jj 2 beta (S (S (NP-SBJ (NP _
2 materials nns 3 alpha _ )NP )NP-SBJ
3 enable vbp 0 alpha (VP _
4 efficient jj 7 beta (NP _
5 energy nn 6 beta _ )NP
6 storage nn 7 beta _ )NP
7 solutions nns 3 alpha _ )NP )VP )S
8 . punct 3 beta _ )S

# Sentence: The neural interface translates brain signals into digital commands.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 neural jj 3 beta _ )NP
3 interface nn 4 alpha _ )NP )NP-SBJ
4 translates vbz 0 alpha (VP _
5 brain nn 6 beta (NP _
6 signals nns 4 alpha _ )NP
7 into in 6 beta (PP _
8 digital jj 9 beta (NP _
9 commands nns 7 alpha _ )NP )PP )VP )S
10 . punct 4 beta _ )S

# Sentence: Sustainable manufacturing practices reduce environmental impact significantly.

IDX LEX POS HD ELEM RHS LHS
1 sustainable jj 3 beta (S (S (NP-SBJ (NP _
2 manufacturing nn 3 beta _ )NP
3 practices nns 4 alpha _ )NP )NP-SBJ
4 reduce vbp 0 alpha (VP _
5 environmental jj 6 beta (NP _
6 impact nn 4 alpha _ )NP
7 significantly rb 4 beta (ADVP _ )ADVP )VP )S
8 . punct 4 beta _ )S

# Sentence: The quantum mechanics research revealed fundamental particle interactions.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 quantum jj 3 beta _ )NP
3 mechanics nn 4 beta _ )NP
4 research nn 5 alpha _ )NP )NP-SBJ
5 revealed vbd 0 alpha (VP _
6 fundamental jj 8 beta (NP _
7 particle nn 8 beta _ )NP
8 interactions nns 5 alpha _ )NP )VP )S
9 . punct 5 beta _ )S

# Sentence: Advanced robotics technology enables precise surgical procedures.

IDX LEX POS HD ELEM RHS LHS
1 advanced jj 3 beta (S (S (NP-SBJ (NP _
2 robotics nn 3 beta _ )NP
3 technology nn 4 alpha _ )NP )NP-SBJ
4 enables vbz 0 alpha (VP _
5 precise jj 7 beta (NP _
6 surgical jj 7 beta _ )NP
7 procedures nns 4 alpha _ )NP )VP )S
8 . punct 4 beta _ )S

# Sentence: The artificial neural network processes complex data patterns efficiently.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 artificial jj 3 beta _ )NP
3 neural jj 4 beta _ )NP
4 network nn 5 alpha _ )NP )NP-SBJ
5 processes vbz 0 alpha (VP _
6 complex jj 8 beta (NP _
7 data nn 8 beta _ )NP
8 patterns nns 5 alpha _ )NP
9 efficiently rb 5 beta (ADVP _ )ADVP )VP )S
10 . punct 5 beta _ )S

# Sentence: Sustainable energy solutions reduce carbon emissions significantly.

IDX LEX POS HD ELEM RHS LHS
1 sustainable jj 3 beta (S (S (NP-SBJ (NP _
2 energy nn 3 beta _ )NP
3 solutions nns 4 alpha _ )NP )NP-SBJ
4 reduce vbp 0 alpha (VP _
5 carbon nn 6 beta (NP _
6 emissions nns 4 alpha _ )NP
7 significantly rb 4 beta (ADVP _ )ADVP )VP )S
8 . punct 4 beta _ )S

# Sentence: The biotechnology startup develops innovative cancer treatments.

IDX LEX POS HD ELEM RHS LHS
1 the dt 3 alpha (S (S (NP-SBJ (NP (DT _
2 biotechnology nn 3 beta _ )NP
3 startup nn 4 alpha _ )NP )NP-SBJ
4 develops vbz 0 alpha (VP _
5 innovative jj 7 beta (NP _
6 cancer nn 7 beta _ )NP
7 treatments nns 4 alpha _ )NP )VP )S
8 . punct 4 beta _ )S

# Sentence: Advanced machine learning algorithms optimize industrial processes efficiently.

IDX LEX POS HD ELEM RHS LHS
1 advanced jj 3 beta (S (S (NP-SBJ (NP _
2 machine nn 3 beta _ )NP
3 learning nn 4 beta _ )NP
4 algorithms nns 5 alpha _ )NP-SBJ
5 optimize vbp 0 alpha (VP _
6 industrial jj 7 beta (NP _
7 processes nns 5 alpha _ )NP
8 efficiently rb 5 beta (ADVP _ )ADVP )VP )S
9 . punct 5 beta _ )S

# Sentence: The quantum computing research demonstrates breakthrough performance improvements.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 quantum jj 3 beta _ )NP
3 computing nn 4 beta _ )NP
4 research nn 5 alpha _ )NP )NP-SBJ
5 demonstrates vbz 0 alpha (VP _
6 breakthrough nn 8 beta (NP _
7 performance nn 8 beta _ )NP
8 improvements nns 5 alpha _ )NP )VP )S
9 . punct 5 beta _ )S

# Sentence: Environmental monitoring systems detect climate change patterns accurately.

IDX LEX POS HD ELEM RHS LHS
1 environmental jj 3 beta (S (S (NP-SBJ (NP _
2 monitoring nn 3 beta _ )NP
3 systems nns 4 alpha _ )NP )NP-SBJ
4 detect vbp 0 alpha (VP _
5 climate nn 6 beta (NP _
6 change nn 7 beta _ )NP
7 patterns nns 4 alpha _ )NP
8 accurately rb 4 beta (ADVP _ )ADVP )VP )S
9 . punct 4 beta _ )S

# Sentence: Innovative blockchain solutions transform financial transaction systems globally.

IDX LEX POS HD ELEM RHS LHS
1 innovative jj 3 beta (S (S (NP-SBJ (NP _
2 blockchain nn 3 beta _ )NP
3 solutions nns 4 alpha _ )NP )NP-SBJ
4 transform vbp 0 alpha (VP _
5 financial jj 7 beta (NP _
6 transaction nn 7 beta _ )NP
7 systems nns 4 alpha _ )NP
8 globally rb 4 beta (ADVP _ )ADVP )VP )S
9 . punct 4 beta _ )S

# Sentence: The autonomous navigation system processes sensor data intelligently.

IDX LEX POS HD ELEM RHS LHS
1 the dt 4 alpha (S (S (NP-SBJ (NP (DT _
2 autonomous jj 3 beta _ )NP
3 navigation nn 4 beta _ )NP
4 system nn 5 alpha _ )NP )NP-SBJ
5 processes vbz 0 alpha (VP _
6 sensor nn 7 beta (NP _
7 data nn 5 alpha _ )NP
8 intelligently rb 5 beta (ADVP _ )ADVP )VP )S
9 . punct 5 beta _ )S