file bin/fuzzy_lda.exe

#break CalculateLikelihood
#run ../../../Python/YJCFetch/NewsPages/corpus.txt		\
#../../../Python/YJCFetch/NewsPages/my_output.txt		\
#../../../Python/YJCFetch/NewsPages/basis_terms.txt	2

run ../../../Python/YJCFetch/NewsPages/fuzzy-corpus.txt	\
../../../Python/YJCFetch/NewsPages/my_output.txt		\
top_basis_terms.txt	2

