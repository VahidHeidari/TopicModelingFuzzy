#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define INIT_RANDOM

#define Z_IDX(D, N)				((D * NUM_BASIS_TERMS) + N)
#define SMPL_Z_IDX(D, N, K)		((D * NUM_BASIS_TERMS * NUM_TOPICS) + (N * NUM_TOPICS) + K)
#define THETA_IDX(D, K)  		((D * NUM_TOPICS) + K)
#define CNT_TPC_IDX(D, K)		((D * NUM_TOPICS) + K)
#define PHI_IDX(K, N)    		((K * NUM_BASIS_TERMS) + N)
#define CNT_WRD_IDX(K, N)		((K * NUM_BASIS_TERMS) + N)
#define FUZZY_WRD_IDX(D, BT)	((D * NUM_BASIS_TERMS) + BT)



// Constants
constexpr int MCMC_ITERATIONS = 1000;
constexpr int BURNIN_ITERATIONS = MCMC_ITERATIONS / 3;
constexpr int TOTAL_MAX_ITERATIONS = MCMC_ITERATIONS + BURNIN_ITERATIONS;

constexpr int MIN_SAMPLE_SIZE = 100;
constexpr double EPSILON = 1e-7;
constexpr int THIN_NUM = 2;

/// Parameters
constexpr double ALPHA = 0.5;
constexpr double BETA = 0.01;

int NUM_TOPICS = 0;					/// K
int NUM_VOCABS = 0;					/// V
int MAX_WORD_IN_DOC = 0;			/// N
int NUM_DOCS = 0;					/// D
int MIN_TERM = 0;
int TOTAL_WORDS = 0;
std::random_device dev;

std::vector<int> z;					/// [D][V]
std::vector<double> cnt_words;		/// [K][V]
std::vector<double> cnt_topics;		/// [D][K]
std::vector<double> phi;			/// [K][V]
std::vector<double> theta;			/// [D][K]

/// Samples
int num_smpls = 0;
std::vector<int> smpl_z;
std::vector<double> smpl_phi;
std::vector<double> smpl_theta;

/// Fuzzy bag of word
int NUM_BASIS_TERMS = 0;
int NUM_WORD_2_VEC = 0;
int FBOW_SIZE = 0;
std::vector<double> words_vectors;
std::vector<int> basis_terms;
std::vector<double> fuzzy_corpus;



struct Word
{
	int term;
	int count;
};

typedef std::vector<Word> Doc;
typedef std::vector<Doc> Corpus;



bool ReadCorpus(const char* corpus_path, Corpus& corpus)
{
	std::ifstream in_file(corpus_path);
	if (!in_file.is_open())
		return false;

	TOTAL_WORDS = 0;
	corpus.clear();
	int min_term = std::numeric_limits<int>::max();
	int max_term = std::numeric_limits<int>::min();
	int max_words_in_doc = std::numeric_limits<int>::min();
	std::string line;
	while (std::getline(in_file, line)) {
		std::istringstream iss(line);
		int num_terms;
		iss >> num_terms;													// Read number of words in current document.

		// Read words.
		Doc doc;
		int total_count = 0;
		for (int n = 0; n < num_terms; ++n) {
			char colon;
			Word word;
			iss >> word.term >> colon >> word.count;						// Read term index and count.

			min_term = std::min(min_term, word.term);
			max_term = std::max(max_term, word.term);
			total_count += word.count;
			doc.push_back(word);
		}
		TOTAL_WORDS += total_count;
		max_words_in_doc = std::max(max_words_in_doc, total_count);
		corpus.push_back(doc);
	}

	// Update some parameters.
	NUM_DOCS = static_cast<int>(corpus.size());
	NUM_VOCABS = max_term - min_term + 1;
	MAX_WORD_IN_DOC = max_words_in_doc;
	MIN_TERM = min_term;
	return true;
}

bool ReadWord2VecEmbeddingMatrix(const char* em_mat_path, std::vector<double>& words,
		int& num_w2v, int& vect_size)
{
	std::ifstream in_file(em_mat_path);
	if (!in_file.is_open())
		return false;

	words.clear();
	std::string line;
	in_file >> num_w2v >> vect_size;
	std::cout << "Num Words:" << num_w2v << std::endl;
	std::cout << "Vect size:" << vect_size << std::endl;
	std::cout << std::endl;

	for (int i = 0; i < num_w2v; ++i) {
		std::string word_name;
		in_file >> word_name;
		float sm = 0.0;
		for (int j = 0; j < vect_size; ++j) {
			float f;
			in_file >> f;
			sm += f * f;
			words.push_back(f);
		}
		const double ln = sqrt(sm);
		for (int j = 0; j < vect_size; ++j)
			words[i * vect_size + j] /= ln;
	}
	return true;
}

bool ReadBasisTerms(const char* basis_terms_path, std::vector<int>& basis_terms, int& num_basis_terms)
{
	std::ifstream in_file(basis_terms_path);
	if (!in_file.is_open())
		return false;

	basis_terms.clear();
	std::string line;
	while (std::getline(in_file, line))
		basis_terms.push_back(std::stoi(line));
	num_basis_terms = static_cast<int>(basis_terms.size());
	return true;
}

inline double Dot(int a, int b)
{
	const int AIDX = a * FBOW_SIZE;
	const int BIDX = b * FBOW_SIZE;
	double res = 0.0;
	for (int i = 0; i < FBOW_SIZE; ++i)
		res += words_vectors[AIDX + i] * words_vectors[BIDX + i];
	return res;
}

inline double GetFuzzyMembership(int basis_idx, int word_idx)
{
	const int A = basis_terms[basis_idx];
	if (A == word_idx)
		return 1.0;

	const double COSINE_SIM = Dot(A, word_idx);
	const int IS_POSITIVE = !!(COSINE_SIM > 0);
	const double RES = IS_POSITIVE * COSINE_SIM;
	return RES;
}

void InitFuzzyCorpus(const Corpus& corpus)
{
	fuzzy_corpus.clear();
	fuzzy_corpus.resize(NUM_DOCS * NUM_BASIS_TERMS);
	for (int d = 0; d < NUM_DOCS; ++d) {
		const Doc& doc = corpus[d];
		for (int i = 0; i < NUM_BASIS_TERMS; ++i) {
			const int FWIDX = FUZZY_WRD_IDX(d, i);
			for (unsigned j = 0; j < doc.size(); ++j) {
				const int WJ = doc[j].term;
				const int CNT = doc[j].count;
				fuzzy_corpus[FWIDX] += GetFuzzyMembership(i, WJ) * CNT;
			}
		}
	}

	// Dump fuzzy representation.
	std::ofstream of("fuzzy-docs.txt");
	for (int d = 0; d < NUM_DOCS; ++d) {
		for (int i = 0; i < NUM_BASIS_TERMS; ++i)
			of << ' ' << fuzzy_corpus[FUZZY_WRD_IDX(d, i)];
		of << std::endl;
	}
}

void PrintCnts(const std::vector<double>& cnt_words, const std::vector<double>& cnt_topics)
{
	std::ofstream cnt_wf("fuzzy-cnt_words.txt");
	cnt_wf << "CNT WRD:    " << NUM_TOPICS << " x " << NUM_BASIS_TERMS << std::endl;
	for (int k = 0; k < NUM_TOPICS; ++k) {
		cnt_wf << "K:" << k << "   ";
		for (int n = 0; n < NUM_BASIS_TERMS; ++n)
			cnt_wf << cnt_words[CNT_WRD_IDX(k, n)] << ' ';
		cnt_wf << std::endl;
	}

	std::ofstream cnt_tf("fuzzy-cnt_topics.txt");
	cnt_tf << "CNT TPC:    " << NUM_DOCS << " x " << NUM_TOPICS << std::endl;
	for (int i = 0; i < NUM_DOCS; ++i) {
		cnt_tf << "D:" << i << "   ";
		for (int k = 0; k < NUM_TOPICS; ++k)
			cnt_tf << cnt_topics[CNT_TPC_IDX(i, k)] << ' ';
		cnt_tf << std::endl;
	}
}

void CountWordsAndTopics(const std::vector<double>& fuzzy_corpus,
		const std::vector<int>& cr_z, std::vector<double>& cnt_w,
		std::vector<double>& cnt_t)
{
	for (unsigned i = 0; i < cnt_words.size(); ++i)
		cnt_words[i] = 0;
	for (unsigned i = 0; i < cnt_topics.size(); ++i)
		cnt_topics[i] = 0;

	for (int d = 0; d < NUM_DOCS; ++d) {
		for (int n = 0; n < NUM_BASIS_TERMS; ++n) {
			const int FWIDX = FUZZY_WRD_IDX(d, n);
			const double F = fuzzy_corpus[FWIDX];
			const int WZ = cr_z[Z_IDX(d, n)];

			const int WIDX = CNT_WRD_IDX(WZ, n);
			cnt_w[WIDX] += F;

			const int TIDX = CNT_TPC_IDX(d, WZ);
			cnt_t[TIDX] += F;
		}
	}

	PrintCnts(cnt_w, cnt_t);
	//exit(100);
}

void SamplePhi()
{
	std::vector<double> smpls(NUM_BASIS_TERMS);
	for (int k = 0; k < NUM_TOPICS; ++k) {
		double sum = 0.0;
		for (int n = 0; n < NUM_BASIS_TERMS; ++n) {
			const int WIDX = CNT_WRD_IDX(k, n);
			const double alpha = cnt_words[WIDX] + ALPHA;
			std::gamma_distribution<double> gamma(alpha, 1);
			smpls[n] = gamma(dev);
			sum += smpls[n];
		}
		for (int n = 0; n < NUM_BASIS_TERMS; ++n)
			phi[PHI_IDX(k, n)] = smpls[n] / sum;
	}
}

void SampleTheta()
{
	std::vector<double> smpls(NUM_TOPICS);
	for (int i = 0; i < NUM_DOCS; ++i) {
		double sum = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k) {
			const int TIDX = CNT_TPC_IDX(i, k);
			const double alpha = cnt_topics[TIDX] + BETA;
			std::gamma_distribution<double> gamma(alpha, 1);
			smpls[k] = gamma(dev);
			sum += smpls[k];
		}
		for (int k = 0; k < NUM_TOPICS; ++k)
			theta[THETA_IDX(i, k)] = smpls[k] / sum;
	}
}

double LogSum(double log_a, double log_b)
{
	if (log_b < log_a)
		std::swap(log_a, log_b);

	double res = log_b + log(1 + exp(log_a - log_b));
	return res;
}

int SelectOption(const std::vector<double>& lg_probs, double sm_lg_probs)
{
	std::uniform_real_distribution<double> unif(0, 1);
	const double U = unif(dev);
	double asum = 0;
	for (unsigned k = 0; k < lg_probs.size(); ++k) {
		asum += exp(lg_probs[k] - sm_lg_probs);
		if (U < asum)
			return static_cast<int>(k);
	}

	return static_cast<int>(lg_probs.size()) - 1;
}

void SampleZ(const std::vector<double>& fuzzy_corpus)
{
	double sm_lg_p = 0.0;
	std::vector<double> lg_p_t(NUM_TOPICS);
	for (int i = 0; i < NUM_DOCS; ++i) {
		for (int n = 0; n < NUM_BASIS_TERMS; ++n) {
			const int ZIDX = Z_IDX(i, n);
			const int OLD_Z = z[ZIDX];

			for (int k = 0; k < NUM_TOPICS; ++k) {
				const int PIDX = PHI_IDX(k, n);
				const double p = phi[PIDX];

				const int TIDX = THETA_IDX(i, k);
				const double t = theta[TIDX];

				lg_p_t[k] = log(p) + log(t);
				if (k == 0)
					sm_lg_p = lg_p_t[0];
				else
					sm_lg_p = LogSum(lg_p_t[k], sm_lg_p);
			}
			const int NEW_Z = SelectOption(lg_p_t, sm_lg_p);
			if (NEW_Z != OLD_Z) {
				z[ZIDX] = NEW_Z;

				const int FWIDX = FUZZY_WRD_IDX(i, n);
				const double F = fuzzy_corpus[FWIDX];
				const int OLD_WIDX = CNT_WRD_IDX(OLD_Z, n);
				const int NEW_WIDX = CNT_WRD_IDX(NEW_Z, n);
				cnt_words[OLD_WIDX] -= F;
				cnt_words[NEW_WIDX] += F;

				const int OLD_TIDX = CNT_TPC_IDX(i, OLD_Z);
				const int NEW_TIDX = CNT_TPC_IDX(i, NEW_Z);
				cnt_topics[OLD_TIDX] -= F;
				cnt_topics[NEW_TIDX] += F;
			}
		}		// foreach word
	}		// foreach doc
}

void DumpPhi(const std::vector<double>& p, int num_smpls)
{
	std::ofstream out_file("fuzzy-phi.txt");
	out_file.precision(7);
	out_file << std::fixed;
	for (int k = 0; k < NUM_TOPICS; ++k) {
		out_file << k << " ->   " << p[PHI_IDX(k, 0)] / num_smpls;
		for (int n = 1; n < NUM_BASIS_TERMS; ++n)
			out_file << ' ' << p[PHI_IDX(k, n)] / num_smpls;
		if (k + 1 < NUM_TOPICS)
			out_file << std::endl;
	}
}

template <typename T>
int GetDocTopic(int d, const std::vector<T>& t)
{
	int mk = 0;
	double mg = t[THETA_IDX(d, 0)];
	for (int k = 1; k < NUM_TOPICS; ++k)
		if (t[THETA_IDX(d, k)] > mg) {
			mk = k;
			mg = t[THETA_IDX(d, k)];
		}
	return mk;
}

void DumpTheta(const std::vector<double>& t, int num_smpls)
{
	std::ofstream out_file("fuzzy-theta.txt");
	for (int d = 0; d < NUM_DOCS; ++d) {
		int doc_topic = GetDocTopic(d, t);
		out_file << std::setw(3) << (d + 1) << " ->  " << doc_topic << "  " << t[THETA_IDX(d, 0)] / num_smpls;
		for (int k = 1; k < NUM_TOPICS; ++k)
			out_file << "   " << t[THETA_IDX(d, k)] / num_smpls;
		if (d + 1 < NUM_DOCS)
			out_file << std::endl;
	}
}

void DumpZ(const std::vector<int>& z, const char* z_path="fuzzy-z.txt")
{
	std::ofstream out_file(z_path);
	for (int i = 0; i < NUM_DOCS; ++i) {
		out_file << i << " -> ";
		for (int n = 0; n < NUM_BASIS_TERMS; ++n) {
			int wz = 0;
			int mx_z = z[SMPL_Z_IDX(i, n, 0)];
			for (int k = 1; k < NUM_TOPICS; ++k)
				if (mx_z < z[SMPL_Z_IDX(i, n, k)]) {
					mx_z = z[SMPL_Z_IDX(i, n, k)];
					wz = k;
				}
			out_file << wz << ' ';
		}
		out_file << std::endl;
	}
}

void DumpZ_2(const std::vector<int>& z, const char* z_path="fuzzy-z2.txt")
{
	std::ofstream out_file(z_path);
	for (int i = 0; i < NUM_DOCS; ++i) {
		out_file << i << "    ";
		for (int n = 0; n < NUM_BASIS_TERMS; ++n) {
			out_file << z[Z_IDX(i, n)] << ' ';
		}
		out_file << std::endl;
	}
}

double CalculateLikelihood(const std::vector<double>& fuzzy_corpus,
		const std::vector<int>& cur_z, /*const std::vector<double>& cur_theta,*/
		const std::vector<double>& cur_phi, bool is_divided_by_words=true)
{
	double llhood = 0.0;
	for (int i = 0; i < NUM_DOCS; ++i) {
		for (int n = 0; n < NUM_BASIS_TERMS; ++n) {
			const int FWIDX = FUZZY_WRD_IDX(i, n);
			const double F = fuzzy_corpus[FWIDX];
			const int ZIDX = Z_IDX(i, n);
			const int WZ = cur_z[ZIDX];
			const int PIDX = PHI_IDX(WZ, n);
			llhood += F * log(cur_phi[PIDX]);
		}
	}
	if (is_divided_by_words)
		return llhood / TOTAL_WORDS;
	return llhood;
}

void SaveSamples()
{
	++num_smpls;
	for (unsigned i = 0; i < phi.size(); ++i)
		smpl_phi[i] += phi[i];
	for (unsigned i = 0; i < theta.size(); ++i)
		smpl_theta[i] += theta[i];
	for (int i = 0; i < NUM_DOCS; ++i) {
		for (int n = 0; n < NUM_BASIS_TERMS; ++n) {
			const int ZIDX = Z_IDX(i, n);
			const int WZ = z[ZIDX];
			const int SZIDX = SMPL_Z_IDX(i, n, WZ);
			++smpl_z[SZIDX];
		}
	}
}

void CalcAccuracy(const std::vector<double>& smpl_t, const char* msg="")
{
	if (NUM_TOPICS > 5)
		return;

	std::vector<std::vector<int>> cluster_count(NUM_TOPICS, std::vector<int>(NUM_TOPICS, 0));

	// Count clusters.
	const int NUM_DOCS_IN_TOPIC = NUM_DOCS / NUM_TOPICS;
	for (int k = 0; k < NUM_TOPICS; ++k)
		for (int d = 0; d < NUM_DOCS_IN_TOPIC; ++d)
			++cluster_count[k][GetDocTopic(k * NUM_DOCS_IN_TOPIC + d, smpl_t)];

	std::vector<int> perm(NUM_TOPICS);
	for (int i = 0; i < NUM_TOPICS; ++i)
		perm[i] = i;

	// Calculate accuracies and find maximum one for report.
	std::vector<double> accs;
	do {
		double sm = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k)
			sm += cluster_count[k][perm[k]];
		const double ACC = sm / static_cast<double>(NUM_DOCS);
		accs.push_back(ACC);
	} while (std::next_permutation(perm.begin(), perm.end()));

	// Print result.
	std::cout << ' ' << msg << " Accuracy: " << *std::max_element(accs.begin(), accs.end()) << "   [ ";
	for (const auto& a : accs)
		std::cout << a << ' ';
	std::cout << ']' << std::endl;
}

void InitializeParameters()
{
#if defined(INIT_RANDOM)
	std::uniform_int_distribution<int> int_unif(0, NUM_TOPICS - 1);
	std::uniform_real_distribution<double> real_unif(0, 1);
#endif

	z.resize(NUM_DOCS * NUM_BASIS_TERMS);
	for (int i = 0; i < NUM_DOCS; ++i)
		for (int n = 0; n < NUM_BASIS_TERMS; ++n)
#if defined(INIT_RANDOM)
			z[Z_IDX(i, n)] = int_unif(dev);
#else
			z[Z_IDX(i, n)] = i / (NUM_DOCS / NUM_TOPICS);
#endif

	phi.resize(NUM_TOPICS * NUM_BASIS_TERMS);
	for (int i = 0; i < NUM_BASIS_TERMS; ++i)
		for (int k = 0; k < NUM_TOPICS; ++k)
#if defined(INIT_RANDOM)
			phi[PHI_IDX(k, i)] = real_unif(dev);
#else
			phi[PHI_IDX(k, i)] = 1.0 / NUM_TOPICS;
#endif

	theta.resize(NUM_DOCS * NUM_TOPICS);
	for (int i = 0; i < NUM_DOCS; ++i)
		for (int k = 0; k < NUM_TOPICS; ++k)
#if defined(INIT_RANDOM)
			theta[THETA_IDX(i, k)] = real_unif(dev);
#else
			theta[THETA_IDX(i, k)] = !!(k == (i / (NUM_DOCS / NUM_TOPICS)));
#endif

#if defined(INIT_RANDOM)
	for (int i = 0; i < NUM_DOCS; ++i) {
		double sm = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k)
			sm += theta[THETA_IDX(i, k)];
		for (int k = 0; k < NUM_TOPICS; ++k)
			theta[THETA_IDX(i, k)] /= sm;
	}
	for (int k = 0; k < NUM_TOPICS; ++k) {
		double sm = 0.0;
		for (int i = 0; i < NUM_BASIS_TERMS; ++i)
			sm += phi[PHI_IDX(k, i)];
		for (int i = 0; i < NUM_BASIS_TERMS; ++i)
			phi[PHI_IDX(k, i)] /= sm;
	}
#endif
}



int main(int argc, char** argv)
{
	// Check number of command line arguments.
	if (argc <= 2) {
		// Print usage.
		std::cout << "Usage:   " << argv[0] << " CORPUS_PATH  W2V_PATH  BASIS_TERMS_PATH  K" << std::endl;
		return 1;
	}

	// Parse command line arguments.
	const char* input_file = argv[1];
	const char* w2v_out_path = argv[2];// "D:\\C++\\word2vec\\dav\\build\\bin\\my_output.txt";
	const char* basis_terms_path = argv[3];
	NUM_TOPICS = atoi(argv[4]);

	// Read corpus.
	Corpus corpus;
	if (!ReadCorpus(input_file, corpus)) {
		std::cout << "Could not read `" << input_file << "' file!" << std::endl;
		return 1;
	}

	if (!ReadWord2VecEmbeddingMatrix(w2v_out_path, words_vectors, NUM_WORD_2_VEC, FBOW_SIZE)) {
		std::cout << "Could not read `" << w2v_out_path << "' word2vec file!" << std::endl;
		return 2;
	}

	if (!ReadBasisTerms(basis_terms_path, basis_terms, NUM_BASIS_TERMS)) {
		std::cout << "Could not read `" << basis_terms_path << "' basis terms file!" << std::endl;
		return 3;
	}

	// Print some statistics.
	std::cout << "Num topics           : " << NUM_TOPICS << std::endl;
	std::cout << "Num vocabularies     : " << NUM_VOCABS << std::endl;
	std::cout << "Corpus size          : " << corpus.size() << std::endl;
	std::cout << "Max num words in doc : " << MAX_WORD_IN_DOC << std::endl;
	std::cout << "Min Term             : " << MIN_TERM << std::endl;
#if defined(INIT_RANDOM)
	std::cout << "Initialized Randomly!" << std::endl;
#else
	std::cout << "Initialized By Clusters!" << std::endl;
#endif

	InitFuzzyCorpus(corpus);
	corpus.clear();					// Clear unused variables.

	// Initialize parameters.
	InitializeParameters();

	cnt_words.resize(NUM_TOPICS * NUM_BASIS_TERMS);
	cnt_topics.resize(NUM_DOCS * NUM_TOPICS);
	CountWordsAndTopics(fuzzy_corpus, z, cnt_words, cnt_topics);

	// Initialize samples.
	smpl_z.resize(NUM_DOCS * NUM_BASIS_TERMS * NUM_TOPICS);
	smpl_phi.resize(NUM_TOPICS * NUM_BASIS_TERMS);
	smpl_theta.resize(NUM_DOCS * NUM_TOPICS);

	// MCMC loop.
	double old_likelihood = -std::numeric_limits<double>::max();
	int thin_cnt = 0;
	for (int itr = 0; itr < TOTAL_MAX_ITERATIONS; ++itr) {
		std::cout << "---------- itr #" << (itr + 1)
			<< " of " << TOTAL_MAX_ITERATIONS
			<< " (" << (itr < BURNIN_ITERATIONS ? "BURNIN" : "MCMC");
		if (itr < BURNIN_ITERATIONS)
			std::cout << ":" << BURNIN_ITERATIONS;
		std::cout << ") ----------" << std::endl;

		// Draw Phi, Theta, and Z samples.
		SamplePhi();
		SampleTheta();
		SampleZ(fuzzy_corpus);

		// Save samples.
		if (itr > BURNIN_ITERATIONS) {
			++thin_cnt;
			if (thin_cnt >= THIN_NUM) {
				thin_cnt = 0;
				SaveSamples();
			}
		}

		// Check convergence.
		const double corpus_likelihood = CalculateLikelihood(fuzzy_corpus, z, /*theta,*/ phi);
		std::cout << "Corpus likelihood : " << corpus_likelihood << std::endl;
		const double diff_likelihood = corpus_likelihood - old_likelihood;
		std::cout << "diff likelihood   : " << diff_likelihood << std::endl;
		std::cout << "samples           : " << num_smpls << " of Min(" << MIN_SAMPLE_SIZE << ')' << std::endl;
		CalcAccuracy(theta, "Current");
		if (itr > BURNIN_ITERATIONS && diff_likelihood < EPSILON && num_smpls >= MIN_SAMPLE_SIZE) {
			std::cout << "********** Converged! **********" << std::endl;
			std::cout << "diff likelihood: " << diff_likelihood << std::endl;
			std::cout << "********************************" << std::endl;
			break;
		}

		old_likelihood = corpus_likelihood;
	}

	// Write topic proportions and word probability estimates for each topic.
	DumpPhi(smpl_phi, num_smpls);
	DumpTheta(smpl_theta, num_smpls);
	DumpZ(smpl_z);
	//PrintCnts(cnt_words, cnt_topics);
	CalcAccuracy(smpl_theta);
	return 0;
}

