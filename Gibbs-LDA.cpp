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

#define Z_IDX(D, N)				((D * NUM_VOCABS) + N)
#define SMPL_Z_IDX(D, N, K)		((D * NUM_VOCABS * NUM_TOPICS) + (N * NUM_TOPICS) + K)
#define THETA_IDX(D, K)  		((D * NUM_TOPICS) + K)
#define CNT_TPC_IDX(D, K)		((D * NUM_TOPICS) + K)
#define PHI_IDX(K, N)    		((K * NUM_VOCABS) + N)
#define CNT_WRD_IDX(K, N)		((K * NUM_VOCABS) + N)



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
std::random_device dev;

std::vector<int> z;					/// [D][V]
std::vector<int> cnt_words;			/// [K][V]
std::vector<int> cnt_topics;		/// [D][K]
std::vector<double> phi;			/// [K][V]
std::vector<double> theta;			/// [D][K]

/// Samples
int num_smpls = 0;
std::vector<int> smpl_z;
std::vector<double> smpl_phi;
std::vector<double> smpl_theta;



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

void CountWordsAndTopics(const Corpus& corpus, const std::vector<int>& cr_z,
		std::vector<int>& cnt_w, std::vector<int>& cnt_t)
{
	for (unsigned i = 0; i < cnt_w.size(); ++i)
		cnt_words[i] = 0;
	for (unsigned i = 0; i < cnt_t.size(); ++i)
		cnt_topics[i] = 0;

	for (unsigned i = 0; i < corpus.size(); ++i) {
		const Doc& d = corpus[i];
		for (unsigned n = 0; n < d.size(); ++n) {
			const int W = d[n].term - MIN_TERM;
			const int CNT = d[n].count;
			const int WZ = cr_z[Z_IDX(i, W)];

			const int WIDX = CNT_WRD_IDX(WZ, W);
			cnt_w[WIDX] += CNT;

			const int TIDX = CNT_TPC_IDX(i, WZ);
			cnt_t[TIDX] += CNT;
		}
	}
}

void SamplePhi()
{
	std::vector<double> smpls(NUM_VOCABS);
	for (int k = 0; k < NUM_TOPICS; ++k) {
		double sum = 0.0;
		for (int n = 0; n < NUM_VOCABS; ++n) {
			const int WIDX = CNT_WRD_IDX(k, n);
			const double alpha = cnt_words[WIDX] + ALPHA;
			std::gamma_distribution<double> gamma(alpha, 1);
			smpls[n] = gamma(dev);
			sum += smpls[n];
		}
		for (int n = 0; n < NUM_VOCABS; ++n)
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

void SampleZ(const Corpus& corpus)
{
	double sm_lg_p = 0.0;
	std::vector<double> lg_p_t(NUM_TOPICS);
	for (unsigned i = 0; i < corpus.size(); ++i) {
		const Doc& d = corpus[i];
		for (unsigned n = 0; n < d.size(); ++n) {
			const int W = d[n].term - MIN_TERM;
			const int ZIDX = Z_IDX(i, W);
			const int OLD_Z = z[ZIDX];

			for (int k = 0; k < NUM_TOPICS; ++k) {
				const int PIDX = PHI_IDX(k, W);
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

				const int CNT = d[n].count;
				const int OLD_WIDX = CNT_WRD_IDX(OLD_Z, W);
				const int NEW_WIDX = CNT_WRD_IDX(NEW_Z, W);
				cnt_words[OLD_WIDX] -= CNT;
				cnt_words[NEW_WIDX] += CNT;

				const int OLD_TIDX = CNT_TPC_IDX(i, OLD_Z);
				const int NEW_TIDX = CNT_TPC_IDX(i, NEW_Z);
				cnt_topics[OLD_TIDX] -= CNT;
				cnt_topics[NEW_TIDX] += CNT;
			}
		}		// foreach word
	}		// foreach doc
}

void DumpPhi(const std::vector<double>& p, int num_smpls)
{
	std::ofstream out_file("phi.txt");
	out_file << "Min term: " << MIN_TERM << std::endl;
	out_file.precision(7);
	out_file << std::fixed;
	for (int k = 0; k < NUM_TOPICS; ++k) {
		out_file << k << " ->   " << p[PHI_IDX(k, 0)] / num_smpls;
		for (int n = 1; n < NUM_VOCABS; ++n)
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
	std::ofstream out_file("theta.txt");
	for (int d = 0; d < NUM_DOCS; ++d) {
		int doc_topic = GetDocTopic(d, t);
		out_file << std::setw(3) << (d + 1) << " ->  " << doc_topic << "  " << t[THETA_IDX(d, 0)] / num_smpls;
		for (int k = 1; k < NUM_TOPICS; ++k)
			out_file << "   " << t[THETA_IDX(d, k)] / num_smpls;
		if (d + 1 < NUM_DOCS)
			out_file << std::endl;
	}
}

void DumpZ(const std::vector<int>& z, const char* z_path="z.txt")
{
	std::ofstream out_file(z_path);
	for (int i = 0; i < NUM_DOCS; ++i) {
		out_file << i << " -> ";
		for (int n = 0; n < NUM_VOCABS; ++n) {
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

void DumpZ_2(const std::vector<int>& z, const char* z_path="z2.txt")
{
	std::ofstream out_file(z_path);
	for (int i = 0; i < NUM_DOCS; ++i) {
		out_file << i << "    ";
		for (int n = 0; n < NUM_VOCABS; ++n) {
			out_file << z[Z_IDX(i, n)] << ' ';
		}
		out_file << std::endl;
	}
}

void PrintCnts(const std::vector<int>& cnt_words, const std::vector<int>& cnt_topics)
{
	std::cout << "CNT WRD:" << std::endl;
	for (int k = 0; k < NUM_TOPICS; ++k) {
		std::cout << "K:" << k << "   ";
		for (int n = 0; n < NUM_VOCABS; ++n)
			std::cout << cnt_words[CNT_WRD_IDX(k, n)] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "CNT TPC:" << std::endl;
	for (int i = 0; i < NUM_DOCS; ++i) {
		std::cout << "D:" << i << "   ";
		for (int k = 0; k < NUM_TOPICS; ++k)
			std::cout << cnt_topics[CNT_TPC_IDX(i, k)] << ' ';
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

double CalculateLikelihood(const Corpus& corpus, const std::vector<int>& cur_z,
		/*const std::vector<double>& cur_theta,*/ const std::vector<double>& cur_phi,
		bool is_divided_by_words=true)
{
	double llhood = 0.0;
	int total_words = 0;
	for (unsigned i = 0; i < corpus.size(); ++i) {
		const Doc& d = corpus[i];
		for (unsigned n = 0; n < d.size(); ++n) {
			const int W = d[n].term - MIN_TERM;
			const int CNT = d[n].count;
			const int ZIDX = Z_IDX(i, W);
			const int WZ = cur_z[ZIDX];
			const int PIDX = PHI_IDX(WZ, W);
			llhood += CNT * log(cur_phi[PIDX]);
			total_words += CNT;
		}
	}
	if (is_divided_by_words)
		return llhood / total_words;
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
		for (int n = 0; n < NUM_VOCABS; ++n) {
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

void CheckCounts(const Corpus& corpus)
{
	std::vector<int> tmp_wrd(NUM_TOPICS * NUM_VOCABS);
	std::vector<int> tmp_tpc(NUM_DOCS * NUM_TOPICS);
	CountWordsAndTopics(corpus, z, tmp_wrd, tmp_tpc);

	for (unsigned i = 0; i < tmp_wrd.size(); ++i) {
		if (tmp_wrd[i] != cnt_words[i]) {
			std::cout<<"tmp_wrd[i:"<<i<<"]="<<tmp_wrd[i]<<"  cnt_words[i:"<<i<<"]="<<cnt_words[i]<<std::endl;
			exit(100);
		}
	}

	for (unsigned i = 0; i < tmp_tpc.size(); ++i) {
		if (tmp_tpc[i] != cnt_topics[i]) {
			std::cout<<"tmp_tpc[i:"<<i<<"]="<<tmp_tpc[i]<<"  cnt_topics[i:"<<i<<"]="<<cnt_topics[i]<<std::endl;
			exit(100);
		}
	}
}

void InitializeParameters()
{
#if defined(INIT_RANDOM)
	std::uniform_int_distribution<int> int_unif(0, NUM_TOPICS - 1);
	std::uniform_real_distribution<double> real_unif(0, 1);
#endif

	z.resize(NUM_DOCS * NUM_VOCABS);
	for (int i = 0; i < NUM_DOCS; ++i)
		for (int n = 0; n < NUM_VOCABS; ++n)
#if defined(INIT_RANDOM)
			z[Z_IDX(i, n)] = int_unif(dev);
#else
			z[Z_IDX(i, n)] = i / (NUM_DOCS / NUM_TOPICS);
#endif

	phi.resize(NUM_TOPICS * NUM_VOCABS);
	for (int i = 0; i < NUM_VOCABS; ++i)
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

	// Normalize matrices.
	for (int i = 0; i < NUM_DOCS; ++i) {
		double sm = 0.0;
		for (int k = 0; k < NUM_TOPICS; ++k)
			sm += theta[THETA_IDX(i, k)];
		for (int k = 0; k < NUM_TOPICS; ++k)
			theta[THETA_IDX(i, k)] /= sm;
	}
	for (int k = 0; k < NUM_TOPICS; ++k) {
		double sm = 0.0;
		for (int i = 0; i < NUM_VOCABS; ++i)
			sm += phi[PHI_IDX(k, i)];
		for (int i = 0; i < NUM_VOCABS; ++i)
			phi[PHI_IDX(k, i)] /= sm;
	}
}



int main(int argc, char** argv)
{
	// Check number of command line arguments.
	if (argc <= 2) {
		// Print usage.
		std::cout << "Usage:   " << argv[0] << " CORPUS_PATH   K" << std::endl;
		return 1;
	}

	// Parse command line arguments.
	const char* input_file = argv[1];
	NUM_TOPICS = atoi(argv[2]);

	// Read corpus.
	Corpus corpus;
	if (!ReadCorpus(input_file, corpus)) {
		std::cout << "Could not read `" << input_file << "' file!" << std::endl;
		return 1;
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

	// Initialize parameters.
	InitializeParameters();

	cnt_words.resize(NUM_TOPICS * NUM_VOCABS);
	cnt_topics.resize(NUM_DOCS * NUM_TOPICS);
	CountWordsAndTopics(corpus, z, cnt_words, cnt_topics);

	// Initialize samples.
	smpl_z.resize(NUM_DOCS * NUM_VOCABS * NUM_TOPICS);
	smpl_phi.resize(NUM_TOPICS * NUM_VOCABS);
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
		SampleZ(corpus);

		// Save samples.
		if (itr > BURNIN_ITERATIONS) {
			++thin_cnt;
			if (thin_cnt >= THIN_NUM) {
				thin_cnt = 0;
				SaveSamples();
			}
		}

		// Check convergence.
		const double corpus_likelihood = CalculateLikelihood(corpus, z, /*theta,*/ phi);
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

