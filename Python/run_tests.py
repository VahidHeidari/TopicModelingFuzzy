import os
import subprocess
import time



FUZZY_CORPUS_PATH = os.path.join('Datasets', 'fuzzy-corpus.txt')
FUZZY_CORPUS = [ l.strip() for l in open(FUZZY_CORPUS_PATH, 'r') ]

NUM_TOPICS = 2
NUM_TEST_RUNS = 20
NUM_DOCS = [ 100, 200, 300, 400, 500, 600, ]

IS_VERBOSE = True
CYGWIN_PATH = os.path.join('C:\\', 'cygwin64', 'bin')



def Log(msg, is_verbose=IS_VERBOSE):
    if not IS_VERBOSE:
        return

    time_str = time.ctime(time.time())
    out_msg = time_str + ' ' + str(msg)
    print(out_msg)
    log_file_date = time.strftime('%Y%m%d')
    with open('log-py-{}.txt'.format(log_file_date), 'a') as f:
        f.write(out_msg + '\n')
        f.flush()


def AddCygwinPath():
    if os.name != 'nt':
        return
    env_path = os.environ['PATH']
    paths = [ p.lower() for p in env_path.split(';') ]
    if CYGWIN_PATH not in paths:
        Log('Add cygwin path . . .')
        new_env = CYGWIN_PATH + ';' + env_path
        os.environ['PATH'] = new_env
    else:
        Log('cygwin path exists!')


def RemoveCygwinPath():
    if os.name != 'nt':
        return
    env_path = os.environ['PATH']
    paths = [ p.lower() for p in env_path.split(';') ]
    if CYGWIN_PATH in paths:
        Log('remove cygwin path . . .')
        paths.remove(CYGWIN_PATH)
        new_env = ''
        for p in paths:
            new_env += p + ';'
        os.environ['PATH'] = new_env


def RunCommand(cmd):
    AddCygwinPath()
    Log('START : ' + time.ctime(time.time()))
    cmd = [str(c) for c in cmd]
    Log(' '.join(cmd))

    prc = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd='.')
    (output, err) = prc.communicate()
    exit_code = prc.wait()

    if exit_code != 0:
        Log('  Exit Code : %d' % exit_code)
        Log('  Output    : `%s\'' % output)
        Log('  Error     : `%s\'' % err)
    Log('END   : ' + time.ctime(time.time()))
    RemoveCygwinPath()
    return exit_code, output, err


def GetOutPath(num_docs):
    out_path = os.path.join('Datasets', '{}-corpus.txt'.format(num_docs))
    return out_path


def MakeCorpus(num_docs, corpus, num_topics):
    docs_in_topic = len(corpus) / num_topics
    sub_docs_in_topics = min(docs_in_topic, num_docs / num_topics)
    out_path = GetOutPath(num_docs)
    Log(' making  ->  %s' % out_path)
    Log('--------------------------------------')
    Log('  Num docs              : %d' % num_docs)
    Log('  Num total docs        : %d' % len(corpus))
    Log('  Num topics            : %d' % num_topics)
    Log('')
    Log('  Num docs in topics    : %d' % docs_in_topic)
    Log('  Num sub docs in topic : %d\n' % sub_docs_in_topics)

    sub_corpus = []
    for k in range(num_topics):
        st = k * docs_in_topic
        sub_corpus += corpus[st : st + sub_docs_in_topics]
    with open(out_path, 'w') as f:
        f.write('\n'.join(sub_corpus))


def GetAccFromLine(acc_line):
    st = acc_line.index(':') + 1
    ed = acc_line.index('[') - 1
    acc = float(acc_line[st : ed])
    return acc


def WriteAccsToFile(out_name, num_docs, accs):
    acc_path = os.path.join('Datasets', 'accs_{}.txt'.format(out_name))
    with open(acc_path, 'a') as f:
        f.write('{}: {}\n'.format(num_docs, str(accs)))


def RunGibbsLDA(num_docs, num_topics):
    Log('Running Gibbs LDA for #%d times of corpus with #%d docs . . .' % (NUM_TEST_RUNS, num_docs))
    LDA_EXE = os.path.join('bin', 'gibbs_lda.exe')
    NUM_TOPICS_STR = str(num_topics)
    accs = []
    for i in range(NUM_TEST_RUNS):
        TEST_CORPUS_PATH = os.path.join(BASE_PATH, GetOutPath(num_docs))
        cmd = [ LDA_EXE, TEST_CORPUS_PATH, NUM_TOPICS_STR ]
        exit_code, output, err_out = RunCommand(cmd)

        Log('``%s\'\'' % output.split('\n')[-2])
        accs.append(GetAccFromLine(output.split('\n')[-2]))
        time.sleep(1)
    WriteAccsToFile('gibbs_lda', num_docs, accs)


def RunFuzzyLDA(num_docs, num_topics):
    Log('Running Fuzzy LDA for #%d times of corpus with #%d docs . . .' % (NUM_TEST_RUNS, num_docs))

    LDA_EXE     = os.path.join('bin', 'fuzzy_lda')
    BASIS_TERMS = os.path.join('Datasets', 'top_basis_terms.txt')

    EMBEDDING_PATH = os.path.join('Datasets', 'my_output.txt')

    NUM_TOPICS_STR = str(num_topics)
    accs = []
    for i in range(NUM_TEST_RUNS):
        TEST_CORPUS_PATH = os.path.join(BASE_PATH, GetOutPath(num_docs))
        cmd = [ LDA_EXE, TEST_CORPUS_PATH, EMBEDDING_PATH, BASIS_TERMS, NUM_TOPICS_STR ]
        exit_code, output, err_out = RunCommand(cmd)

        Log('``%s\'\'' % output.split('\n')[-2])
        accs.append(GetAccFromLine(output.split('\n')[-2]))
        time.sleep(1)
    WriteAccsToFile('fuzzy_lda', num_docs, accs)



if __name__ == '__main__':
    Log('\n\n')
    Log('Input corpus path  : %s' % FUZZY_CORPUS_PATH)
    Log('Corpus exists      : {}'.format(os.path.isfile(FUZZY_CORPUS_PATH)))
    Log('Num docs in corpus : %d\n' % len(FUZZY_CORPUS))

    if not os.path.isdir('Datasets'):
        Log('Makding `Datasets\' directory . . .')
        os.makedirs('Datasets')

    for num_docs in NUM_DOCS:
        MakeCorpus(num_docs, FUZZY_CORPUS, NUM_TOPICS)
        RunGibbsLDA(num_docs, NUM_TOPICS)
        RunFuzzyLDA(num_docs, NUM_TOPICS)

