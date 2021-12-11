import os

import numpy as np



accs_gibbs_lda = [ l.strip() for l in open(os.path.join('Datasets', 'accs_gibbs_lda.txt'), 'r') ]
accs_fuzzy_lda = [ l.strip() for l in open(os.path.join('Datasets', 'accs_fuzzy_lda.txt'), 'r') ]

def FormatPoints(accs, is_max):
    s = []
    old_x = 0
    i = 0
    for l in accs:
        st = l.index(':')
        x = int(l[0:st])
        y = sorted(eval(l[st + 1 :]), reverse=True)
        if is_max:
            y = np.array(y) * 100.0
            y = y.max()
        else:
            y = np.array(y[0:10]) * 100.0
            y = y.mean()

        n_line = ''
        if i == 3:
            i = 0
            n_line = '\n'
        if old_x > x:
            i = 0
            n_line += '\n' if n_line else '\n\n'
        s.append('%s(%d, %2.2f)' % (n_line, x, y))
        old_x = x
        i += 1
    print(' '.join(s))


def PrintAccs(is_max):
    print('********** Gibbs LDA ' + ('(Max) ' if is_max else '') + '**********')
    FormatPoints(accs_gibbs_lda, is_max)
    print('')
    print('********** Fuzzy LDA ' + ('(Max) ' if is_max else '') + '**********')
    FormatPoints(accs_fuzzy_lda, is_max)
    print('')



if __name__ == '__main__':
    PrintAccs(False)
    print('')
    PrintAccs(True)

