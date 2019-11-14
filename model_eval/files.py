from os import chdir, getcwd, listdir, mkdir, path, remove, rmdir, sep

DATA_DIR = 'data' + sep
TMP_DIR = 'tmp' + sep

DATA_SET_FILE = DATA_DIR + 'bangumi.csv'
SKIP_WORD_FILE = DATA_DIR + 'skip_word.txt'
STYLE_TRANS_FILE = DATA_DIR + 'style_trans.txt'

PRE_PROCESSED_FILE = TMP_DIR + '.bangumi.npy'
FAST_TEXT_FILE = TMP_DIR + '.fast.txt'


def get_feature_file(scale):
    return TMP_DIR + '.feature_' + scale + '.npy'


def init():
    full_tmp_dir = getcwd() + sep + TMP_DIR

    if not path.exists(full_tmp_dir):
        mkdir(full_tmp_dir)


def clean():
    cur_dir = getcwd()
    chdir(cur_dir + sep + TMP_DIR)

    for f in listdir(getcwd()):
        remove(f)

    chdir(cur_dir)
    rmdir(TMP_DIR)
