import subprocess


def run_cifar(batch_size, num_examples, filenamelogits='logits.bin', test_batch='test_batch.bin'):
    cmd = 'python3 cifar10_eval.py {} {} --batch_size {} --num_examples {}'.format(
                                     filenamelogits,
                                     test_batch,
                                     batch_size,
                                     num_examples)
    print(cmd)    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    print(''.join(map(chr, out)))


def process_batch(filename):
    import re
    with open(filename, 'r', encoding='utf8') as f:
        # num#den_batch_ind_bsize.bin
        for line in f:
            if line[-1] in ('\n','\r'):
                line = line[:-1]
            lxm = re.split(' |_|\.|\n|\r', line)
            filenamelogits = 'logits_{}_{}.bin'.format(lxm[0], lxm[2])  # logits_1#1_1.bin
            elm = int(lxm[3])
            run_cifar(batch_size=elm,
                      num_examples=elm,
                      filenamelogits=filenamelogits,
                      test_batch=line)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')

    exgroup = parser.add_mutually_exclusive_group(required=True)
    exgroup.add_argument('--list', action='store_true', help='')
    exgroup.add_argument('--one', action='store_true', help='')

    parser.add_argument('--filename', help='List of batch filenames to process. '
                                           'Filename format: num#den_batch_ind_bsize.bin')

    parser.add_argument('--bsize', help='Number of images to process in a batch')
    parser.add_argument('--nexamples', help='Number of examples to run')
    parser.add_argument('--filelogits', help='Filenename of logits')
    parser.add_argument('--tbatch', help='Filename of batch to process')

    args = parser.parse_args()
    if args.one:
        fn = 'logits.bin'
        tb = 'test_batch.bin'

        if args.filelogits is not None:
            fn = args.filelogits
        if args.tbatch is not None:
            tb = args.tbatch

        run_cifar(batch_size=args.bsize,
                  num_examples=args.nexamples,
                  filenamelogits=fn,
                  test_batch=tb)
    elif args.list:
        process_batch(args.filename)
