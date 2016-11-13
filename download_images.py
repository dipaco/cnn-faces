from urllib.request import urlretrieve
from urllib.error import HTTPError
import sys
# import os


def download_images(urlstxt, pattern):
    with open(urlstxt, 'r', encoding='utf8') as file:
        t = f = 0
        log = []

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        for i, url in enumerate(file):
            if url[-1] in ('\n', '\r'):
                url = url[:-1]
            filename = '{}_{}.jpg'.format(pattern, i)
            try:
                filepath, _ = urlretrieve(url=url, filename=filename, reporthook=_progress)
                print()
                t += 1
            except HTTPError as e:
                print(e)
                log.append('Error: {} at file:{} with url:{}\n'.format(e, filename, url))
                f += 1
            except Exception as ex:
                print(ex)
                log.append('Error: {} at file:{} with url:{}\n'.format(ex, filename, url))
                f += 1
        print('Successfully downloaded {}'.format(t))
        print('Download errors         {}'.format(f))
        with open('log.txt', 'w', encoding='utf8') as l:
            l.writelines(log)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        download_images(sys.argv[1], sys.argv[2])
