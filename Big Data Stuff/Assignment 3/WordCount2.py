from mrjob.job import MRJob
import re

WORD_RE = re.compile(r"[\w']+")


class WordCount2(MRJob):

    def mapper(self, _, line):
        for word in WORD_RE.findall(line):
            first_letter = word[0].lower()
            if 'a' <= first_letter <= 'n':
                yield 'a-n', 1
            else:
                yield 'others', 1

    def combiner(self, category, counts):
        yield category, sum(counts)

    def reducer(self, category, counts):
        yield category, sum(counts)


if __name__ == '__main__':
    WordCount2.run()
