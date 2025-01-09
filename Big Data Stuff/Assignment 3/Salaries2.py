from mrjob.job import MRJob

class Salaries2(MRJob):
    def mapper(self, _, line):
        (name,jobTitle,agencyID,agency,hireDate,annualSalary,grossPay) = line.split('\t')
        if line.startswith("name"):
            return

        try:
            (name, jobTitle, agencyID, agency, hireDate, annualSalary, grossPay) = line.split('\t')
            annualSalary = float(annualSalary.replace(",", ""))
            if annualSalary >= 100000.00:
                yield 'High', 1
            elif 50000.00 <= annualSalary < 100000.00:
                yield 'Medium', 1
            else:
                yield 'Low', 1
        except ValueError:
            # In case of a value error due to incorrect format, skip the line
            return

    def combiner(self, salaryCategory, counts):
        yield salaryCategory, sum(counts)

    def reducer(self, salaryCategory, counts):
        yield salaryCategory, sum(counts)

if __name__ == '__main__':
    Salaries2.run()
