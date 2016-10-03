from random import randint

__author__ = 'madrugado'
import sys
import pandas



def zeros(test):
    with open("zeros.out", "wt") as f, open(test) as t:
        count = 0
        t.readline()
        f.write("id,label\n")
        for _ in t:
            f.write(str(count) + ",0\n")
            count += 1


def random_out(test):
    with open("random.out", "wt") as f, open(test) as t:
        count = 0
        f.write("id,label\n")
        t.readline()
        for _ in t:
            f.write(str(count) + "," + str(randint(0, 1)) + "\n")
            count += 1


def sample(data, test):
    from sklearn.linear_model import LinearRegression

    regr = LinearRegression()

    df = pandas.read_csv(data)
    X = df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].as_matrix()
    y = df['label'].as_matrix()
    regr.fit(X, y)
    tf = pandas.read_csv(test)
    X_test = tf[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].as_matrix()
    ans = [0 if x < 0.3 else 1 for x in regr.predict(X_test)]
    with open("linear.out", "wt") as f:
        count = 0
        f.write("id,label\n")
        for a in ans:
            f.write(str(count) + "," + str(a) + "\n")
            count += 1


def knn(data, test):
    from sklearn.neighbors import KNeighborsClassifier

    regr = KNeighborsClassifier()

    df = pandas.read_csv(data)
    X = df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].as_matrix()
    y = df['label'].as_matrix()
    regr.fit(X, y)
    tf = pandas.read_csv(test)
    X_test = tf[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].as_matrix()
    ans = regr.predict(X_test)
    with open("knn.out", "wt") as f:
        count = 0
        f.write("id,label\n")
        for a in ans:
            f.write(str(count) + "," + str(a) + "\n")
            count += 1


def solution(test):
    df = pandas.read_csv(test)
    ans = df['label'].as_matrix()
    with open("solution.out", "wt") as f:
        count = 0
        f.write("id,label\n")
        for a in ans:
            f.write(str(count) + "," + str(a) + "\n")
            count += 1

if __name__ == "__main__":
    zeros(sys.argv[2])
    random_out(sys.argv[2])
    knn(sys.argv[1], sys.argv[2])
    solution(sys.argv[2])