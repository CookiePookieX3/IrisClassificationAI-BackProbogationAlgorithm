def stringToNums(string):
    if string == "Iris-setosa\n":
        return "1.0 0.0 0.0"
    if string == "Iris-versicolor\n": 
        return "0.0 1.0 0.0"
    if string == "Iris-virginica\n" or string == "Iris-virginica":
        return "0.0 0.0 1.0"

f = open("/Users/nikitaackevic/Desktop/cppstuff/irisClassificationAI/DataSet/IrisDataSet.txt")
with open("/Users/nikitaackevic/Desktop/cppstuff/irisClassificationAI/DataSet/IrisDataSetParsed.txt", "w") as fout:
    for i in range(150):
        _,a,b,c,d,name = f.readline().split(',')
        fout.write(f"{a} {b} {c} {d} {stringToNums(name)}\n")