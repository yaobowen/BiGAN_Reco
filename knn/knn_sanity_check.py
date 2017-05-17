from easy_classifier_evaluation_toolbox import knnRecommender
import numpy as np

def printMenu():
    print "What do you want to do?"
    print "[1] Simple 0/1 training points."
    print "[2] Simple graph."

def simple01():
    X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_train = np.array([0,1,0,1])
    X_test = np.array([[0,-1],[0,2]])
    y_test = np.array([0,1])

    test_rec = knnRecommender(X_train, y_train)
    recommend = test_rec.recommend(X_test, k=2)
    print recommend
    precision = test_rec.get_precision(recommend, y_test)
    print precision

# Main function:
printMenu()
choice = int(raw_input("Your choice (type the number, type 123 to quit): "))

while choice != 123:
    if choice == 1:
        simple01()
    elif choice == 2:
        print "Not yet implemented."
    else:
        print "Invalid input."
    printMenu()
    choice = int(raw_input("Your choice (type the number, type 123 to quit): "))
    continue

print "Bye-bye!"
