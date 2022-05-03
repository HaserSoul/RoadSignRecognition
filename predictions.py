import csv

def get_class_name(class_nr):
    with open('labels.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')

        mydict = dict((rows[0],rows[1]) for rows in reader)

    return str(mydict[class_nr])

if __name__ == "__main__":
    print(get_class_name('70'))