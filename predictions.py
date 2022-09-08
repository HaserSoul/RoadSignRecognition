import csv

def get_class_name(class_nr):
    with open('labels.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')

        mydict = dict((rows[0],rows[1]) for rows in reader)

    return str(mydict[class_nr])

red_blue_signs_nrs = [str(i) for i in range(25, 77) if i != 58]

yellow_orange_signs_nrs = [str(i) for i in range(0, 25)]+['58']

if __name__ == "__main__":
    print(red_blue_signs_nrs)