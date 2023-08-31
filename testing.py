import csv

myCsvRow = 'S1;REST1;1000;250;-820.58;59.52'
myCsvRow = myCsvRow.split(';')
with open('myCSV.csv','a',newline='') as fd:
    writer = csv.writer(fd, delimiter=';')
    writer.writerow(myCsvRow)
