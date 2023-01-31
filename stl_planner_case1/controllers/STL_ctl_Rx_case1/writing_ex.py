import csv  

header = ['name', 'area', 'country_code2', 'country_code3']
data = ['Afghanistan', 652090, 'AF', 'AFG']

#with open('distances_to_obs.csv', 'w', encoding='UTF8') as f:
#    writer = csv.writer(f)
#
#    # write the header
#    writer.writerow(header)
#
#    # write the data
#    writer.writerow(data)

# open the file in the write mode
f = open('distances_to_obs.csv', 'w', encoding='UTF8')

# create the csv writer
writer = csv.writer(f)

for i in range(5):
    print("number i: ", i)
    #writer.writerow([i, i+1, i+2])
    writer.writerow([i])




## write the header
#writer.writerow(header)
## write the data
#writer.writerow(data)

# close the file
f.close()
