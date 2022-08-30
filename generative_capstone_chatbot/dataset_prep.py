dataset = open('dialogs.txt', 'r')
pairs = []
for line in dataset:
    lst = line.split("\t")
    lst[1] = lst[1].rstrip()
    pairs.append(lst)


