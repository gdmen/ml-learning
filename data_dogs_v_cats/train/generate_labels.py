with open("files.txt") as f:
    with open("labels.csv", "w") as of:
        of.write("id,label\n")
        for line in f.readlines():
            of.write("%s,%d\n" % (line.strip(), 0 if "cat" in line else 1))
