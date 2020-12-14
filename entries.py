import os
from time import gmtime,strftime, localtime


class Queue:
    def __init__(self):
        self.items = []
        self.frontIdx = 0

    def __compress(self):
        newlst = []
        for i in range(self.frontIdx, len(self.items)):
            newlst.append(self.items[i])

        self.items = newlst
        self.frontIdx = 0

    def dequeue(self):
        if self.isEmpty():
            raise RuntimeError("Attempt to dequeue an empty queue")

        item = self.items[-1]

        self.items.pop(-1)

        return item

    def enqueue(self, item):
        self.items.insert(0,item)

    def front(self):
        if self.isEmpty():
            raise RuntimeError("Attempt to access front of empty queue")

        return self.items[self.frontIdx]

    def isEmpty(self):
        return self.frontIdx == len(self.items)

    def __repr__(self):
        print(str(self.items))

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for item in self.items:
            yield item

def close_ex(str1):
   # print(str1)
   # print(str1, "HAHAHAA")
    return str((str1.split(","))[4])

def close_str(lst1, curr):
    print(lst1)
    n_str = curr.split(",")
    r_str = "".join(n_str[0].split("-"))
    r_str = r_str + ","+ close_ex(lst1[2]) +","+close_ex(lst1[1])+","+close_ex(lst1[0])
    r_str = r_str + "," + str(n_str[1]) + "," + str(n_str[2]) + "," + str(n_str[3]) + "," + str((n_str[5])[:-1]) + "," + str(n_str[4]+"\n")
    return r_str

#print(close_str([ "2018-07-23,106.3000,108.1400,106.1300,107.9700,29706955", "2018-07-24,108.5700,108.8200,107.2600,107.6600,26316619","2018-07-25,107.9550,111.1500,107.6000,110.8300,30567644"], "2018-07-20,108.0800,108.2000,106.0800,106.2700,56038827"))

def lookup(stock):

    cs_name =  "CSVs/" + stock+"1"+".csv"

    os.system(('curl -L "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+stock+'&apikey=demo&datatype=csv&outputsize=full&apikey=2QWQ7S1MCLNSVFA5" -o '+ cs_name))
    l_close = Queue()
    header1 = False

    csv_name = "CSVs/" + stock+"-"+str((strftime("%Y-%m-%d-%H:%M" , localtime())))+".csv"
    file1 = open(csv_name, "w")
    lines =[]
    with open(cs_name) as f:
        for line in f:
            if header1 == False:
                file1.write("Date,Close-1,Close-2,Close-3,Open-1,High-1,Low-1,Volume-1,Close\n")
                header1 =True
                continue
            if len(l_close.items) <3:
                l_close.enqueue(line)
                print("f")
                continue

            elif len(l_close.items) ==3:

                temp = l_close.dequeue()
                l_close.enqueue(line)

                lines.insert(0, ( close_str(l_close.items,temp)))

    for l in lines:
        file1.write(l)


    file1.close()
    os.system("rm " + cs_name)

    return csv_name

def main():
    lookup("TSLA")

if __name__ == "__main__":
    main()