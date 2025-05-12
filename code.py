firstname=input("Enter first Name: ")
lastname=input("Enter last Name: ")

name=firstname+" "+lastname
print(name)#this prints

x=2.55
y=3*x**3-5*x**2+6
print(f"y={y:}")

print(5//2)

string=input("Enter a String:")
n=len(string)//2
print(string[n-1],string[n],string[n+1])

str1="PyTHon"

uppercase=""
lowercase=""
n=len(str1)
for i in range (n):
    if str1[i].islower():
        lowercase+=str1[i]
    else:
        uppercase+=str1[i]

print(f"{lowercase}{uppercase}")


string="Hi guys ^8^*&%^ zchiu"
letter=digit=special=space=0
for i in range (len(string)):
    if string[i].isalpha():
        letter+=1
    elif string[i].isnumeric():
        digit+=1
    elif string[i].isspace():
        space+=1
    else:
        special+=1
print(letter,digit,space,special)


string=input()

frequency={}

for char in string:
    if char in frequency:
        frequency[char]+=1
    else:
        frequency[char]=1
print(frequency)


string="Mar always stood first in class. Mary now works at Google"
substring="Mary"

lastposition=string.rfind(substring)
print(lastposition)


str="Hi8374"
nume=""
for char in str:

    if char.isdigit() :
        nume+=char

print(nume)
        




string = "Mary @always &stood first in class"
ans = ""

for char in string:
    if not char.isalnum() and char != ' ':
        ans += "#"
    else:
        ans += char
print(ans)


str="Hi Guys Hello"
str1=""
for char in str:
    if char.isspace():
        str1+="-"
    else:
        str1+=char
print(str1)


import random


my_list=[1,2,4,6,4,6,4]
val=random.choice(my_list)
print(my_list,val)

my_list=[1,2,4,5,56,6,6,(6,4),6,4]
count=0
for i in my_list:
    if isinstance(i,tuple):
        break
    else:
        count+=1
print(count)

countries={"INDIA":"INR","RUSSIA":"RUS"}
print(countries.get("RUSSIA"))

my_tuple=(2,4,6,8,9)
rev_tup=my_tuple[::-1]
print(rev_tup)


import numpy as np

array=np.zeros((3,3))
print(array)

import numpy as np

arr=np.array([[1,2,3],[2,3,4],[6,8,4]])
row=[1,2,3]
ans=np.any(np.all(arr==row,axis=1))
print(ans)



ls=list((1,3,4,6,7,1))
ls.sort()
print(ls)

import numpy as np

print(np.ones((2,3)))

first_names = np.array(["abhisek", "Shelley", "Lanell", "Genesis", "Margery"])
last_names = np.array(["Battle", "Brien", "Plotner", "Stahl", "Woolum"])

sorted_index=np.lexsort((first_names,last_names))
print(sorted_index)

import numpy as np
arr1=np.array([10,20,30])
arr2=np.array([40,50,60])

res=np.column_stack((arr1,arr2))
print(res)

import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
column_sums = np.sum(arr, axis=0)                #coln for row axis=1
print(f"Sum of columns: {column_sums}")

import numpy as np
arr = np.array([1, 2, 3, 4, 5])

avg=np.average(arr)
print(avg)
var=np.var(arr)
sd=np.std(arr)
print(var,sd)

import numpy as np

char_array=np.array(["apple",'banana'])

space=[]
for i in char_array:
    space.append(" ".join(i))
print(space)

new_arr=np.array(["    ".join(old) for old in char_array])
print(new_arr)

import numpy as np
matrix = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
sorted=np.sort(matrix)
print(sorted)


import numpy as np

d=dict()

d[1] = 123
d[2] = 345

for i in d:
    print("%d %d"%(i,d[i]))


a=int(input())
print(a)

c=20
while(c!=10):
    f=(9/5)*c+32
    print(f"Celcius:{c},farhenheit:{f}")
    c-=1


input=[13,12,11,13,14,13,7,7,13,14,12]
freq={}
for i in input:
    if i in freq:
        freq[i]+=1
    else:
        freq[i]=1



print(freq)

input=[13,12,11,13,14,13,7,7,13,14,12]

hash=[0]*100
for i in range(len(input)):
    hash[input[i]]+=1

myset=set(input)
print(myset)
for i in myset:
    print(i,hash[i])


def find_prime(num):
    count=0
    for i in range(2,num-1):
        if num%i==0:
            count=1
            break
    
    if count==0:
        print("Prime")
    else:
        print("Not Prime")

n=int(input("Enter A no:"))
find_prime(n)


def fact(num):
    fact=1
    for i in range (1,num+1):
        fact*=i
    print(fact)

fact(5)

def factrec(num):
    if (num==0 or num==1):
        return 1
    else:
        return num*factrec(num-1)
    
ans=factrec(5)
print(ans)

def iseven(num):
    if num%2==0:
        return 1
    else:
        return 0
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

even=list(filter(iseven,numbers))
print(even)


dataset=[
 (34587, "Learning Python, Mark Lutz", 4, 40.95),
 (98762, "Programming Python, Mark Lutz", 5, 56.80),
 (77226, "Head First Python, Paul Barry", 3, 32.95),
 (88112, "Einführung in Python3, Bernd Klein", 3, 24.99)
]

order_total=list(map(lambda x: (x[0],x[2]*x[3] if x[2]*x[3]>=100 else x[2]*x[3]+10),dataset))
print(order_total)

temps_celsius = [0, 10, 20, 30, 40]
temp_farhenheit=list(map(lambda x:((x * 9/5) + 32),temps_celsius))
print(temp_farhenheit)

words = ["madam", "hello", "racecar", "world", "level"]
print(words)
pal=list(filter(lambda x:(x==x[::-1]),words))
print(pal)


input=[('English', 88), ('Science', 90), ('Maths', 97), ('Social sciences', 82)]
output=sorted(input,key=lambda x:x[0])
print(output)

from datetime import datetime

date_time=datetime.now()
print(date_time)
print(date_time.month)
print(date_time.year)
print(date_time.day)
print(date_time.time())

Students = [{'name': 'Zia', 'score': 85}, {'name': 'Trisha', 'score': 92}, {'name': "Alisha",
'score': 78}]
sorteds=sorted(Students,key=lambda x:x['score'])
print(sorteds)

from functools import reduce 
cart = [{'item': 'Laptop', 'price': 1000}, {'item': 'Phone', 'price': 500},
{'item': 'Headphones', 'price': 100}, {'item': 'Monitor', 'price': 500}, {'item':
'Mouse', 'price': 100}]

total=reduce(lambda x,y : x+y['price'],cart,0)
print(total)


students_scores = [{'id': 1, 'score': 85}, {'id': 2, 'score': 90}]
students_names = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]

merged = list(map(
    lambda s: {**s, **next(n for n in students_names if n['id'] == s['id'])},
    students_scores
))

print(merged)


num = 284
g=num
ls = []
while num != 0:
    ls.append(num % 10)
    num //= 10  # Corrected line to update num

ls=ls[::-1]
print(ls)
new=sorted(ls)
nw=new[::-1]
print(nw)

num = 0
for digit in nw:
    num = num * 10 + digit
print(num)

print(g==num)

og=['Red', 'Green', 'Blue', 'White', 'Black']
new=list(map(lambda x: x[::-1],og))
print(new)

nums = [3, 4, 5, 8, 0, 3, 8, 5, 0, 3, 1, 5, 2, 3, 4, 2]
count_dict = {num: nums.count(num) for num in set(nums)}
print(count_dict)

from functools import reduce
nums = [1, 2, 3, 4, 5]

pdt=reduce(lambda x,y:x*y,nums,1)
print(pdt)


list1=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list2=[2, 4, 6, 8]

filteredlist=list(filter(lambda x: x not in list2,list1))
print(filteredlist)

marks = [400, 450, 490, 300, 380]
per=list(map(lambda x:x*100/500,marks))
print(per)

top3=sorted(per)
print(top3[4],top3[3],top3[2])

fizzbuzz=list(map(lambda x: "Fizzbuzz" if x%3==0 and x%5==0 else "fizz" if x%3==0 else "Buzz" if x%5==0 else x ,range(0,21)))
print(fizzbuzz)

import numpy as np



numbers = [10, 20, 30, 40, 50]
total=np.sum(numbers)
print(total)
print(np.average(numbers))

filename="UserInput.txt"

with open(filename,'w') as file:
    
    while 1:
        
        line=input()
        if line=="0":
            break
        else:
            file.write(line +"\n")

with open ("UserInput.txt" ,'r') as file:
    content=file.read()
    count=space=0
    for i in content:
        if i==" ":
            count+=1
        elif i=="\n":
            space+=1
    print(content)
    print(count+space)

with open(filename,'a') as file:
    line=input()
    file.write("This Line Is appended without overwrite"+line)

filename="exp9.py"
try:
    with open(filename,'r'):
        print("File Opened Successfully")

except FileNotFoundError:
    print(FileNotFoundError)

src="UserInput.txt"
des="NewCopy.txt"

with open(src ,'r') as source:
    content=source.read()
    print(content)

    with open (des,'w') as destination:
        destination.write(content)


with open (des,'r') as file:
    content=file.read()

    print(content)
    lines=len(content)
    print(lines)
    space=line=char=1
    for i in content:
        if i==" ":
            space+=1
        elif i=="\n":
            line+=1
        else:
            char+=1
    print(line,line-1+space,char)

import csv
with open("housing_data.csv",'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(" | ".join(row)) 

word_to_search = "yaha"
with open(src, "r") as file:
 for index, line in enumerate(file, start=1):
  if word_to_search in line:
   print(f"Line {index}")

data=b"Hello sir"
with open("binaryfile.bin",'wb')as file:
    file.write(data)

with open("binaryfile.bin",'rb')as file:
    content=file.read()
    print(content)

import zipfile as zf
with zf.ZipFile("allfile.zip","w")as file:
    file.write(src)
    file.write(des)

with zf.ZipFile("allfile.zip","r")as file:
    file.extractall("Unzipped_files")


try:
 with open("no_file.txt", "r") as f:
  f.read()
except Exception as e:
 with open("error_log.txt", "a") as log:
  log.write(f"Error: {str(e)}\n")

import pickle

data_list = [1, 2, 3]
data_dict = {"a": 1, "b": 2}
data_set = {4, 5, 6}

with open("multipleObject.pkl",'wb')as file:
    pickle.dump(data_list,file)
    pickle.dump(data_dict,file)
    pickle.dump(data_set,file)


with open ("multipleObject.pkl",'rb')as file:
    list_data=pickle.load(file)
    dict_data=pickle.load(file)
    set_data=pickle.load(file)

    print(list_data,dict_data,set_data)

with open ("userfile.txt",'w')as file:
    while 1:
        data=input()
        if data=="STOP":
            break

        file.write(data+"\n")

with open("binaryfile.bin", 'rb') as file:
            binary_data = file.read()
            print(binary_data)
            hex_output = binary_data.hex()
            print("Hexadecimal representation of the file:")
            print(hex_output)



import pandas as pd

# Read the CSV file
df = pd.read_csv("adult.csv")




# Count the number of men and women
# gender_count = df['sex'].value_counts()
# print("\nNumber of men and women:")
# print(gender_count)

# average_age_women = df[df['sex'] == 'Female']['age'].mean()
# print(f"Average age of women: {average_age_women:.2f}")

# german_percentage = (df[df['native-country'] == 'Germany'].shape[0] / df.shape[0]) * 100
# print(f"Percentage of German citizens: {german_percentage:.2f}%")

# high_income = df[df['income'] == '>50K']
# low_income = df[df['income'] == '<=50K']

# high_income_age_mean=high_income["age"].mean()
# high_income_age_sd=high_income["age"].std()

# low_income_age_mean=low_income["age"].mean()
# low_income_age_sd=low_income["age"].std()

# print(f"\nFor those earning >50K:\nMean Age: {high_income_age_mean:.2f}, Standard Deviation: {high_income_age_sd:.2f}")
# print(f"For those earning <=50K:\nMean Age: {low_income_age_mean:.2f}, Standard Deviation: {low_income_age_sd:.2f}")

import numpy as np
from scipy import stats

from scipy import special

a=special.sindg(30)
print(a)



data=np.random.random(100)
print(data)

mean=stats.tmean(data)
print(mean)
var=stats.tvar(data)
print(var)
median=stats.median_abs_deviation(data)
print(median)

from scipy import linalg
matrix=np.array([[2, 1, 3],[1, 0, 2],[4, 1, 8]])
det=linalg.det(matrix)
inv=linalg.inv(matrix)
eigenvalues=linalg.eig(matrix)

print(matrix,det,inv,eigenvalues)


from scipy import interpolate
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 10)
y = np.sin(x)
f = interpolate.interp1d(x, y, kind='cubic')
xnew = np.linspace(0, 10, 100)
ynew = f(xnew)
plt.plot(x, y, 'o', label='Data points')
plt.plot(xnew, ynew, '-', label='Cubic Interpolation')
plt.legend()
plt.show()

from scipy.spatial import distance
point_A = (2, 3)
point_B = (5, 7)

dist = distance.euclidean(point_A,point_B)
print("Euclidean Distance between A and B:", dist)


import numpy as np
data = [12, 45, 67, 23, 89, 56, 90, 44]
mean = stats.tmean(data)
median = stats.scoreatpercentile(data, 50)
std_dev = stats.tstd(data)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)


import numpy as np
from scipy.linalg import solve
A = np.array([[2, 1],[3, 2]])
B = np.array([5, 12])

solution=solve(A,B)
print(solution)
x, y = solution
print(f"Solution:\nx = {x}\ny = {y}")

import numpy as np
coeffs = [2,-4, 5,-1]
roots = np.roots(coeffs)
print("Roots of the polynomial are:", roots)


import numpy as np
from scipy import linalg
A = np.array([[4, -2],[1, 1]])
eigenvalues, eigenvectors = linalg.eig(A)
print(eigenvalues)
print(eigenvectors)





import matplotlib.pyplot as plt

fig = plt.figure(figsize=(50,60))
ax = fig.add_subplot(1, 1, 1) # Add one Axes to the
ax.set_title("Sample Axes") 
plt.show()

x = [1, 2, 3, 4]
y = [2, 4, 1, 3]

plt.plot(x, y, marker='o')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Line Plot")
plt.show()

import pandas as pd
data=pd.read_csv("lineplot.csv")

plt.plot(data['Date'],data['Open'],marker='o')
plt.grid(1)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
products = ['A', 'B', 'C']
regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4']
sales = [[40, 50, 60], [55, 65, 75], [30, 40, 50], [70, 80, 90]]
sales = np.array(sales)
x = np.arange(len(products))
bar_width = 0.2
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
for i in range(4):
 plt.bar(x + i*bar_width, sales[i], width=bar_width, label=regions[i])
plt.xticks(x + bar_width*1.5, products)
plt.title("Vertical Bar Chart")
plt.legend()
plt.subplot(1, 2, 2)
for i in range(4):
 plt.barh(x + i*bar_width, sales[i], height=bar_width, label=regions[i])
plt.yticks(x + bar_width*1.5, products)
plt.title("Horizontal Bar Chart")
plt.legend()
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
students = ["Student 1", "Student 2", "Student 3"]
math = [75, 85, 90]
science = [80, 78, 92]
english = [88, 79, 84]
x = np.arange(len(students))
plt.bar(x, math, label="Math")
plt.bar(x, science, bottom=math, label="Science")
bottom_english = np.array(math) + np.array(science)
plt.bar(x, english, bottom=bottom_english, label="English")
plt.xticks(x, students)
plt.ylabel("Marks")
plt.title("Stackedbargrap")
plt.show()



import matplotlib.pyplot as plt
import numpy as np
brands = ['Apple', 'Samsung', 'Xiaomi', 'Oppo', 'Vivo']
market_share = [25, 35, 15, 15, 10]
plt.pie(market_share,labels=brands ,autopct='%d%%')
plt.title("Market Share of Smartphone Brands")
# plt.axis('equal')
plt.show()

histogram

scores = [56, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 87, 90, 92, 95, 97, 100]
plt.hist(scores, bins=5,edgecolor="black")
plt.title("Distribution of Exam Scores")
plt.xlabel("Score Range")
plt.ylabel("Number of Students")
plt.grid()
plt.show()


#Scatter Plot - Subject Marks Comparison:

math = [88, 92, 80, 89, 100, 80, 60, 100, 80, 34]
science = [35, 79, 79, 48, 100, 88, 32, 45, 20, 30]
plt.scatter(math, science, color='green')
plt.title("Math vs Science Marks")
plt.xlabel("Math")
plt.ylabel("Science")
plt.grid()
plt.show()

#Subplots - Multiple Visualizations:

temperature = [30, 32, 35, 33, 31, 29, 28, 30, 32, 35, 34, 33]
rainfall = [100, 120, 140, 150, 130, 100, 90, 80, 70, 75, 85, 95]
months = list(range(1, 13))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(months, temperature, marker='o')
plt.title("Monthly Temperature")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.subplot(1,2,2)
plt.bar(months, rainfall)
plt.title("Monthly Rainfall")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.tight_layout()
plt.show()

#3D Scatter Plot:

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
hours_studied = [1, 2, 3, 4, 5, 6, 7]
test_scores = [55, 60, 65, 70, 75, 80, 85]
sleep_duration = [8, 7, 6, 7, 8, 6, 7]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hours_studied, test_scores, sleep_duration, c='red', marker='o')
ax.set_xlabel('Hours Studied')
ax.set_ylabel('Test Scores')
ax.set_zlabel('Sleep Duration')
plt.title("3D Scatter Plot")
plt.show()

