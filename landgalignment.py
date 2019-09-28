import numpy as np
from tkinter import *
import tkinter.font as tkFont
import tkinter.filedialog

match = 0
gap = 0
mismatch = 0

option = ""

stringA = ""
stringB = ""

path = ""

count = 1

def setPath(inPath):
    global path
    path = inPath

def readFile(path):
    match = 0
    gap = 0
    mismatch = 0

    seqA = ""
    seqB = ""

    try:
        with open(path,"r") as f:
            file = f.readlines()

            option = file[0]
            option = option.strip()


            data = file[1]
            seqA = file[2]
            seqB = file[3]
            
            data = data.split(" ")


            match = int(data[0])
            gap = int(data[1])
            mismatch = int(data[2])

            seqA = seqA.strip()
            seqA = str(seqA)
            seqB = seqB.strip()
            seqB = str(seqB)
    except:
        print("error")
    

    return match,gap,mismatch,seqA,seqB,option


def setValue(inMatch,inGap,inMismatch,inStringA,inStringB,inOption):
    global match,gap,mismatch
    global stringA,stringB
    global option

    option = inOption

    match = inMatch
    gap = inGap
    mismatch = inMismatch
    stringA = inStringA
    stringB = inStringB


def initMatrix(a,b):#a and b is two string input from the file
   x = len(a)+1
   y = len(b)+1
   matrix = np.zeros((y,x))
   return matrix



def showMatrix(matrix):
    print("------------------")
    print(matrix)
    print("------------------")



def initGlobalMatrix(matrix):
    global gap

    for i in range(1,len(matrix)):
        matrix[i][0] = matrix[i-1][0]+gap

    for j in range(1,len(matrix[0])):
        matrix[0][j] = matrix[0][j-1]+gap

    return matrix


def initLocalMatrix(matrix):

    for i in range(len(matrix)):
        matrix[i][0] = 0

    for j in range(len(matrix[0])):
        matrix[0][j] = 0

    return matrix


def globalAlignment(matrix):

    global stringA,stringB
    global match,gap,mismatch
    resultAlignment = {}
    alignmentTable = []
    seqA = stringA
    seqB = stringB

    i=1
    while i!=len(matrix):
        j=1
        while j!=len(matrix[0]):

            up = matrix[i-1][j]
            left = matrix[i][j-1]
            diag = matrix[i-1][j-1]

            y = i-1
            x = j-1

            if seqA[x] == seqB[y]:
                
                matrix[i][j] = max(up+gap,left+gap,diag+match)
                if matrix[i][j] == up+gap:
                    resultAlignment[i,j] = "up"
                    alignmentTable.append([i,j,"up"])
                if matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                    alignmentTable.append([i,j,"left"])       
                if matrix[i][j] == diag+match:
                    resultAlignment[i,j] = "match"
                    alignmentTable.append([i,j,"diag"])
                

            else:
            
                matrix[i][j] = max(up+gap,left+gap,diag+mismatch)
                if matrix[i][j] == up+gap:
                    resultAlignment[i,j] = "up"
                    alignmentTable.append([i,j,"up"])
                if matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                    alignmentTable.append([i,j,"left"])
                if matrix[i][j] == diag+mismatch:
                    resultAlignment[i,j] = "mismatch"
                    alignmentTable.append([i,j,"diag"])
            
            j+=1
        i+=1

    i = len(matrix)-1
    j = len(matrix[0])-1

    arrowMove = []
    
    while i!=0 or j!=0:
        if resultAlignment[i,j] == "match":
            i-=1
            j-=1
            arrowMove.append("match")
        elif resultAlignment[i,j] == "mismatch":
            i-=1
            j-=1
            arrowMove.append("mismatch")
        elif resultAlignment[i,j] == "up":
            i-=1
            arrowMove.append("up")
        else:
            j-=1
            arrowMove.append("left")


    resultA = ""
    resultB = ""
    resultShow = ""

    i=0
    j=0
    x = len(arrowMove)-1
    while x!=-1:
        
        if arrowMove[x] == "match":
            resultA = resultA+str(seqA[j])
            resultB = resultB+str(seqB[i])
            resultShow = resultShow+"m"
            i+=1
            j+=1
        elif arrowMove[x] == "mismatch":
            resultA = resultA+str(seqA[j])
            resultB = resultB+str(seqB[i])
            resultShow = resultShow+"s"
            i+=1
            j+=1
        elif arrowMove[x] == "left":
            resultA = resultA+str(seqA[j])
            resultB = resultB+"-"
            resultShow = resultShow+"d"
            j+=1
        else:
            resultA = resultA+"-"
            resultB = resultB+str(seqB[i])
            resultShow = resultShow+"d"
            i+=1
        
        x-=1
    pathCont = 0
    i = len(matrix)-1
    j = len(matrix[0])-1
    pathCont = alignmentCont([i,j],alignmentTable)
    optimal = matrix[len(matrix)-1][len(matrix[0])-1]
    print(matrix)
    return resultA,resultB,resultShow,optimal,pathCont




def localAlignment(matrix):
    
    global stringA,stringB
    global match,gap,mismatch
    resultAlignment = {}
    alignmentTable = []
    seqA = stringA
    seqB = stringB

    i=1
    while i!=len(matrix):
        j=1
        while j!=len(matrix[0]):

            up = matrix[i-1][j]
            left = matrix[i][j-1]
            diag = matrix[i-1][j-1]

            y = i-1
            x = j-1

            if seqA[x] == seqB[y]:
                matrix[i][j] = max(up+gap,left+gap,diag+match,0)
       
                if matrix[i][j] == up+gap:
                    resultAlignment[i,j] = "up"
                    alignmentTable.append([i,j,"up"])
                if matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                    alignmentTable.append([i,j,"left"])
                if matrix[i][j] == diag+match:
                    resultAlignment[i,j] = "match"
                    alignmentTable.append([i,j,"diag"])
                if matrix[i][j] == 0:
                    resultAlignment[i,j] = 0

            else:
                matrix[i][j] = max(up+gap,left+gap,diag+mismatch,0)

                if matrix[i][j] == up+gap:
                    resultAlignment[i,j] = "up"
                    alignmentTable.append([i,j,"up"])
                if matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                    alignmentTable.append([i,j,"left"])
                if matrix[i][j] == diag+mismatch:
                    resultAlignment[i,j] = "mismatch"
                    alignmentTable.append([i,j,"diag"])
                if matrix[i][j] == 0:
                    resultAlignment[i,j] = 0

            j+=1
        i+=1

    maxVlue = 0
    indexX = 0
    indexY = 0
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if matrix[x][y]>maxVlue:
                maxVlue = matrix[x][y]
                indexX = x
                indexY = y

    i= indexX
    j = indexY

    endPos = [indexX,indexY]

    arrowMove = []

    while matrix[i,j]!=0:
        if resultAlignment[i,j] == "match":
            i-=1
            j-=1
            arrowMove.append("match")
        elif resultAlignment[i,j] == "mismatch":
            i-=1
            j-=1
            arrowMove.append("mismatch")
        elif resultAlignment[i,j] == "up":
            i-=1
            arrowMove.append("up")
        else:
            j-=1
            arrowMove.append("left")

    startPos = [i,j]

    i = startPos[0]
    j = startPos[1]

    resultA = ""
    resultB = ""
    resultShow = ""


    x = len(arrowMove)-1
    while x!=-1:
        
        if arrowMove[x] == "match":
            resultA = resultA+str(seqA[j])
            resultB = resultB+str(seqB[i])
            resultShow = resultShow+"m"
            i+=1
            j+=1
        elif arrowMove[x] == "mismatch":
            resultA = resultA+str(seqA[j])
            resultB = resultB+str(seqB[i])
            resultShow = resultShow+"s"
            i+=1
            j+=1
        elif arrowMove[x] == "left":
            resultA = resultA+str(seqA[j])
            resultB = resultB+"-"
            resultShow = resultShow+"d"
            j+=1
        else:
            resultA = resultA+"-"
            resultB = resultB+str(seqB[i])
            resultShow = resultShow+"d"
            i+=1
        
        x-=1

    tempA = ""
    for i in range(len(seqA)):
        if i == startPos[1]:
            tempA = tempA+"("
        if i == endPos[1]:
            tempA = tempA+")"
        tempA = tempA+seqA[i]
        

    tempB = ""
    for i in range(len(seqB)):
        if i == startPos[0]:
            tempB = tempB+"("
        if i == endPos[0]:
            tempB = tempB+")"
        tempB = tempB+seqB[i]

    pathCont = 0
    i = indexX
    j = indexY

    optimal = matrix[i][j]
    pathCont = alignmentCont([i,j],alignmentTable)

    
    return resultA,resultB,resultShow,tempA,tempB,optimal,pathCont
        
def alignmentCont(start,alignmentTable):
    path = []
    i = start[0]
    j = start[1]
    temp = []
    temp.append([i,j])
    path.append(temp)
    while checkFinish(alignmentTable,path):
        for i in range(len(path)):
            last = path[i][len(path[i])-1]
            result = searchTable(alignmentTable,last[0],last[1])
            if len(result)==1:
                result = result[0]
                if result == "left":
                    path[i].append([last[0],last[1]-1])
                elif result == "diag":
                    path[i].append([last[0]-1,last[1]-1])
                else:
                    path[i].append([last[0]-1,last[1]])
            elif len(result)==2:
                if result[0] == "left":
                    path[i].append([last[0],last[1]-1])
                if result[0] == "diag":
                    path[i].append([last[0]-1,last[1]-1])
                if result[0] == "up":
                    path[i].append([last[0]-1,last[1]])
                temp = []
                if result[1] == "left":
                    temp.append([last[0],last[1]-1])
                if result[1] == "diag":
                    temp.append([last[0]-1,last[1]-1])
                if result[1] == "up":
                    temp.append(last[0]-1,last[1])
                path.append(temp)

    return len(path)  

def checkFinish(alignmentTable,path):
    for element in path:
        last = element[len(element)-1]
        result = searchTable(alignmentTable,last[0],last[1])
        if len(result)>0:
            return True
    return False


def searchTable(alignmentTable,i,j):
    result = []
    for element in alignmentTable:
        if element[0] == i and element[1]==j:
            result.append(element[2])
    return result

def toString(resultA,resultB,resultShow,optimal,pathCont,wholeStringA = "",wholeStringB = ""):
    global match,gap,mismatch

    global stringA,stringB

    answer = ""
    if wholeStringA != "" and wholeStringB != "":
        answer = ""
        answer = "localAlignment"+"\n"
        answer = answer+"match: "+str(match)+"\n"
        answer = answer+"gap: "+str(gap)+"\n"
        answer = answer+"mismatch: "+str(mismatch)+"\n"
        answer = answer+"originSeqA: "+stringA+"\n"
        answer = answer+"originSeqB: "+stringB+"\n"

        answer = answer+"--------------\n"

        answer = answer+"wholeStringA: "+wholeStringA+"\n"
        answer = answer+"wholeStringB: "+wholeStringB+"\n"
        answer = answer+"alignSeqA: "+resultA+"\n"
        answer = answer+"alignSeqB: "+resultB+"\n"
        answer = answer+"spec:      "+resultShow+"\n"
        answer = answer+"optimal score: "+str(optimal)+"\n"
        answer = answer+"number of solution: "+str(pathCont)+"\n"

    else:
        
        answer = ""
        answer = "globalAlignment"+"\n"
        answer = answer+"match: "+str(match)+"\n"
        answer = answer+"gap: "+str(gap)+"\n"
        answer = answer+"mismatch: "+str(mismatch)+"\n"
        answer = answer+"originSeqA: "+stringA+"\n"
        answer = answer+"originSeqB: "+stringB+"\n"

        answer = answer+"--------------\n"

        answer = answer+"alignSeqA: "+resultA+"\n"
        answer = answer+"alignSeqB: "+resultB+"\n"
        answer = answer+"spec:      "+resultShow+"\n"
        answer = answer+"optimal score: "+str(optimal)+"\n"
        answer = answer+"number of solution: "+str(pathCont)+"\n"

    return answer


def filePicker():
    filename = tkinter.filedialog.askopenfilename()
    if filename != "":
        lb.config(text = "The File you choose is ："+filename)
        setPath(filename)
    else:
        lb.config(text = "No file select")


def startAlignment():
    global path
    match,gap,mismatch,seqA,seqB,option = readFile(path)
    setValue(match,gap,mismatch,seqA,seqB,option)
    global stringA,stringB
    matrix = initMatrix(stringA,stringB) 
    
    if option == "l":
        matrix = initLocalMatrix(matrix)
        resultA,resultB,resultShow,tempA,tempB,optimal,pathCont = localAlignment(matrix)
        answer = toString(resultA,resultB,resultShow,optimal,pathCont,wholeStringA = tempA,wholeStringB = tempB)
        var.set(answer)

    else:
        matrix = initGlobalMatrix(matrix)
        resultA,resultB,resultShow,optimal,pathCont = globalAlignment(matrix)
        answer = toString(resultA,resultB,resultShow,optimal,pathCont)
        var.set(answer)


root = Tk()
root.title("Sequence Alignment")
root.geometry('800x600')
var = StringVar()
btn = Button(root,text="Choose your file",command=filePicker)
btn.pack()
lb = Label(root,text = '')
lb.pack()
com = Button(root,text = 'startAlignment', command = startAlignment) 
com.pack() 
ft = tkFont.Font(family='Fixdsys', size=20, weight=tkFont.BOLD)
label = Label(root,textvariable = var,width = 300,height = 200,bg='white',justify = 'left',font=ft)
label.pack()

root.mainloop()


