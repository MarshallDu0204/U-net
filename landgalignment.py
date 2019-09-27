import numpy as np

match = 0
gap = 0
mismatch = 0

stringA = ""
stringB = ""


def readFile(path):
    with open(path,"r") as f:
        a = f.readlines()

def setValue(inMatch,inGap,inMismatch,inStringA,inStringB):
    global match,gap,mismatch
    global stringA,stringB

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
                elif matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                else:
                    resultAlignment[i,j] = "match"

            else:
                matrix[i][j] = max(up+gap,left+gap,diag+mismatch)
                if matrix[i][j] == up+gap:
                    resultAlignment[i,j] = "up"
                elif matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                else:
                    resultAlignment[i,j] = "mismatch"

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

    return resultA,resultB,resultShow


def exeGlobalAlignment():
    #readFile()
    setValue(1,-1,-1,"agta","ata")
    global stringA,stringB
    matrix = initMatrix(stringA,stringB) 
    matrix = initGlobalMatrix(matrix)


    resultA,resultB,resultShow = globalAlignment(matrix)
    print(resultA)
    print(resultB)
    print(resultShow)

#exeGlobalAlignment()

def localAlignment(matrix):
    
    global stringA,stringB
    global match,gap,mismatch
    resultAlignment = {}
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

                if matrix[i][j] == 0:
                    resultAlignment[i,j] = 0
                elif matrix[i][j] == up+gap:
                    resultAlignment[i,j] = "up"
                elif matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                else:
                    resultAlignment[i,j] = "match"

            else:
                matrix[i][j] = max(up+gap,left+gap,diag+mismatch,0)

                if matrix[i][j] == 0:
                    resultAlignment[i,j] = 0
                elif matrix[i][j] == up+gap:
                    resultAlignment[i,j] = "up"
                elif matrix[i][j] == left+gap:
                    resultAlignment[i,j] = "left"
                else:
                    resultAlignment[i,j] = "mismatch"

            j+=1
        i+=1

    print(matrix)
    print(resultAlignment)

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

    return resultA,resultB,resultShow,tempA,tempB
        


def exeLocalAlignment():
    #readFile()
    setValue(3,-2,-3,"tgttacgg","ggttgacta")
    global stringA,stringB
    matrix = initMatrix(stringA,stringB) 
    matrix = initLocalMatrix(matrix)
    localAlignment(matrix)
    
    resultA,resultB,resultShow,tempA,tempB = localAlignment(matrix)

