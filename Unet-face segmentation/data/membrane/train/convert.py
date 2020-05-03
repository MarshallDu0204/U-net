import cv2
import os.path
import glob
def convertjpg(jpgfile,outdir,i):
	img=cv2.imread(jpgfile)
	try:
		new_img= cv2.resize(img,(256,256),interpolation = cv2.INTER_CUBIC)
		savePath = 'test1/'+str(i)+'.png'
		cv2.imwrite(savePath,new_img)
	except Exception as e:
		print(e)
dirList = os.listdir('C:/Users/24400/Desktop/bioinformatics-master/data/membrane/test/')
i = 0
for jpgfile in dirList:
	jpgfile = 'C:/Users/24400/Desktop/bioinformatics-master/data/membrane/test/'+jpgfile
	print(jpgfile)
	convertjpg(jpgfile,"test1",i)
	i+=1