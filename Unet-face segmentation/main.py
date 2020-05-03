from model import *
from data import *
import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet(pretrained_weights = 'unet.hdf5',model_type = 0)

model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=200,epochs=1,callbacks=[model_checkpoint])

acc = history.history['accuracy']
loss = history.history['loss']

epoch = range(1,len(acc)+1)
plt.plot(epoch,loss,'bo',label = "training_loss")
plt.plot(epoch,acc,'b',label = "accurancy")
plt.title('Training loss and accurancy')
plt.xlabel('epoch')
plt.ylabel('loss and acc')
plt.savefig("u-net-alter.png")

test_path = "data/membrane/test"
testList = []
for i in range(6):
    img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = False)
    img = img / 255
    testList.append(img)

results = model.predict([testList],verbose=1)

newResult = []
for item in results:
	item[item<=0.5] = 0
	item[item>0.5] = 255
	newResult.append(item)
print(newResult[0])

def savePic(save_path,newResult):
    for i,item in enumerate(newResult):
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),item)
savePic("data/membrane/test",newResult)
