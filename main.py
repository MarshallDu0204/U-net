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

model = unet(model_type = 1)
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=300,epochs=5,callbacks=[model_checkpoint])

acc = history.history['accuracy']
loss = history.history['loss']

epoch = range(1,len(acc)+1)
plt.plot(epoch,loss,'bo',label = "training_loss")
plt.plot(epoch,acc,'b',label = "accurancy")
plt.title('Training loss and accurancy')
plt.xlabel('epoch')
plt.ylabel('loss and acc')
plt.savefig("u-net-alter.png")


testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)