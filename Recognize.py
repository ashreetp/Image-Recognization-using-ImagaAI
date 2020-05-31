

from imageai.Prediction.Custom import CustomImagePrediction

import os



execution_path = os.getcwd()



prediction = CustomImagePrediction()

prediction.setModelTypeAsResNet()

prediction.setModelPath("Identify.h5")

prediction.setJsonPath("Identify.json")


prediction.loadModel(num_objects=10)


predictions, probabilities = prediction.predictImage("test.jpg", result_count=3)



for eachPrediction, eachProbability in zip(predictions, probabilities):

    print(eachPrediction , " : " , eachProbability)