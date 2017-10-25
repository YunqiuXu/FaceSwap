# Modified by Yunqiu Xu
# Build  render dataset for gan

import dlib
import cv2
import numpy as np
import glob

import models
import NonLinearLeastSquares
import ImageProcessing
import FaceRendering
import utils


# Set pathes
predictor_path = "shape_predictor_68_face_landmarks.dat"
file_names = sorted(glob.glob("/home/venturer/COMP9517/generate_gan_dataset/img_dataset/*"))
mask_path = "/home/venturer/COMP9517/generate_gan_dataset/mask/"
rendered_path = "/home/venturer/COMP9517/generate_gan_dataset/rendered/"

# Initialize pretrained models
image_name = file_names[0]
maxImageSizeForDetection = 500
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("candide.npz")
# idxs3D, idxs2D = utils.refine_idxs(idxs3D, idxs2D) # useless

# Initialize 3d model
projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])
modelParams = None
cameraImg = cv2.imread(image_name)
textureImg = cv2.imread(image_name)

# you can resize the image to improve performance
textureImg = cv2.resize(textureImg, (textureImg.shape[1] / 2, textureImg.shape[0] / 2))
cameraImg = cv2.resize(cameraImg, (cameraImg.shape[1] / 2, cameraImg.shape[0] / 2))
print "Resized shape : " + str(cameraImg.shape) 

textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)


for image_name in file_names:

    # load a new image
    cameraImg = cv2.imread(image_name)
    # resize 
    cameraImg = cv2.resize(cameraImg, (cameraImg.shape[1] / 2, cameraImg.shape[0] / 2))
    # get facial keypoints
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    if shapes2D is not None:
        shape2D = shapes2D[0]
        # 3D model parameter initialization
        modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

        # 3D model parameter optimization
        modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

        # rendering the model to an image
        shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams) # np.darray
        renderedImg = renderer.render(shape3D)

        cv2.imshow('rendered image', renderedImg)
        write_name = rendered_path + image_name[-11:]
        print "Write into " + write_name
        cv2.imwrite(write_name, renderedImg)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
