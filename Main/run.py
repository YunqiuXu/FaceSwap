# Modified by Yunqiu Xu
# Combined with GAN

import dlib
import cv2
import numpy as np
import models
import NonLinearLeastSquares
import ImageProcessing
import FaceRendering
import utils
import tensorflow as tf
import gan_func
from time import time

# Super resolution
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor

# Load the keypoint detection model, target image and 3D model
predictor_path = "shape_predictor_68_face_landmarks.dat"
image_name = "Trump.jpg"
# image_name = "JimCarrey.jpg"
# image_name = "JGL.jpeg"
maxImageSizeForDetection = 200 # if it's too small the face will not be detected
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("candide.npz")
idxs3D, idxs2D = utils.refine_idxs(idxs3D, idxs2D)
projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

# Init 3D model
cap = cv2.VideoCapture(0)
cameraImg = cap.read()[1]
textureImg = cv2.imread(image_name)
textureImg = cv2.resize(textureImg, (textureImg.shape[1] / 4, textureImg.shape[0] / 4))
cameraImg = cv2.resize(cameraImg, (cameraImg.shape[1] / 4, cameraImg.shape[0] / 4))
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

# Init GAN model
graph = gan_func.load_graph("frozen_model_500.pb")
image_tensor = graph.get_tensor_by_name('image_tensor:0')
output_tensor = graph.get_tensor_by_name('generate_output/output:0')
sess = tf.Session(graph=graph)

# Init SR model
sr_model = torch.load("model_scale_3_batch_4_epoch_500.pth")

# Start running
count = 0
while True:
    # Skip frame
    if count % 5 != 0:
        count += 1
        continue

    t0 = time()
    cameraImg = cap.read()[1]
    cameraImg = cv2.resize(cameraImg, (cameraImg.shape[1] / 4, cameraImg.shape[0] / 4))

    # Get 68 facial keypoints
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    if shapes2D is not None:
        for shape2D in shapes2D:

            # Mouth generating
            if count % 10 == 0:
                # Build input for GAN
                mouth_landmarks, mouth_leftup_coord, m_size= gan_func.getMouthKeypoints(shape2D, 64)
                combined_image = np.concatenate([mouth_landmarks, mouth_landmarks], axis=1)
                image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2RGB)

                try:
                    # Generate mouth
                    generated_image = sess.run(output_tensor, feed_dict={image_tensor: image_rgb})
                    generated_image = np.squeeze(generated_image) # 64 * 64 in RGB

                    # TEST
                    # print "1 passed"
                    # print generated_image.shape

                    # Super resolution
                    testImg = Image.fromarray(generated_image)
                    img = testImg.convert('YCbCr')
                    y, cb, cr = img.split()

                    # TEST
                    # print img
                    # print "2 passed"

                    sr_input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
                    out = sr_model(sr_input)
                    sr_output = out.cpu()

                    # TEST
                    # print "3 passed"

                    out_img_y = sr_output.data[0].numpy()
                    out_img_y *= 255.0
                    out_img_y = out_img_y.clip(0, 255)
                    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
                    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
                    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
                    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
                    out_img = np.array(out_img)
                    image_bgr = cv2.cvtColor(np.squeeze(out_img), cv2.COLOR_RGB2BGR)
                    image_bgr = cv2.resize(image_bgr, (image_bgr.shape[1] / 3, image_bgr.shape[0] / 3))

                    print image_bgr[1,1]
                    # Remap generated mouth to original size and position
                    remapped_mouth = gan_func.setMouth(cameraImg, image_bgr, mouth_leftup_coord, m_size) # Put generated mouth back to renderedImg

                    # Blend remapped mouth with original frame
                    mask_mouth = np.copy(remapped_mouth[:,:,0])
                    remapped_mouth = ImageProcessing.colorTransfer(cameraImg, remapped_mouth, mask_mouth)
                    cameraImg = ImageProcessing.blendImages(remapped_mouth, cameraImg, mask_mouth)

                    # TEST
                    print "SR for mouth is finished!"

                except:
                    pass
            else:
                try:
                    # Blend remapped mouth with original frame
                    # mask_mouth = np.copy(remapped_mouth[:,:,0])
                    # remapped_mouth = ImageProcessing.colorTransfer(cameraImg, remapped_mouth, mask_mouth)
                    cameraImg = ImageProcessing.blendImages(remapped_mouth, cameraImg, mask_mouth)
                except:
                    pass

            # Face rendering
            if count % 10 == 0:
                # Init params of 3D model, then perform optimisation
                modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])
                modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

                # Build 3D model for rendering
                shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)

                # Get rendered image
                renderedImg = renderer.render(shape3D)

                # Blend of rendered face with original frame
                mask_renderedImg = np.copy(renderedImg[:, :, 0])
                renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask_renderedImg)
                cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask_renderedImg)
            else:
                try:
                    cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask_renderedImg)
                except:
                    pass

            # Super resolution of the whole frame
            # testImg = cv2.cvtColor(cameraImg, cv2.COLOR_BGR2RGB)
            # testImg = Image.fromarray(testImg)
            # img = testImg.convert('YCbCr')
            # y, cb, cr = img.split()
            # input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
            # out = sr_model(input)
            # out = out.cpu()
            # out_img_y = out.data[0].numpy()
            # out_img_y *= 255.0
            # out_img_y = out_img_y.clip(0, 255)
            # out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
            # out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            # out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            # out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
            # cameraImg = np.array(out_img)
            # cameraImg = cv2.cvtColor(cameraImg, cv2.COLOR_RGB2BGR)

    t1 = time()    
    cv2.putText(cameraImg, 'FPS: '+str(1 / (t1 - t0))[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.imshow('image', cameraImg)
    count += 1
    key = cv2.waitKey(1)
    if key == 27:
        break

sess.close()
cv2.destroyAllWindows()
