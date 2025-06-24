# ✨ Sign Language Detection System

📋 **Project Description**

This project is a Sign Language Detection System that can:

-Detect and classify sign language gestures from images and real-time webcam feed.

-Operate only between a restricted timeframe.

-Provide a simple GUI built with Tkinter that allows:
  
   -Uploading an image for prediction.

   -Starting and stopping webcam-based real-time detection.

The system uses MediaPipe for hand detection and a fine-tuned MobileNetV2 model for gesture classification.

🔨 **Project Features**


-Real-time hand detection using MediaPipe.

-Gesture classification using a pretrained MobileNetV2 model.

-Confidence-based prediction with thresholding.

-GUI with:

   -Upload image button.
  
   -Start/stop webcam buttons.

-Automatic time validation to restrict system operation between specified hours.


🖼️**Dataset Preparation**

Data Collection:

   -Collected real-time images with full background using OpenCV.

   -Used MediaPipe to detect and crop hands from each image on the fly (without saving cropped versions).


Data Augmentation:

   -Applied augmentations (flipping, rotation, brightness changes) to expand to 200 images per class.


Data Splitting:

   -Dataset split into train and test folders using an 80:20 ratio.

🧠 **Model Training**

-Used MobileNetV2 pretrained on ImageNet.

-Fine-tuned the model on the collected dataset.

-Applied ReduceLROnPlateau for learning rate scheduling.

-Focused fine-tuning on classes with confusion like yes, no, and stop.

🎯 **Model Evaluation**

-Achieved ~98% training and validation accuracy.

-Observed class-wise confusion (especially yes, no, stop) and corrected it by:

-Collecting additional real-time images for these classes.

-Fine-tuning the model on this focused dataset.

💻 **GUI Features**

-Built using Tkinter.

-Buttons:

   -Upload Image: Select an image for hand detection and sign prediction.

   -Start Webcam: Start real-time video prediction.

   -Stop Webcam: Stop video feed.

-Displays model predictions with confidence scores.
