Files:
|-main.py : Python file to detect object and door(method two), Door Open/Close and distance calculation
|-Doorwithknobs.py : Python file to door and knobs(method one), Door Open/Close and distance calculation
|-calibrationFinal*.mat : Calibration session for calculating camera intrisic and extrinsic parameter for left and right camera
|-*Tr.mat : calculated camera intrisic and extrinsic parameter for left and right camera
|-Viola_Jones : This folder contains training images, model and execution for outdoor door detection.
     |- model : Model cascade.xml
     |- outdoor_detect_doors.py: Executable
     |- model_train_files : Model train data
          |- Positive Samples: Cropped door images
          |- Negative Samples: background images
          |- vec File: Sampled features for Viola-Jones classifier



How to execute:
1)Door open/close with obstacle detection

$python main.py <door/obs>
door-> door detection
obs -> Obstacle detection

    Eg. $python main.py door
	
2)Door and knob detection

$python Doorwithknob.py

3)Outdoor door detection

$python outdoor_detect_doors.py


Note:
A video demo of the project is available in the repository.(vision_final.mp4)
